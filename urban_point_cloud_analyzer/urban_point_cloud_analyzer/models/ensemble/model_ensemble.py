# urban_point_cloud_analyzer/models/ensemble/model_ensemble.py
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

class SegmentationEnsemble(nn.Module):
    """
    Ensemble of segmentation models.
    """
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        """
        Initialize segmentation ensemble.
        
        Args:
            models: List of segmentation models
            weights: Optional list of model weights (must sum to 1)
        """
        super(SegmentationEnsemble, self).__init__()
        
        self.models = nn.ModuleList(models)
        
        # Set weights or use equal weighting
        if weights is None:
            self.weights = torch.ones(len(models)) / len(models)
        else:
            assert len(weights) == len(models), "Number of weights must match number of models"
            assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
            self.weights = torch.tensor(weights)
    
    def forward(self, points: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            points: (B, N, 3) tensor of point coordinates
            features: (B, N, C) tensor of point features (optional)
            
        Returns:
            (B, num_classes, N) tensor of per-point logits
        """
        # Run all models
        outputs = []
        
        for i, model in enumerate(self.models):
            with torch.cuda.amp.autocast(enabled=True):
                output = model(points, features)
                outputs.append(output)
        
        # Apply softmax to get probabilities
        probs = []
        for output in outputs:
            prob = torch.softmax(output, dim=1)
            probs.append(prob)
        
        # Weight and combine probabilities
        weighted_probs = torch.zeros_like(probs[0])
        for i, prob in enumerate(probs):
            weight = self.weights[i]
            weighted_probs += prob * weight
        
        # Convert back to logits (optional, depending on loss function)
        # weighted_logits = torch.log(weighted_probs + 1e-10)
        
        return weighted_probs
    
    def predict(self, points: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Make predictions.
        
        Args:
            points: (B, N, 3) tensor of point coordinates
            features: (B, N, C) tensor of point features (optional)
            
        Returns:
            (B, N) tensor of per-point class predictions
        """
        # Get weighted probabilities
        weighted_probs = self.forward(points, features)
        
        # Get class with highest probability
        predictions = torch.argmax(weighted_probs, dim=1)
        
        return predictions


class MultiScaleEnsemble(nn.Module):
    """
    Multi-scale ensemble for segmentation.
    Processes point cloud at multiple scales and combines predictions.
    """
    
    def __init__(self, 
                 base_model: nn.Module, 
                 scales: List[float] = [0.5, 1.0, 2.0],
                 weights: Optional[List[float]] = None):
        """
        Initialize multi-scale ensemble.
        
        Args:
            base_model: Base segmentation model
            scales: List of scales to process point cloud at
            weights: Optional list of scale weights (must sum to 1)
        """
        super(MultiScaleEnsemble, self).__init__()
        
        self.base_model = base_model
        self.scales = scales
        
        # Set weights or use equal weighting
        if weights is None:
            self.weights = torch.ones(len(scales)) / len(scales)
        else:
            assert len(weights) == len(scales), "Number of weights must match number of scales"
            assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
            self.weights = torch.tensor(weights)
    
    def forward(self, points: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            points: (B, N, 3) tensor of point coordinates
            features: (B, N, C) tensor of point features (optional)
            
        Returns:
            (B, num_classes, N) tensor of per-point logits
        """
        # Process at multiple scales
        outputs = []
        
        for i, scale in enumerate(self.scales):
            # Scale points
            scaled_points = points.clone()
            scaled_points[:, :, :3] = scaled_points[:, :, :3] * scale
            
            # Process scaled points
            with torch.cuda.amp.autocast(enabled=True):
                output = self.base_model(scaled_points, features)
                outputs.append(output)
        
        # Apply softmax to get probabilities
        probs = []
        for output in outputs:
            prob = torch.softmax(output, dim=1)
            probs.append(prob)
        
        # Weight and combine probabilities
        weighted_probs = torch.zeros_like(probs[0])
        for i, prob in enumerate(probs):
            weight = self.weights[i]
            weighted_probs += prob * weight
        
        return weighted_probs
    
    def predict(self, points: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Make predictions.
        
        Args:
            points: (B, N, 3) tensor of point coordinates
            features: (B, N, C) tensor of point features (optional)
            
        Returns:
            (B, N) tensor of per-point class predictions
        """
        # Get weighted probabilities
        weighted_probs = self.forward(points, features)
        
        # Get class with highest probability
        predictions = torch.argmax(weighted_probs, dim=1)
        
        return predictions


class SpecialistEnsemble(nn.Module):
    """
    Ensemble of specialist models, each focused on specific classes.
    """
    
    def __init__(self, 
                 models: List[nn.Module], 
                 class_assignments: List[List[int]],
                 num_classes: int):
        """
        Initialize specialist ensemble.
        
        Args:
            models: List of specialist models
            class_assignments: List of class assignments for each model
            num_classes: Total number of classes
        """
        super(SpecialistEnsemble, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.class_assignments = class_assignments
        self.num_classes = num_classes
        
        # Validate class assignments
        assert len(models) == len(class_assignments), "Number of models must match number of class assignments"
        
        # Create a mapping from class to models
        self.class_to_models = [[] for _ in range(num_classes)]
        for i, classes in enumerate(class_assignments):
            for cls in classes:
                self.class_to_models[cls].append(i)
    
    def forward(self, points: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            points: (B, N, 3) tensor of point coordinates
            features: (B, N, C) tensor of point features (optional)
            
        Returns:
            (B, num_classes, N) tensor of per-point logits
        """
        batch_size = points.shape[0]
        num_points = points.shape[1]
        device = points.device
        
        # Process with all models
        model_outputs = []
        
        for model in self.models:
            with torch.cuda.amp.autocast(enabled=True):
                output = model(points, features)
                model_outputs.append(output)
        
        # Initialize output probabilities
        output_probs = torch.zeros((batch_size, self.num_classes, num_points), device=device)
        
        # Combine model outputs based on specialist areas
        for cls in range(self.num_classes):
            if not self.class_to_models[cls]:
                continue
            
            # Average predictions from models that handle this class
            cls_probs = torch.zeros((batch_size, num_points), device=device)
            count = 0
            
            for model_idx in self.class_to_models[cls]:
                # Get probability for this class
                model_output = model_outputs[model_idx]
                cls_prob = torch.softmax(model_output, dim=1)[:, cls, :]
                cls_probs += cls_prob
                count += 1
            
            if count > 0:
                output_probs[:, cls, :] = cls_probs / count
        
        # Normalize to ensure valid probabilities
        output_probs = output_probs / (output_probs.sum(dim=1, keepdim=True) + 1e-10)
        
        return output_probs
    
    def predict(self, points: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Make predictions.
        
        Args:
            points: (B, N, 3) tensor of point coordinates
            features: (B, N, C) tensor of point features (optional)
            
        Returns:
            (B, N) tensor of per-point class predictions
        """
        # Get probabilities
        probs = self.forward(points, features)
        
        # Get class with highest probability
        predictions = torch.argmax(probs, dim=1)
        
        return predictions