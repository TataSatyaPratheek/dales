# urban_point_cloud_analyzer/models/segmentation/multiscale_feature_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

class MultiScaleFeatureFusion(nn.Module):
    """
    Multi-scale feature fusion module that combines features
    from different scales for better segmentation.
    """
    def __init__(self, 
                feature_dims: List[int],
                output_dim: int,
                dropout: float = 0.5,
                use_checkpoint: bool = True):
        super(MultiScaleFeatureFusion, self).__init__()
        
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        self.use_checkpoint = use_checkpoint
        
        # MLP layers to align feature dimensions
        self.mlp_layers = nn.ModuleList()
        
        for dim in feature_dims:
            self.mlp_layers.append(
                nn.Sequential(
                    nn.Linear(dim, output_dim),
                    nn.BatchNorm1d(output_dim),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Attention weights to combine features
        self.attention = nn.Sequential(
            nn.Linear(output_dim * len(feature_dims), len(feature_dims)),
            nn.Softmax(dim=2)
        )
        
        # Final MLP
        self.final_mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
    def _fuse_features(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse features from multiple scales.
        
        Args:
            features_list: List of (B, N, C) feature tensors
            
        Returns:
            (B, N, output_dim) fused feature tensor
        """
        batch_size, num_points, _ = features_list[0].shape
        
        # Align feature dimensions
        aligned_features = []
        
        for i, features in enumerate(features_list):
            # Reshape for batch norm
            features_flat = features.reshape(-1, features.shape[-1])
            aligned = self.mlp_layers[i](features_flat)
            aligned = aligned.reshape(batch_size, num_points, -1)
            aligned_features.append(aligned)
        
        # Concatenate aligned features
        concat_features = torch.cat(aligned_features, dim=2)
        
        # Calculate attention weights
        attention_weights = self.attention(concat_features)  # (B, N, num_scales)
        
        # Apply attention weights
        fused_features = torch.zeros(
            (batch_size, num_points, self.output_dim),
            device=features_list[0].device
        )
        
        for i, aligned in enumerate(aligned_features):
            # Extract weight for this scale
            weight = attention_weights[:, :, i].unsqueeze(2)  # (B, N, 1)
            
            # Apply weight
            fused_features += aligned * weight
        
        # Apply final MLP
        fused_features_flat = fused_features.reshape(-1, self.output_dim)
        fused_features_flat = self.final_mlp(fused_features_flat)
        fused_features = fused_features_flat.reshape(batch_size, num_points, -1)
        
        return fused_features
    
    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features_list: List of (B, N, C) feature tensors
            
        Returns:
            (B, N, output_dim) fused feature tensor
        """
        # Check that we have the right number of features
        assert len(features_list) == len(self.feature_dims), \
            f"Expected {len(self.feature_dims)} feature tensors, got {len(features_list)}"
        
        # Check that feature dimensions match
        for i, (features, dim) in enumerate(zip(features_list, self.feature_dims)):
            assert features.shape[2] == dim, \
                f"Feature tensor {i} has dimension {features.shape[2]}, expected {dim}"
        
        # Use gradient checkpointing if requested to save memory
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(self._fuse_features, features_list)
        else:
            return self._fuse_features(features_list)


class MultiScaleSegmentationModel(nn.Module):
    """
    Multi-scale segmentation model that combines features from
    different scales for better segmentation.
    """
    def __init__(self, 
                 base_model: nn.Module,
                 scales: List[float] = [0.5, 1.0, 2.0],
                 output_dim: int = 128,
                 num_classes: int = 8,
                 dropout: float = 0.5,
                 use_checkpoint: bool = True):
        """
        Initialize multi-scale segmentation model.
        
        Args:
            base_model: Base segmentation model
            scales: List of scales to process point cloud at
            output_dim: Dimension of output features
            num_classes: Number of segmentation classes
            dropout: Dropout probability
            use_checkpoint: Whether to use gradient checkpointing
        """
        super(MultiScaleSegmentationModel, self).__init__()
        
        self.base_model = base_model
        self.scales = scales
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        
        # Determine feature dimensions from base model
        # This assumes the base model returns logits with shape (B, C, N)
        self.feature_dims = [num_classes] * len(scales)
        
        # Multi-scale feature fusion
        self.fusion = MultiScaleFeatureFusion(
            feature_dims=self.feature_dims,
            output_dim=output_dim,
            dropout=dropout,
            use_checkpoint=use_checkpoint
        )
        
        # Final segmentation layer
        self.segmentation_head = nn.Linear(output_dim, num_classes)
    
    def _process_at_scale(self, points: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Process point cloud at a specific scale.
        
        Args:
            points: (B, N, 3) tensor of point coordinates
            scale: Scale factor
            
        Returns:
            (B, C, N) tensor of features
        """
        # Scale points
        scaled_points = points.clone()
        scaled_points[:, :, :3] = scaled_points[:, :, :3] * scale
        
        # Process with base model
        return self.base_model(scaled_points)
    
    def forward(self, points: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            points: (B, N, 3) tensor of point coordinates
            features: Optional (B, N, C) tensor of point features
            
        Returns:
            (B, num_classes, N) tensor of logits
        """
        batch_size, num_points, _ = points.shape
        
        # Process at multiple scales
        multi_scale_features = []
        
        for scale in self.scales:
            # Use gradient checkpointing for scale processing
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                scale_features = torch.utils.checkpoint.checkpoint(
                    self._process_at_scale, points, scale
                )
            else:
                scale_features = self._process_at_scale(points, scale)
            
            # Convert to (B, N, C) format for fusion
            scale_features = scale_features.transpose(1, 2).contiguous()
            multi_scale_features.append(scale_features)
        
        # Fuse multi-scale features
        fused_features = self.fusion(multi_scale_features)
        
        # Apply segmentation head
        # Reshape for batch norm
        fused_features_flat = fused_features.reshape(-1, fused_features.shape[-1])
        logits_flat = self.segmentation_head(fused_features_flat)
        logits = logits_flat.reshape(batch_size, num_points, self.num_classes)
        
        # Transpose to (B, C, N) format for consistency with other models
        logits = logits.transpose(1, 2).contiguous()
        
        return logits
    
    def get_loss(self, logits: torch.Tensor, labels: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate loss.
        
        Args:
            logits: (B, num_classes, N) tensor of per-point logits
            labels: (B, N) tensor of point labels
            weights: Optional (num_classes,) tensor of class weights
            
        Returns:
            Loss tensor
        """
        # Reshape logits for cross-entropy loss
        logits = logits.transpose(1, 2).contiguous()  # (B, N, num_classes)
        batch_size, num_points, _ = logits.shape
        
        # Flatten predictions and labels
        logits = logits.reshape(-1, self.num_classes)
        labels = labels.reshape(-1)
        
        # Calculate cross-entropy loss
        if weights is not None:
            loss = F.cross_entropy(logits, labels, weight=weights)
        else:
            loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def predict(self, points: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Make predictions.
        
        Args:
            points: (B, N, 3) tensor of point coordinates
            features: Optional (B, N, C) tensor of point features
            
        Returns:
            (B, N) tensor of per-point predictions
        """
        logits = self.forward(points, features)
        predictions = torch.argmax(logits, dim=1)
        return predictions