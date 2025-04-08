# urban_point_cloud_analyzer/training/loss_functions/__init__.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union

def get_loss_function(config: Dict) -> nn.Module:
    """
    Create a loss function based on configuration.
    
    Args:
        config: Loss configuration
        
    Returns:
        Loss function
    """
    loss_type = config.get('segmentation', 'cross_entropy')
    
    if loss_type == 'cross_entropy':
        return SegmentationLoss(weight=config.get('class_weights', None))
    elif loss_type == 'focal':
        return FocalLoss(
            gamma=config.get('focal_gamma', 2.0),
            alpha=config.get('focal_alpha', None)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

class SegmentationLoss(nn.Module):
    """Cross entropy loss with optional class weights for segmentation."""
    def __init__(self, weight: Optional[torch.Tensor] = None):
        super(SegmentationLoss, self).__init__()
        self.weight = weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pred: (B, C, N) tensor of predictions
            target: (B, N) tensor of target labels
            
        Returns:
            Loss value
        """
        # Reshape for cross-entropy
        pred = pred.transpose(1, 2).contiguous()  # (B, N, C)
        batch_size, num_points, num_classes = pred.shape
        
        # Flatten
        pred = pred.view(-1, num_classes)
        target = target.view(-1)
        
        # Mask out invalid targets (commonly -1)
        valid_mask = target >= 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        return F.cross_entropy(pred, target, weight=self.weight)

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pred: (B, C, N) tensor of predictions
            target: (B, N) tensor of target labels
            
        Returns:
            Loss value
        """
        # Reshape for focal loss
        pred = pred.transpose(1, 2).contiguous()  # (B, N, C)
        batch_size, num_points, num_classes = pred.shape
        
        # Flatten
        pred = pred.view(-1, num_classes)
        target = target.view(-1)
        
        # Mask out invalid targets
        valid_mask = target >= 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        # Apply softmax to get probabilities
        pred_softmax = F.softmax(pred, dim=1)
        
        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes=num_classes).float()
        
        # Calculate focal loss
        pt = (target_one_hot * pred_softmax).sum(1)
        focal_weight = (1 - pt).pow(self.gamma)
        
        loss = F.cross_entropy(pred, target, weight=self.alpha, reduction='none')
        loss = (focal_weight * loss).mean()
        
        return loss