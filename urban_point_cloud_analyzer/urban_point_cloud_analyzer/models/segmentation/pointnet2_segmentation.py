# urban_point_cloud_analyzer/models/segmentation/pointnet2_segmentation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from urban_point_cloud_analyzer.models.backbones.pointnet2_backbone import PointNet2Encoder

class PointNet2SegmentationModel(nn.Module):
    """
    PointNet++ segmentation model
    """
    def __init__(self, 
                 num_classes: int = 8, 
                 use_normals: bool = True,
                 in_channels: int = 3,
                 dropout: float = 0.5,
                 use_height: bool = True,
                 use_checkpoint: bool = True):
        """
        Initialize PointNet++ segmentation model.
        
        Args:
            num_classes: Number of segmentation classes
            use_normals: Whether to use normals as additional input features
            in_channels: Number of input channels
            dropout: Dropout probability
            use_height: Whether to use height as an additional feature
            use_checkpoint: Whether to use gradient checkpointing (for memory efficiency)
        """
        super(PointNet2SegmentationModel, self).__init__()
        
        # Feature extraction backbone
        self.encoder = PointNet2Encoder(
            use_normals=use_normals,
            in_channels=in_channels,
            num_classes=num_classes,
            use_checkpoint=use_checkpoint,
            use_height=use_height
        )
        
        # Store config for reference
        self.num_classes = num_classes
        self.use_normals = use_normals
        self.in_channels = in_channels
        self.dropout = dropout
        
    def forward(self, points: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            points: (B, N, 3) tensor of point coordinates
            features: (B, N, C) tensor of point features (optional)
            
        Returns:
            (B, num_classes, N) tensor of per-point logits
        """
        return self.encoder(points, features)
    
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
        logits = logits.view(-1, self.num_classes)
        labels = labels.view(-1)
        
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
            features: (B, N, C) tensor of point features (optional)
            
        Returns:
            (B, N) tensor of per-point predictions
        """
        logits = self.forward(points, features)
        predictions = torch.argmax(logits, dim=1)
        return predictions