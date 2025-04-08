# urban_point_cloud_analyzer/models/segmentation/spvcnn_segmentation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from urban_point_cloud_analyzer.optimization.sparse_ops import create_sparse_tensor, SparseTensor

class SparseConvBlock(nn.Module):
    """
    Basic Sparse Convolution Block.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super(SparseConvBlock, self).__init__()
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, C, D, H, W) sparse tensor
            
        Returns:
            (B, C', D', H', W') sparse tensor
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SparseResBlock(nn.Module):
    """
    Sparse Residual Block.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(SparseResBlock, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels)
            )
        
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, C, D, H, W) sparse tensor
            
        Returns:
            (B, C', D, H, W) sparse tensor
        """
        residual = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x += self.shortcut(residual)
        x = self.relu2(x)
        
        return x


class PointToVoxel(nn.Module):
    """
    Convert point cloud to voxel grid.
    """
    def __init__(self, voxel_size: float = 0.05, use_checkpoint: bool = True):
        super(PointToVoxel, self).__init__()
        
        self.voxel_size = voxel_size
        self.use_checkpoint = use_checkpoint
    
    def _get_voxel_coords(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Get voxel coordinates from point cloud.
        
        Args:
            xyz: (B, N, 3) tensor of point coordinates
            
        Returns:
            (B, N, 3) tensor of voxel coordinates
        """
        # Quantize point coordinates to voxel grid
        voxel_coords = torch.floor(xyz / self.voxel_size).int()
        
        # Shift coordinates to be non-negative
        min_coords = torch.min(voxel_coords, dim=1, keepdim=True)[0]
        voxel_coords = voxel_coords - min_coords
        
        return voxel_coords
    
    def forward(self, xyz: torch.Tensor, features: Optional[torch.Tensor] = None) -> Tuple[SparseTensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            xyz: (B, N, 3) tensor of point coordinates
            features: Optional (B, N, C) tensor of point features
            
        Returns:
            Tuple of (voxel_features, voxel_coords)
        """
        # Use gradient checkpointing if requested to save memory
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            voxel_coords = torch.utils.checkpoint.checkpoint(self._get_voxel_coords, xyz)
        else:
            voxel_coords = self._get_voxel_coords(xyz)
        
        # Create sparse voxel tensor
        sparse_tensor = create_sparse_tensor(voxel_coords, features)
        
        return sparse_tensor, voxel_coords


class VoxelToPoint(nn.Module):
    """
    Convert voxel grid features back to point cloud.
    """
    def __init__(self, use_checkpoint: bool = True):
        super(VoxelToPoint, self).__init__()
        
        self.use_checkpoint = use_checkpoint
    
    def _map_voxel_to_point(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, num_points: int) -> torch.Tensor:
        """
        Map voxel features back to original point cloud.
        
        Args:
            voxel_features: (B, C, D, H, W) tensor of voxel features
            voxel_coords: (B, N, 3) tensor of voxel coordinates
            num_points: Number of points in original point cloud
            
        Returns:
            (B, N, C) tensor of point features
        """
        batch_size, channels = voxel_features.shape[0], voxel_features.shape[1]
        
        # Initialize point features
        point_features = torch.zeros((batch_size, num_points, channels), device=voxel_features.device)
        
        # For each batch
        for b in range(batch_size):
            # Get features for each point based on its voxel coordinate
            for i in range(num_points):
                x, y, z = voxel_coords[b, i]
                
                # Ensure coordinates are in bounds
                if x < voxel_features.shape[2] and y < voxel_features.shape[3] and z < voxel_features.shape[4]:
                    point_features[b, i] = voxel_features[b, :, x, y, z]
        
        return point_features
    
    def forward(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, num_points: int) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            voxel_features: (B, C, D, H, W) tensor of voxel features
            voxel_coords: (B, N, 3) tensor of voxel coordinates
            num_points: Number of points in original point cloud
            
        Returns:
            (B, N, C) tensor of point features
        """
        # Use gradient checkpointing if requested to save memory
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(self._map_voxel_to_point, voxel_features, voxel_coords, num_points)
        else:
            return self._map_voxel_to_point(voxel_features, voxel_coords, num_points)


class SPVCNNBackbone(nn.Module):
    """
    SPVCNN backbone for point cloud feature extraction.
    Combines sparse voxel convolutions with PointNet-style operations.
    """
    def __init__(self, 
                 in_channels: int = 3,
                 feature_dims: List[int] = [32, 64, 128, 256],
                 voxel_size: float = 0.05,
                 use_checkpoint: bool = True):
        super(SPVCNNBackbone, self).__init__()
        
        self.in_channels = in_channels
        self.feature_dims = feature_dims
        self.voxel_size = voxel_size
        self.use_checkpoint = use_checkpoint
        
        # Point to voxel conversion
        self.point_to_voxel = PointToVoxel(voxel_size=voxel_size, use_checkpoint=use_checkpoint)
        
        # Initial MLP to process point features
        self.point_mlp = nn.Sequential(
            nn.Linear(in_channels, feature_dims[0]),
            nn.BatchNorm1d(feature_dims[0]),
            nn.ReLU(inplace=True)
        )
        
        # Sparse voxel convolution blocks
        self.voxel_blocks = nn.ModuleList()
        
        # Create voxel convolution blocks
        for i in range(len(feature_dims) - 1):
            self.voxel_blocks.append(
                SparseResBlock(
                    in_channels=feature_dims[i],
                    out_channels=feature_dims[i+1]
                )
            )
        
        # Voxel to point conversion
        self.voxel_to_point = VoxelToPoint(use_checkpoint=use_checkpoint)
        
        # Additional point MLPs for feature refinement
        self.point_refine = nn.ModuleList()
        
        for i in range(len(feature_dims) - 1):
            self.point_refine.append(
                nn.Sequential(
                    nn.Linear(feature_dims[i] + feature_dims[i+1], feature_dims[i+1]),
                    nn.BatchNorm1d(feature_dims[i+1]),
                    nn.ReLU(inplace=True)
                )
            )
    
    def forward(self, xyz: torch.Tensor, features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            xyz: (B, N, 3) tensor of point coordinates
            features: Optional (B, N, C) tensor of point features
            
        Returns:
            Tuple of (points, features) where features is the final feature tensor
        """
        batch_size, num_points, _ = xyz.shape
        
        # If no input features, use point coordinates
        if features is None:
            features = xyz
        
        # Store original points for skip connections
        original_xyz = xyz
        
        # Initial point MLP
        point_features = self.point_mlp(features)
        
        # Convert points to voxels
        voxel_tensor, voxel_coords = self.point_to_voxel(xyz, point_features)
        
        # Process with voxel convolutions and point refinement
        features_list = [point_features]
        voxel_features = voxel_tensor.features
        
        for i, voxel_block in enumerate(self.voxel_blocks):
            # Apply voxel convolution
            voxel_features = voxel_block(voxel_features)
            
            # Convert back to point features
            point_feat = self.voxel_to_point(voxel_features, voxel_coords, num_points)
            
            # Refine with skip connection
            # Ensure compatible shapes for concatenation
            if point_feat.shape[1] == features_list[-1].shape[1]:
                refined_features = torch.cat([point_feat, features_list[-1]], dim=2)
                refined_features = self.point_refine[i](refined_features)
                features_list.append(refined_features)
            else:
                features_list.append(point_feat)
        
        # Return points and final feature
        return original_xyz, features_list[-1]


class SPVCNNSegmentationModel(nn.Module):
    """
    SPVCNN segmentation model
    """
    def __init__(self, 
                 num_classes: int = 8, 
                 in_channels: int = 3,
                 feature_dims: List[int] = [32, 64, 128, 256],
                 voxel_size: float = 0.05,
                 dropout: float = 0.5,
                 use_checkpoint: bool = True):
        """
        Initialize SPVCNN segmentation model.
        
        Args:
            num_classes: Number of segmentation classes
            in_channels: Number of input channels
            feature_dims: Feature dimensions for each layer
            voxel_size: Voxel size for voxelization
            dropout: Dropout probability
            use_checkpoint: Whether to use gradient checkpointing (for memory efficiency)
        """
        super(SPVCNNSegmentationModel, self).__init__()
        
        # Store config for reference
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feature_dims = feature_dims
        self.voxel_size = voxel_size
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint
        
        # Feature extraction backbone
        self.backbone = SPVCNNBackbone(
            in_channels=in_channels,
            feature_dims=feature_dims,
            voxel_size=voxel_size,
            use_checkpoint=use_checkpoint
        )
        
        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Linear(feature_dims[-1], 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, points: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            points: (B, N, 3) tensor of point coordinates
            features: Optional (B, N, C) tensor of input features
                
        Returns:
            (B, num_classes, N) tensor of per-point logits
        """
        # Extract points and features
        xyz = points[:, :, :3]
        
        # Forward through backbone
        _, point_features = self.backbone(xyz, features)
        
        # Apply segmentation head
        batch_size, num_points, feature_dim = point_features.shape
        
        # Reshape for batch norm in segmentation head
        point_features_flat = point_features.reshape(-1, feature_dim)
        logits_flat = self.segmentation_head(point_features_flat)
        
        # Reshape back
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