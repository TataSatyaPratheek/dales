# urban_point_cloud_analyzer/urban_point_cloud_analyzer/models/backbones/pointnet2_backbone_cpu.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

class PointNetSetAbstraction(nn.Module):
    """Simplified PointNet++ set abstraction module for CPU/MPS"""
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        
        # MLP layers
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
            
    def forward(self, xyz, points):
        """
        CPU implementation of set abstraction
        xyz: (B, N, 3)
        points: (B, C, N)
        """
        B, N, _ = xyz.shape
        
        # Simplified version - just downsample by choosing points at regular intervals
        if self.npoint is not None:
            if N >= self.npoint:
                idx = torch.linspace(0, N-1, self.npoint, dtype=torch.long, device=xyz.device)
                new_xyz = xyz[:, idx, :]
                if points is not None:
                    new_points = points[:, :, idx]
                else:
                    new_points = xyz.transpose(1, 2)[:, :, idx]
            else:
                # If we have fewer points than needed, duplicate some
                idx = torch.randint(0, N, (self.npoint,), device=xyz.device)
                new_xyz = xyz[:, idx, :]
                if points is not None:
                    new_points = points[:, :, idx]
                else:
                    new_points = xyz.transpose(1, 2)[:, :, idx]
        else:
            # If no downsampling, use global features
            new_xyz = xyz.mean(dim=1, keepdim=True)
            new_points = torch.max(points, dim=2, keepdim=True)[0]
        
        # Apply MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
            
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    """Simplified PointNet++ feature propagation module for CPU/MPS"""
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
            
    def forward(self, xyz1, xyz2, points1, points2):
        """
        CPU implementation of feature propagation
        xyz1: (B, N1, 3)
        xyz2: (B, N2, 3)
        points1: (B, C1, N1)
        points2: (B, C2, N2)
        """
        # Simple interpolation - just duplicate the features from the nearest point
        B, N1, _ = xyz1.shape
        _, N2, _ = xyz2.shape
        
        if N1 < N2:
            # Interpolate from fewer points to more points
            # This is a simplified version - in real PointNet++ this is a more complex interpolation
            # For each point in xyz1, find its nearest neighbor in xyz2
            dist = torch.cdist(xyz1, xyz2)
            idx = torch.argmin(dist, dim=2)  # (B, N1)
            
            # Get the features from these neighbors
            interpolated_points = torch.zeros((B, points2.shape[1], N1), device=points1.device)
            for b in range(B):
                interpolated_points[b, :, :] = points2[b, :, idx[b]]
        else:
            # Just duplicate features for all points
            interpolated_points = points2.expand(-1, -1, N1)
        
        if points1 is not None:
            new_points = torch.cat([interpolated_points, points1], dim=1)
        else:
            new_points = interpolated_points
            
        # Apply MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
            
        return new_points


class PointNet2BackboneCPU(nn.Module):
    """
    PointNet++ feature extraction backbone - CPU/MPS fallback version
    """
    def __init__(self, 
                 use_normals: bool = True,
                 in_channels: int = 3,  # Added in_channels parameter
                 num_points: int = 16384,
                 num_layers: int = 4,
                 feature_dims: List[int] = None,
                 use_checkpoint: bool = False,  # Added use_checkpoint parameter
                 use_height: bool = False):     # Added use_height parameter
        """
        Initialize PointNet++ backbone.
        
        Args:
            use_normals: Whether to use normals as additional input features
            in_channels: Number of input channels
            num_points: Number of input points
            num_layers: Number of set abstraction layers
            feature_dims: List of feature dimensions for each layer
            use_checkpoint: Whether to use gradient checkpointing (for memory efficiency)
            use_height: Whether to use height as an additional feature
        """
        super(PointNet2BackboneCPU, self).__init__()
        
        self.use_normals = use_normals
        self.in_channels = in_channels
        self.num_points = num_points
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint
        self.use_height = use_height
        
        # Define default configurations if not provided
        if feature_dims is None:
            self.feature_dims = [64, 128, 256, 512]
        else:
            self.feature_dims = feature_dims
            
        # Input feature dimension depends on whether we use height and normals
        in_channel = in_channels
        if use_height:
            in_channel += 1
        if use_normals:
            in_channel += 3
        
        # Create set abstraction layers
        self.sa_modules = nn.ModuleList()
        
        # Simplified architecture with fewer parameters for M1 Macs
        self.sa_modules.append(
            PointNetSetAbstraction(
                npoint=512, radius=0.2, nsample=32,
                in_channel=in_channel, mlp=[64, 64, 128]
            )
        )
        
        self.sa_modules.append(
            PointNetSetAbstraction(
                npoint=128, radius=0.4, nsample=32,
                in_channel=128, mlp=[128, 128, 256]
            )
        )
        
        self.sa_modules.append(
            PointNetSetAbstraction(
                npoint=None, radius=None, nsample=None,
                in_channel=256, mlp=[256, 512, 1024]
            )
        )
        
        # Feature propagation layers for upsampling
        self.fp_modules = nn.ModuleList()
        
        self.fp_modules.append(
            PointNetFeaturePropagation(
                in_channel=1024+256, mlp=[512, 256]
            )
        )
        
        self.fp_modules.append(
            PointNetFeaturePropagation(
                in_channel=256+128, mlp=[256, 128]
            )
        )
        
        self.fp_modules.append(
            PointNetFeaturePropagation(
                in_channel=128+in_channel, mlp=[128, 128]
            )
        )
    
    def forward(self, 
                xyz: torch.Tensor, 
                features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            xyz: (B, N, 3) tensor of point coordinates
            features: (B, N, C) tensor of point features
            
        Returns:
            Tuple of (points, features) where features has shape (B, C, N)
        """
        # Add height as a feature if requested
        if self.use_height:
            height = xyz[:, :, 2:3]
            if features is None:
                features = height
            else:
                features = torch.cat([features, height], dim=2)
                
        # If features are not provided, use xyz coordinates
        if features is None:
            features = xyz.transpose(1, 2).contiguous()
        else:
            features = features.transpose(1, 2).contiguous()
        
        # Store point features at each level for skip connections
        sa_points = []
        sa_features = []
        
        # Initial point coordinates and features
        points = xyz
        
        # Apply set abstraction modules
        for i, sa_module in enumerate(self.sa_modules):
            # Add to list before processing for skip connections
            sa_points.append(points)
            sa_features.append(features)
            
            # Apply set abstraction
            points, features = sa_module(points, features)
        
        # Apply feature propagation modules in reverse order
        for i in range(len(self.fp_modules)):
            target_idx = -(i+2) if i < len(self.fp_modules)-1 else 0
            # Get target points and features from stored list
            target_points = sa_points[target_idx]
            target_features = sa_features[target_idx]
            
            # Apply feature propagation
            features = self.fp_modules[i](target_points, points, target_features, features)
            points = target_points
        
        # Return original points and final features
        return xyz, features