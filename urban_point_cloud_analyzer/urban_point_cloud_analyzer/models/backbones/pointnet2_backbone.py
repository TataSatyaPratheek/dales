# urban_point_cloud_analyzer/models/backbones/pointnet2_backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

try:
    import pointnet2_ops.pointnet2_modules as pn2
except ImportError:
    raise ImportError("PointNet++ ops not found. Please install pointnet2_ops_lib.")

class PointNet2Backbone(nn.Module):
    """
    PointNet++ feature extraction backbone
    """
    def __init__(self, 
                 use_normals: bool = True,
                 num_points: int = 16384,
                 num_layers: int = 4,
                 feature_dims: List[int] = None,
                 sa_n_points: List[int] = None,
                 sa_n_samples: List[int] = None,
                 sa_radii: List[float] = None):
        """
        Initialize PointNet++ backbone.
        
        Args:
            use_normals: Whether to use normals as additional input features
            num_points: Number of input points
            num_layers: Number of set abstraction layers
            feature_dims: List of feature dimensions for each layer
            sa_n_points: List of number of points for each set abstraction layer
            sa_n_samples: List of number of samples for each set abstraction layer
            sa_radii: List of radii for each set abstraction layer
        """
        super(PointNet2Backbone, self).__init__()
        
        self.use_normals = use_normals
        self.num_points = num_points
        self.num_layers = num_layers
        
        # Define default configurations if not provided
        if feature_dims is None:
            self.feature_dims = [64, 128, 256, 512]
        else:
            self.feature_dims = feature_dims
            
        if sa_n_points is None:
            self.sa_n_points = [2048, 512, 128, 32]
        else:
            self.sa_n_points = sa_n_points
            
        if sa_n_samples is None:
            self.sa_n_samples = [64, 32, 16, 8]
        else:
            self.sa_n_samples = sa_n_samples
            
        if sa_radii is None:
            self.sa_radii = [0.1, 0.2, 0.4, 0.8]
        else:
            self.sa_radii = sa_radii
        
        # Ensure all configuration lists have the right length
        assert len(self.feature_dims) >= num_layers
        assert len(self.sa_n_points) >= num_layers
        assert len(self.sa_n_samples) >= num_layers
        assert len(self.sa_radii) >= num_layers
        
        # Input feature dimension depends on whether we use normals
        in_channel = 3 if not use_normals else 6
        
        # Create set abstraction layers
        self.sa_modules = nn.ModuleList()
        
        for i in range(num_layers):
            # Input channels for first layer is 3 (or 6 with normals)
            # For subsequent layers, use the previous layer's output channels
            if i == 0:
                input_channels = in_channel
            else:
                input_channels = self.feature_dims[i-1]
            
            # Create set abstraction module
            self.sa_modules.append(
                pn2.PointnetSAModule(
                    npoint=self.sa_n_points[i],
                    radius=self.sa_radii[i],
                    nsample=self.sa_n_samples[i],
                    mlp=[input_channels, self.feature_dims[i] // 2, self.feature_dims[i]]
                )
            )
        
        # Feature propagation layers for upsampling
        self.fp_modules = nn.ModuleList()
        
        # Build feature propagation layers in reverse order
        for i in range(num_layers - 1, 0, -1):
            self.fp_modules.append(
                pn2.PointnetFPModule(
                    mlp=[self.feature_dims[i] + self.feature_dims[i-1], 
                         self.feature_dims[i], 
                         self.feature_dims[i-1]]
                )
            )
        
        # Final feature propagation layer
        self.fp_modules.append(
            pn2.PointnetFPModule(
                mlp=[self.feature_dims[0] + in_channel, 
                     self.feature_dims[0] // 2, 
                     self.feature_dims[0] // 4, 
                     self.feature_dims[0]]
            )
        )
    
    def forward(self, 
                xyz: torch.Tensor, 
                features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            xyz: (B, N, 3) tensor of point coordinates
            features: (B, N, C) tensor of point features
            
        Returns:
            Tuple of (points, features) where:
                points: (B, N, 3) tensor of input points
                features: List of feature tensors, final one has shape (B, N, F)
        """
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
        
        # Apply feature propagation modules
        for i, fp_module in enumerate(self.fp_modules):
            # Get target points and features from stored list
            target_points = sa_points[-(i+2)]
            target_features = sa_features[-(i+2)]
            
            # Apply feature propagation
            features = fp_module(target_points, points, target_features, features)
            points = target_points
        
        # Return original points and final features
        return xyz, features.transpose(1, 2).contiguous()

class PointNet2Encoder(nn.Module):
    """
    PointNet++ encoder for segmentation tasks with CUDA memory optimization
    """
    def __init__(self, 
                 use_normals: bool = True,
                 in_channels: int = 3,
                 num_classes: int = 8, 
                 use_checkpoint: bool = True,
                 use_height: bool = False):
        """
        Initialize PointNet++ encoder.
        
        Args:
            use_normals: Whether to use normals
            in_channels: Number of input channels (3 for XYZ, more if using additional features)
            num_classes: Number of segmentation classes
            use_checkpoint: Whether to use gradient checkpointing (for memory efficiency)
            use_height: Whether to use height as an additional feature
        """
        super(PointNet2Encoder, self).__init__()
        
        # Determine input channels
        if use_height:
            in_channels += 1
        if use_normals:
            in_channels += 3
        
        self.use_normals = use_normals
        self.use_checkpoint = use_checkpoint
        self.use_height = use_height
        
        # First set abstraction layer
        self.sa1 = pn2.PointnetSAModuleMSG(
            npoint=1024,
            radii=[0.05, 0.1],
            nsamples=[16, 32],
            mlps=[[in_channels, 16, 16, 32], [in_channels, 32, 32, 64]]
        )
        
        # Second set abstraction layer
        self.sa2 = pn2.PointnetSAModuleMSG(
            npoint=256,
            radii=[0.1, 0.2],
            nsamples=[16, 32],
            mlps=[[32 + 64, 64, 64, 128], [32 + 64, 64, 96, 128]]
        )
        
        # Third set abstraction layer
        self.sa3 = pn2.PointnetSAModuleMSG(
            npoint=64,
            radii=[0.2, 0.4],
            nsamples=[16, 32],
            mlps=[[128 + 128, 128, 196, 256], [128 + 128, 128, 196, 256]]
        )
        
        # Fourth set abstraction layer
        self.sa4 = pn2.PointnetSAModuleMSG(
            npoint=16,
            radii=[0.4, 0.8],
            nsamples=[16, 32],
            mlps=[[256 + 256, 256, 256, 512], [256 + 256, 256, 384, 512]]
        )
        
        # Feature propagation layers
        self.fp4 = pn2.PointnetFPModule(mlp=[512 + 512 + 256 + 256, 512, 512])
        self.fp3 = pn2.PointnetFPModule(mlp=[512 + 128 + 128, 512, 512])
        self.fp2 = pn2.PointnetFPModule(mlp=[512 + 32 + 64, 256, 256])
        self.fp1 = pn2.PointnetFPModule(mlp=[256 + in_channels, 128, 128, 128])
        
        # Final layer for segmentation
        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1)
        )
    
    def _sa_forward(self, sa_module, xyz, features):
        """Helper function for set abstraction with optional checkpointing"""
        if self.use_checkpoint and self.training:
            # Use checkpoint to save memory during training
            return torch.utils.checkpoint.checkpoint(sa_module, xyz, features)
        else:
            return sa_module(xyz, features)
    
    def _fp_forward(self, fp_module, xyz1, xyz2, features1, features2):
        """Helper function for feature propagation with optional checkpointing"""
        if self.use_checkpoint and self.training:
            # Use checkpoint to save memory during training
            return torch.utils.checkpoint.checkpoint(fp_module, xyz1, xyz2, features1, features2)
        else:
            return fp_module(xyz1, xyz2, features1, features2)
    
    def forward(self, xyz, features=None):
        """
        Forward pass.
        
        Args:
            xyz: (B, N, 3) tensor of point coordinates
            features: (B, N, C) tensor of point features (optional)
        
        Returns:
            (B, num_classes, N) tensor of per-point logits
        """
        batch_size, num_points = xyz.shape[0], xyz.shape[1]
        
        # Add height as a feature if requested
        if self.use_height:
            height = xyz[:, :, 2:3]
            if features is None:
                features = height
            else:
                features = torch.cat([features, height], dim=2)
        
        # If no features are provided, just use xyz
        if features is None:
            features = xyz
        
        # Make sure features are in the right format for PointNet++ modules
        features = features.transpose(1, 2).contiguous()
        
        # Set abstraction layers
        xyz1, features1 = self._sa_forward(self.sa1, xyz, features)
        xyz2, features2 = self._sa_forward(self.sa2, xyz1, features1)
        xyz3, features3 = self._sa_forward(self.sa3, xyz2, features2)
        xyz4, features4 = self._sa_forward(self.sa4, xyz3, features3)
        
        # Feature propagation layers
        features3 = self._fp_forward(self.fp4, xyz3, xyz4, features3, features4)
        features2 = self._fp_forward(self.fp3, xyz2, xyz3, features2, features3)
        features1 = self._fp_forward(self.fp2, xyz1, xyz2, features1, features2)
        features = self._fp_forward(self.fp1, xyz, xyz1, features, features1)
        
        # Final prediction layer
        logits = self.fc_layer(features)
        
        return logits