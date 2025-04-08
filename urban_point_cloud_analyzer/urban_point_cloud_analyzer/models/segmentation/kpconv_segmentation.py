# urban_point_cloud_analyzer/models/segmentation/kpconv_segmentation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

class KPConvLayer(nn.Module):
    """
    Kernel Point Convolution layer.
    Implementation based on Thomas et al. KPConv paper.
    """
    def __init__(self, 
                in_channels: int, 
                out_channels: int,
                kernel_size: int = 15,
                radius: float = 0.05,
                sigma: float = 0.03,
                use_checkpoint: bool = True):
        super(KPConvLayer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.radius = radius
        self.sigma = sigma
        self.use_checkpoint = use_checkpoint
        
        # Kernel points are a fixed set of points defining the shape of the kernel
        self.kernel_points = self._initialize_kernel_points()
        
        # Convolution weights
        self.weights = nn.Parameter(torch.zeros((kernel_size, in_channels, out_channels)))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Weight initialization
        nn.init.kaiming_uniform_(self.weights)
        nn.init.zeros_(self.bias)
        
    def _initialize_kernel_points(self) -> torch.Tensor:
        """
        Initialize kernel points in a sphere.
        Returns:
            Tensor of shape (kernel_size, 3)
        """
        # Simple initialization with fixed points on a sphere
        # In production, this would use a more sophisticated algorithm
        kernel_points = []
        
        # Center point
        kernel_points.append([0, 0, 0])
        
        # Points on a sphere
        if self.kernel_size > 1:
            golden_ratio = (1 + 5**0.5) / 2
            i = torch.arange(1, self.kernel_size)
            theta = 2 * torch.pi * i / golden_ratio
            phi = torch.acos(1 - 2 * i / (self.kernel_size - 1))
            
            x = torch.sin(phi) * torch.cos(theta)
            y = torch.sin(phi) * torch.sin(theta)
            z = torch.cos(phi)
            
            sphere_points = torch.stack([x, y, z], dim=1)
            # Scale points to desired radius
            sphere_points = sphere_points * self.radius
            
            kernel_points.extend(sphere_points.tolist())
        
        return nn.Parameter(torch.tensor(kernel_points, dtype=torch.float), requires_grad=False)
    
    def forward(self, xyz: torch.Tensor, features: torch.Tensor, neighbors_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            xyz: (B, N, 3) tensor of point coordinates
            features: (B, N, C_in) tensor of point features
            neighbors_idx: (B, N, K) tensor of neighbor indices
            
        Returns:
            (B, N, C_out) tensor of output features
        """
        batch_size, num_points, _ = xyz.shape
        _, _, k = neighbors_idx.shape
        
        # Use gradient checkpointing if requested to save memory
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(self._forward_implementation, xyz, features, neighbors_idx)
        else:
            return self._forward_implementation(xyz, features, neighbors_idx)
    
    def _forward_implementation(self, xyz: torch.Tensor, features: torch.Tensor, neighbors_idx: torch.Tensor) -> torch.Tensor:
        """
        Actual forward implementation.
        
        Args:
            xyz: (B, N, 3) tensor of point coordinates
            features: (B, N, C_in) tensor of point features
            neighbors_idx: (B, N, K) tensor of neighbor indices
            
        Returns:
            (B, N, C_out) tensor of output features
        """
        batch_size, num_points, _ = xyz.shape
        _, _, k = neighbors_idx.shape
        
        # Initialize output features
        output_features = xyz.new_zeros((batch_size, num_points, self.out_channels))
        
        # Convert neighbors_idx to proper shape for gathering
        neighbors_idx = neighbors_idx.view(batch_size, -1)
        
        # For each batch
        for batch_idx in range(batch_size):
            # Gather neighbor features and coordinates
            neighbor_indices = neighbors_idx[batch_idx]
            
            # Get coordinates of neighbors
            neighbor_coords = xyz[batch_idx, neighbor_indices].view(num_points, k, 3)
            center_coords = xyz[batch_idx].unsqueeze(1).repeat(1, k, 1)
            
            # Calculate relative positions
            relative_positions = neighbor_coords - center_coords
            
            # Get features of neighbors
            neighbor_features = features[batch_idx, neighbor_indices].view(num_points, k, self.in_channels)
            
            # Compute kernel weights based on relative positions
            # Calculate distances to kernel points
            expanded_kernel_points = self.kernel_points.unsqueeze(0).unsqueeze(0).expand(num_points, k, -1, 3)
            expanded_positions = relative_positions.unsqueeze(2).expand(-1, -1, self.kernel_size, -1)
            distances = torch.sum((expanded_positions - expanded_kernel_points) ** 2, dim=-1)
            
            # Apply Gaussian kernel
            kernel_weights = torch.exp(-distances / (2 * self.sigma ** 2))
            
            # Compute output features through kernel convolution
            for i in range(self.kernel_size):
                # Weighted features by kernel point
                weighted_features = (neighbor_features * kernel_weights[:, :, i:i+1])
                
                # Convolve
                output_features[batch_idx] += torch.matmul(
                    weighted_features.sum(dim=1), 
                    self.weights[i]
                )
            
            # Add bias
            output_features[batch_idx] += self.bias
        
        return output_features


class KPConvBlock(nn.Module):
    """
    KPConv Block with batch normalization and ReLU.
    """
    def __init__(self, 
                in_channels: int, 
                out_channels: int,
                radius: float = 0.05,
                kernel_size: int = 15,
                use_checkpoint: bool = True):
        super(KPConvBlock, self).__init__()
        
        self.kpconv = KPConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            radius=radius,
            use_checkpoint=use_checkpoint
        )
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, xyz: torch.Tensor, features: torch.Tensor, neighbors_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            xyz: (B, N, 3) tensor of point coordinates
            features: (B, N, C_in) tensor of point features
            neighbors_idx: (B, N, K) tensor of neighbor indices
            
        Returns:
            (B, N, C_out) tensor of output features
        """
        # Apply KPConv
        features = self.kpconv(xyz, features, neighbors_idx)
        
        # Reshape for batch norm
        batch_size, num_points, channels = features.shape
        features = features.transpose(1, 2).contiguous()  # (B, C, N)
        
        # Apply batch norm and ReLU
        features = self.bn(features)
        features = self.relu(features)
        
        # Reshape back
        features = features.transpose(1, 2).contiguous()  # (B, N, C)
        
        return features


class KPConvBackbone(nn.Module):
    """
    KPConv backbone for point cloud feature extraction.
    """
    def __init__(self, 
                 in_channels: int = 3,
                 feature_dims: List[int] = [32, 64, 128, 256],
                 radii: List[float] = [0.05, 0.1, 0.2, 0.4],
                 num_neighbors: List[int] = [16, 16, 16, 16],
                 use_checkpoint: bool = True):
        super(KPConvBackbone, self).__init__()
        
        self.in_channels = in_channels
        self.feature_dims = feature_dims
        self.radii = radii
        self.num_neighbors = num_neighbors
        self.use_checkpoint = use_checkpoint
        
        # KPConv layers
        self.encoder_blocks = nn.ModuleList()
        
        # First block takes input features
        self.encoder_blocks.append(
            KPConvBlock(
                in_channels=in_channels,
                out_channels=feature_dims[0],
                radius=radii[0],
                use_checkpoint=use_checkpoint
            )
        )
        
        # Subsequent blocks
        for i in range(1, len(feature_dims)):
            self.encoder_blocks.append(
                KPConvBlock(
                    in_channels=feature_dims[i-1],
                    out_channels=feature_dims[i],
                    radius=radii[i],
                    use_checkpoint=use_checkpoint
                )
            )
    
    def _get_neighbors(self, xyz: torch.Tensor, radius: float, max_neighbors: int) -> torch.Tensor:
        """
        Find neighbors within radius.
        
        Args:
            xyz: (B, N, 3) tensor of point coordinates
            radius: Radius for neighbor search
            max_neighbors: Maximum number of neighbors
            
        Returns:
            (B, N, K) tensor of neighbor indices
        """
        batch_size, num_points, _ = xyz.shape
        device = xyz.device
        
        # Use custom CUDA kernel for efficiency if available
        try:
            from urban_point_cloud_analyzer.optimization.cuda import k_nearest_neighbors
            
            # KNN search with fixed K
            neighbors_idx = k_nearest_neighbors(xyz, max_neighbors)
            
            # Filter by radius (approximate approach)
            # In a full implementation, we would use a proper radius search
            # Here, we're using KNN as an approximation
            valid_mask = torch.ones((batch_size, num_points, max_neighbors), device=device, dtype=torch.bool)
            
            return neighbors_idx, valid_mask
            
        except ImportError:
            # Fallback to a simple implementation
            neighbors_idx = torch.zeros((batch_size, num_points, max_neighbors), 
                                      device=device, dtype=torch.long)
            valid_mask = torch.zeros((batch_size, num_points, max_neighbors), 
                                    device=device, dtype=torch.bool)
            
            # For each batch
            for b in range(batch_size):
                # Calculate pairwise distances
                dist = torch.cdist(xyz[b], xyz[b])
                
                # For each point, get the closest neighbors
                for i in range(num_points):
                    # Get points within radius
                    mask = dist[i] < radius
                    neighbors = torch.nonzero(mask).squeeze(-1)
                    
                    # Sort neighbors by distance
                    distances = dist[i, neighbors]
                    sorted_idx = torch.argsort(distances)
                    neighbors = neighbors[sorted_idx]
                    
                    # Take only max_neighbors
                    k = min(len(neighbors), max_neighbors)
                    if k > 0:
                        neighbors_idx[b, i, :k] = neighbors[:k]
                        valid_mask[b, i, :k] = True
            
            return neighbors_idx, valid_mask
    
    def forward(self, xyz: torch.Tensor, features: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            xyz: (B, N, 3) tensor of point coordinates
            features: Optional (B, N, C) tensor of input features
            
        Returns:
            Tuple of (point_list, feature_list) where each is a list of tensors
        """
        # If no input features, use xyz coordinates
        if features is None:
            features = xyz
        
        # Keep track of points and features at each level
        points_list = [xyz]
        features_list = [features]
        
        # Encoder forward pass
        for i, block in enumerate(self.encoder_blocks):
            # Find neighbors for this radius
            neighbors_idx, valid_mask = self._get_neighbors(
                xyz, self.radii[i], self.num_neighbors[i]
            )
            
            # Apply KPConv block
            new_features = block(xyz, features, neighbors_idx)
            
            # Apply mask for valid neighbors
            new_features = new_features * valid_mask.float().unsqueeze(-1)
            
            # Farthest point sampling for next level
            # In a full implementation, we would use FPS to subsample points
            # Here we're using a simple stride subsampling for demonstration
            if i < len(self.encoder_blocks) - 1:
                stride = xyz.shape[1] // 4
                xyz = xyz[:, ::stride, :]
                new_features = new_features[:, ::stride, :]
            
            # Update features
            features = new_features
            
            # Add to lists
            points_list.append(xyz)
            features_list.append(features)
        
        return points_list, features_list


class KPConvSegmentationModel(nn.Module):
    """
    KPConv segmentation model
    """
    def __init__(self, 
                 num_classes: int = 8, 
                 in_channels: int = 3,
                 dropout: float = 0.5,
                 use_checkpoint: bool = True):
        """
        Initialize KPConv segmentation model.
        
        Args:
            num_classes: Number of segmentation classes
            in_channels: Number of input channels
            dropout: Dropout probability
            use_checkpoint: Whether to use gradient checkpointing (for memory efficiency)
        """
        super(KPConvSegmentationModel, self).__init__()
        
        # Store config for reference
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint
        
        # Feature extraction backbone
        self.backbone = KPConvBackbone(
            in_channels=in_channels,
            feature_dims=[32, 64, 128, 256],
            radii=[0.05, 0.1, 0.2, 0.4],
            num_neighbors=[16, 16, 16, 16],
            use_checkpoint=use_checkpoint
        )
        
        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(128, num_classes, 1)
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
        points_list, features_list = self.backbone(xyz, features)
        
        # Get final features
        final_features = features_list[-1]
        
        # Apply segmentation head
        # Convert to (B, C, N) format for Conv1d
        final_features = final_features.transpose(1, 2).contiguous()
        logits = self.segmentation_head(final_features)
        
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
            features: Optional (B, N, C) tensor of point features
            
        Returns:
            (B, N) tensor of per-point predictions
        """
        logits = self.forward(points, features)
        predictions = torch.argmax(logits, dim=1)
        return predictions