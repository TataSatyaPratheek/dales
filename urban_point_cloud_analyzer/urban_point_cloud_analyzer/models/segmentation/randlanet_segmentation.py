# urban_point_cloud_analyzer/models/segmentation/randlanet_segmentation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

class SharedMLP(nn.Module):
    """
    Shared MLP layer applied to each point independently.
    """
    def __init__(self, in_channels: int, out_channels: List[int], activation=nn.ReLU(inplace=True)):
        super(SharedMLP, self).__init__()
        
        layers = []
        last_channel = in_channels
        
        for out_channel in out_channels:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm1d(out_channel))
            if activation:
                layers.append(activation)
            last_channel = out_channel
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, C, N) tensor of point features
            
        Returns:
            (B, C', N) tensor of transformed features
        """
        return self.layers(x)


class LocalFeatureAggregation(nn.Module):
    """
    Local Feature Aggregation module from RandLA-Net.
    """
    def __init__(self, in_channels: int, out_channels: int, num_neighbors: int = 16, use_checkpoint: bool = True):
        super(LocalFeatureAggregation, self).__init__()
        
        self.num_neighbors = num_neighbors
        self.use_checkpoint = use_checkpoint
        
        # MLP for local spatial encoding
        self.mlp_encoding = SharedMLP(10, [in_channels//2, in_channels])
        
        # Attention module
        self.attention = nn.Sequential(
            SharedMLP(in_channels*2, [in_channels, in_channels]),
            nn.Conv1d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Last MLP
        self.mlp_out = SharedMLP(in_channels, [in_channels, out_channels])
        
        # Shortcut connection
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels)
        )
        
        # Activation function
        self.activation = nn.ReLU(inplace=True)
    
    def _get_local_features(self, xyz: torch.Tensor, features: torch.Tensor, neighbors_idx: torch.Tensor) -> torch.Tensor:
        """
        Get local features for each point.
        
        Args:
            xyz: (B, N, 3) tensor of point coordinates
            features: (B, C, N) tensor of point features
            neighbors_idx: (B, N, K) tensor of neighbor indices
            
        Returns:
            (B, C, N) tensor of aggregated features
        """
        batch_size, num_points, _ = xyz.shape
        _, feature_dim, _ = features.shape
        k = neighbors_idx.shape[2]
        
        # Get feature of each point and its neighbors
        # features shape: (B, C, N)
        # neighbors_idx shape: (B, N, K)
        
        # Reshape neighbors_idx to gather features
        neighbors_idx_flat = neighbors_idx.reshape(batch_size, -1)  # (B, N*K)
        
        # Handle potential invalid indices
        valid_mask = (neighbors_idx_flat >= 0) & (neighbors_idx_flat < num_points)
        neighbors_idx_safe = torch.clamp(neighbors_idx_flat, 0, num_points - 1)
        
        # Gather neighbor features
        batch_indices = torch.arange(batch_size, device=xyz.device).unsqueeze(1).expand(-1, num_points * k)
        neighbor_features = features[batch_indices, :, neighbors_idx_safe]  # (B, C, N*K)
        neighbor_features = neighbor_features.reshape(batch_size, feature_dim, num_points, k)  # (B, C, N, K)
        
        # Apply valid mask
        valid_mask = valid_mask.reshape(batch_size, num_points, k).unsqueeze(1)  # (B, 1, N, K)
        neighbor_features = neighbor_features * valid_mask.float()
        
        # Get relative positions
        # xyz shape: (B, N, 3)
        xyz_expanded = xyz.unsqueeze(2).expand(-1, -1, k, -1)  # (B, N, K, 3)
        
        # Gather neighbor positions
        neighbor_positions = xyz[batch_indices.reshape(batch_size, num_points, k), neighbors_idx.reshape(batch_size, num_points, k)]  # (B, N, K, 3)
        
        # Compute relative positions
        relative_positions = neighbor_positions - xyz_expanded  # (B, N, K, 3)
        
        # Compute relative distances
        relative_distances = torch.norm(relative_positions, dim=3, keepdim=True)  # (B, N, K, 1)
        
        # Concatenate to get full spatial encoding
        spatial_features = torch.cat([
            relative_positions,  # (B, N, K, 3)
            relative_distances,  # (B, N, K, 1)
            xyz_expanded,        # (B, N, K, 3)
            neighbor_positions   # (B, N, K, 3)
        ], dim=3)  # (B, N, K, 10)
        
        # Transpose for MLP
        spatial_features = spatial_features.permute(0, 3, 1, 2)  # (B, 10, N, K)
        spatial_features = spatial_features.reshape(batch_size, 10, -1)  # (B, 10, N*K)
        
        # Apply MLP for local spatial encoding
        encoded_features = self.mlp_encoding(spatial_features)  # (B, C, N*K)
        encoded_features = encoded_features.reshape(batch_size, feature_dim, num_points, k)  # (B, C, N, K)
        
        # Apply max pooling for each point
        aggregated_features, _ = torch.max(encoded_features, dim=3)  # (B, C, N)
        
        return aggregated_features
    
    def forward(self, xyz: torch.Tensor, features: torch.Tensor, neighbors_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            xyz: (B, N, 3) tensor of point coordinates
            features: (B, C, N) tensor of point features
            neighbors_idx: (B, N, K) tensor of neighbor indices
            
        Returns:
            (B, C', N) tensor of aggregated features
        """
        # Use gradient checkpointing if requested to save memory
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            aggregated_features = torch.utils.checkpoint.checkpoint(
                self._get_local_features, xyz, features, neighbors_idx
            )
        else:
            aggregated_features = self._get_local_features(xyz, features, neighbors_idx)
        
        # Attention mechanism
        attention_features = torch.cat([features, aggregated_features], dim=1)  # (B, 2C, N)
        attention_weights = self.attention(attention_features)  # (B, C, N)
        
        # Apply attention
        weighted_features = features * attention_weights  # (B, C, N)
        
        # Apply MLP and shortcut
        output_features = self.mlp_out(weighted_features)  # (B, C', N)
        shortcut_features = self.shortcut(features)  # (B, C', N)
        
        # Residual connection
        output_features = output_features + shortcut_features
        
        # Final activation
        output_features = self.activation(output_features)
        
        return output_features


class RandomSampling(nn.Module):
    """
    Random sampling layer.
    """
    def __init__(self, ratio: float = 0.25):
        super(RandomSampling, self).__init__()
        self.ratio = ratio
    
    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            xyz: (B, N, 3) tensor of point coordinates
            features: (B, C, N) tensor of point features
            
        Returns:
            Tuple of (new_xyz, new_features, indices) where:
                new_xyz: (B, N', 3) tensor of sampled point coordinates
                new_features: (B, C, N') tensor of sampled point features
                indices: (B, N') tensor of indices of sampled points
        """
        batch_size, num_points, _ = xyz.shape
        
        # Determine number of points to sample
        sample_size = max(1, int(num_points * self.ratio))
        
        # Select random indices for each batch
        indices = torch.argsort(torch.rand(batch_size, num_points, device=xyz.device), dim=1)[:, :sample_size]
        
        # Create batch indices
        batch_indices = torch.arange(batch_size, device=xyz.device).unsqueeze(1).expand(-1, sample_size)
        
        # Gather sampled points and features
        new_xyz = xyz[batch_indices, indices]
        new_features = features.transpose(1, 2)[batch_indices, indices].transpose(1, 2)
        
        return new_xyz, new_features, indices


class RandLANetBackbone(nn.Module):
    """
    RandLA-Net backbone for point cloud feature extraction.
    """
    def __init__(self, 
                 in_channels: int = 3,
                 num_classes: int = 8,
                 num_points: int = 16384,
                 feature_dims: List[int] = [32, 64, 128, 256],
                 num_neighbors: int = 16,
                 decimation_ratio: float = 0.25,
                 use_checkpoint: bool = True):
        super(RandLANetBackbone, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_points = num_points
        self.feature_dims = feature_dims
        self.num_neighbors = num_neighbors
        self.decimation_ratio = decimation_ratio
        self.use_checkpoint = use_checkpoint
        
        # Initial MLP to embed features
        self.embedding = SharedMLP(in_channels, [feature_dims[0]])
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        self.sampling_layers = nn.ModuleList()
        
        # Create encoder layers
        for i in range(len(feature_dims) - 1):
            # Add random sampling layer
            self.sampling_layers.append(RandomSampling(decimation_ratio))
            
            # Add local feature aggregation module
            self.encoder_layers.append(
                LocalFeatureAggregation(
                    in_channels=feature_dims[i],
                    out_channels=feature_dims[i+1],
                    num_neighbors=num_neighbors,
                    use_checkpoint=use_checkpoint
                )
            )
    
    def _get_neighbors(self, xyz: torch.Tensor, k: int) -> torch.Tensor:
        """
        Find k nearest neighbors for each point.
        
        Args:
            xyz: (B, N, 3) tensor of point coordinates
            k: Number of neighbors
            
        Returns:
            (B, N, K) tensor of neighbor indices
        """
        # Use optimized CUDA kernel if available
        try:
            from urban_point_cloud_analyzer.optimization.cuda import k_nearest_neighbors
            return k_nearest_neighbors(xyz, k)
        except ImportError:
            # Fallback to torch implementation
            batch_size, num_points, _ = xyz.shape
            device = xyz.device
            
            # Calculate pairwise distances
            neighbors = torch.zeros((batch_size, num_points, k), dtype=torch.long, device=device)
            
            for b in range(batch_size):
                # Calculate pairwise distances for this batch
                dist = torch.cdist(xyz[b], xyz[b])
                
                # Get k nearest neighbors (exclude self)
                dist[torch.arange(num_points), torch.arange(num_points)] = float('inf')
                _, indices = torch.topk(dist, k=k, dim=1, largest=False)
                
                neighbors[b] = indices
            
            return neighbors
    
    def forward(self, xyz: torch.Tensor, features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            xyz: (B, N, 3) tensor of point coordinates
            features: Optional (B, N, C) tensor of input features
            
        Returns:
            Tuple of (points, features) where features is a list of feature tensors
        """
        # If no input features, use xyz coordinates
        if features is None:
            features = xyz
        
        # Convert features to (B, C, N) format
        features = features.transpose(1, 2).contiguous()
        
        # Initial feature embedding
        features = self.embedding(features)
        
        # Keep track of points and features at each level
        points_list = [xyz]
        features_list = [features]
        
        # Apply sampling and encoder layers
        for i in range(len(self.encoder_layers)):
            # Get neighbors for current resolution
            neighbors_idx = self._get_neighbors(xyz, self.num_neighbors)
            
            # Apply local feature aggregation
            features = self.encoder_layers[i](xyz, features, neighbors_idx)
            
            # Store features for skip connections
            features_list.append(features)
            
            # Apply random sampling
            if i < len(self.encoder_layers) - 1:
                xyz, features, _ = self.sampling_layers[i](xyz, features)
                
                # Store points for skip connections
                points_list.append(xyz)
        
        return xyz, features_list


class RandLANetSegmentationModel(nn.Module):
    """
    RandLA-Net segmentation model
    """
    def __init__(self, 
                 num_classes: int = 8, 
                 in_channels: int = 3,
                 dropout: float = 0.5,
                 use_checkpoint: bool = True):
        """
        Initialize RandLA-Net segmentation model.
        
        Args:
            num_classes: Number of segmentation classes
            in_channels: Number of input channels
            dropout: Dropout probability
            use_checkpoint: Whether to use gradient checkpointing (for memory efficiency)
        """
        super(RandLANetSegmentationModel, self).__init__()
        
        # Store config for reference
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint
        
        # Feature extraction backbone
        self.backbone = RandLANetBackbone(
            in_channels=in_channels,
            num_classes=num_classes,
            feature_dims=[32, 64, 128, 256],
            num_neighbors=16,
            decimation_ratio=0.25,
            use_checkpoint=use_checkpoint
        )
        
        # Segmentation head
        self.segmentation_head = nn.Sequential(
            SharedMLP(256, [256, 128]),
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
        # Forward through backbone
        _, features_list = self.backbone(points, features)
        
        # Get final features
        final_features = features_list[-1]
        
        # Apply segmentation head
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