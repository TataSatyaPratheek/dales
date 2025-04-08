# urban_point_cloud_analyzer/optimization/sparse_ops.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

def create_sparse_tensor(points: Union[np.ndarray, torch.Tensor], 
                          features: Optional[Union[np.ndarray, torch.Tensor]] = None) -> torch.Tensor:
    """
    Create a sparse tensor from point cloud data.
    Optimized for memory efficiency on GPU.
    
    Args:
        points: (N, 3) array of point coordinates
        features: Optional (N, C) array of point features
        
    Returns:
        Sparse tensor representation
    """
    # Convert to torch tensors if needed
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points).float()
    
    if features is not None and isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()
    
    # Use torch.sparse if available for efficient GPU memory usage
    try:
        import torch_sparse
        # Create coordinate tensor (3, N)
        coords = points.transpose(0, 1).contiguous()
        
        # Create sparse tensor
        if features is not None:
            sparse_tensor = torch.sparse.FloatTensor(
                coords.long(), features, 
                torch.Size([points[:, 0].max() + 1, points[:, 1].max() + 1, points[:, 2].max() + 1, features.shape[1]])
            )
        else:
            sparse_tensor = torch.sparse.FloatTensor(
                coords.long(), torch.ones(points.shape[0]),
                torch.Size([points[:, 0].max() + 1, points[:, 1].max() + 1, points[:, 2].max() + 1])
            )
        
        return sparse_tensor
        
    except (ImportError, ModuleNotFoundError):
        # Fallback to dense tensor with custom class if torch_sparse not available
        return SparseTensor(points, features)

class SparseTensor:
    """
    Custom sparse tensor representation for point clouds.
    Optimized for memory efficiency on the 1650Ti GPU.
    """
    def __init__(self, indices: torch.Tensor, values: Optional[torch.Tensor] = None):
        """
        Initialize SparseTensor.
        
        Args:
            indices: (N, 3) tensor of point coordinates
            values: Optional (N, C) tensor of point features
        """
        self.indices = indices
        self.values = values if values is not None else torch.ones(indices.shape[0], dtype=torch.float32)
        
        # Get dimensions from point coordinates
        self.spatial_shape = [
            indices[:, 0].max().item() + 1,
            indices[:, 1].max().item() + 1,
            indices[:, 2].max().item() + 1
        ]
        
        # Create a coordinate hash for fast lookups
        self._create_coord_hash()
    
    def _create_coord_hash(self):
        """Create a coordinate hash for fast lookups."""
        # Simple spatial hashing for fast neighbor lookup
        self.coord_hash = {}
        
        for i, coords in enumerate(self.indices):
            key = tuple(coords.tolist())
            self.coord_hash[key] = i
    
    def to(self, device):
        """Move tensor to device."""
        self.indices = self.indices.to(device)
        self.values = self.values.to(device)
        return self
    
    def find_neighbors(self, coords: torch.Tensor, radius: float = 1.0) -> List[int]:
        """
        Find neighbors within radius.
        
        Args:
            coords: (3,) tensor of coordinates
            radius: Search radius
            
        Returns:
            List of neighbor indices
        """
        coords_np = coords.cpu().numpy()
        neighbors = []
        
        # Simple radius search
        for i, idx_coords in enumerate(self.indices.cpu().numpy()):
            dist = np.linalg.norm(idx_coords - coords_np)
            if dist <= radius:
                neighbors.append(i)
        
        return neighbors


class SparseConvolution(nn.Module):
    """
    Memory-efficient sparse convolution for point clouds.
    Optimized for the NVIDIA 1650Ti GPU.
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 use_checkpoint: bool = True):
        """
        Initialize sparse convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size
            stride: Stride
            padding: Padding
            use_checkpoint: Whether to use gradient checkpointing
        """
        super(SparseConvolution, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_checkpoint = use_checkpoint
        
        # Create weight for sparse convolution
        self.weight = nn.Parameter(
            torch.empty(kernel_size, kernel_size, kernel_size, in_channels, out_channels)
        )
        nn.init.kaiming_uniform_(self.weight.view(-1, out_channels))
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Create kernel offsets
        self._create_kernel_offsets()
    
    def _create_kernel_offsets(self):
        """Create kernel offsets for convolution."""
        # Center of the kernel
        center = self.kernel_size // 2
        
        # Create offsets
        offsets = []
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                for k in range(self.kernel_size):
                    offset = [i - center, j - center, k - center]
                    offsets.append(offset)
        
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.long))
    
    def forward(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            points: (B, N, 3) tensor of point coordinates
            features: (B, N, C) tensor of point features
            
        Returns:
            (B, N, C_out) tensor of output features
        """
        batch_size, num_points, _ = points.shape
        device = points.device
        
        # Initialize output features
        output_features = torch.zeros(batch_size, num_points, self.out_channels, device=device)
        
        # Process each batch
        for b in range(batch_size):
            # Use voxel grid for sparse representation
            # This is a simplified version for demonstration
            # Real implementation would use a more efficient data structure
            
            # First, voxelize the points
            voxel_size = 1.0
            voxel_indices = (points[b] / voxel_size).long()
            
            # Create a dictionary from voxel indices to point indices
            voxel_dict = {}
            for i in range(num_points):
                voxel_idx = tuple(voxel_indices[i].cpu().tolist())
                if voxel_idx not in voxel_dict:
                    voxel_dict[voxel_idx] = []
                voxel_dict[voxel_idx].append(i)
            
            # For each point, find neighbors and apply convolution
            for i in range(num_points):
                voxel_idx = tuple(voxel_indices[i].cpu().tolist())
                
                # Get kernel offsets
                neighbor_indices = []
                
                # Find neighboring voxels
                for offset in self.offsets:
                    neighbor_voxel = tuple((voxel_indices[i] + offset).cpu().tolist())
                    if neighbor_voxel in voxel_dict:
                        neighbor_indices.extend(voxel_dict[neighbor_voxel])
                
                # Skip if no neighbors
                if not neighbor_indices:
                    continue
                
                # Get neighbor features
                neighbor_features = features[b, neighbor_indices]
                
                # Apply convolution weights
                # This is a simplified implementation - in practice, you would use
                # a more efficient implementation that leverages CUDA
                for j, neighbor_idx in enumerate(neighbor_indices):
                    rel_pos = voxel_indices[neighbor_idx] - voxel_indices[i] + self.kernel_size // 2
                    
                    # Check if within kernel bounds
                    if (0 <= rel_pos[0] < self.kernel_size and
                        0 <= rel_pos[1] < self.kernel_size and
                        0 <= rel_pos[2] < self.kernel_size):
                        
                        # Get kernel weight
                        kernel_weight = self.weight[rel_pos[0], rel_pos[1], rel_pos[2]]
                        
                        # Apply weight to features
                        contrib = torch.matmul(features[b, neighbor_idx].unsqueeze(0), kernel_weight)
                        output_features[b, i] += contrib.squeeze(0)
        
        # Add bias
        output_features += self.bias
        
        return output_features


def memory_usage_comparison(num_points: int, in_channels: int, out_channels: int) -> Dict:
    """
    Compare memory usage of sparse vs dense operations.
    
    Args:
        num_points: Number of points
        in_channels: Number of input channels
        out_channels: Number of output channels
        
    Returns:
        Dictionary with memory usage statistics
    """
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        return {
            'sparse_memory': 0,
            'dense_memory': 0,
            'memory_saving_percent': 0
        }
    
    # Generate random point cloud
    points = torch.rand(1, num_points, 3, device='cuda')
    features = torch.rand(1, num_points, in_channels, device='cuda')
    
    # Measure dense memory usage
    torch.cuda.reset_peak_memory_stats()
    
    # Create dense convolution
    dense_conv = nn.Conv1d(in_channels, out_channels, 1).cuda()
    
    # Forward pass
    dense_output = dense_conv(features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
    
    # Get memory usage
    dense_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    
    # Clear memory
    del dense_conv, dense_output
    torch.cuda.empty_cache()
    
    # Measure sparse memory usage
    torch.cuda.reset_peak_memory_stats()
    
    # Create sparse convolution
    sparse_conv = SparseConvolution(in_channels, out_channels).cuda()
    
    # Forward pass
    sparse_output = sparse_conv(points, features)
    
    # Get memory usage
    sparse_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    
    # Clear memory
    del sparse_conv, sparse_output
    torch.cuda.empty_cache()
    
    # Calculate memory saving
    memory_saving_percent = (1 - sparse_memory / dense_memory) * 100 if dense_memory > 0 else 0
    
    return {
        'sparse_memory': sparse_memory,
        'dense_memory': dense_memory,
        'memory_saving_percent': memory_saving_percent
    }