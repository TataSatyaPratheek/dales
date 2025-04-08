# urban_point_cloud_analyzer/optimization/cuda/__init__.py
# Try to load the custom CUDA extension
import os
import torch
from torch.utils.cpp_extension import load

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Try to load the CUDA extension
try:
    knn_cuda = load(
        name="knn_cuda",
        sources=[
            os.path.join(current_dir, "knn_cuda.cpp"),
            os.path.join(current_dir, "knn_cuda_kernel.cu")
        ],
        verbose=True
    )
    
    def k_nearest_neighbors(x, k):
        """
        Find k-nearest neighbors for each point
        
        Args:
            x: (B, N, 3) tensor of point coordinates
            k: number of neighbors
            
        Returns:
            (B, N, k) tensor of indices
        """
        return knn_cuda.knn(x, k)
    
except Exception as e:
    print(f"Failed to load CUDA extension: {e}")
    print("Falling back to CPU implementation")
    
    def k_nearest_neighbors(x, k):
        """
        CPU fallback implementation of k-nearest neighbors
        
        Args:
            x: (B, N, 3) tensor of point coordinates
            k: number of neighbors
            
        Returns:
            (B, N, k) tensor of indices
        """
        batch_size, num_points, _ = x.size()
        device = x.device
        
        # Compute squared distances
        x_trans = x.transpose(1, 2).contiguous()  # (B, 3, N)
        inner = torch.matmul(x, x_trans)  # (B, N, N)
        
        xx = torch.sum(x**2, dim=2, keepdim=True)  # (B, N, 1)
        dist = xx + xx.transpose(1, 2) - 2 * inner  # (B, N, N)
        
        # Set self-distances to large value
        dist = dist.clone()
        eye = torch.eye(num_points, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        dist = dist + eye * 1e10
        
        # Find k nearest neighbors
        _, indices = torch.topk(dist, k=k, dim=2, largest=False, sorted=True)
        
        return indices

__all__ = ['k_nearest_neighbors']