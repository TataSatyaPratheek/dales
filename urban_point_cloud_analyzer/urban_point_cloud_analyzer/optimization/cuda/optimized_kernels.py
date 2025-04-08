# urban_point_cloud_analyzer/optimization/cuda/optimized_kernels.py
import os
import torch
from torch.utils.cpp_extension import load
from typing import Dict, List, Optional, Tuple, Union
import time

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Try to load the optimized CUDA extension
try:
    knn_optimized = load(
        name="knn_optimized",
        sources=[
            os.path.join(current_dir, "knn_cuda_optimized.cpp"),
            os.path.join(current_dir, "knn_cuda_optimized.cu")
        ],
        verbose=True,
        extra_cuda_cflags=["-O3", "--use_fast_math"]  # Optimize for speed
    )
    
    OPTIMIZED_CUDA_AVAILABLE = True
    
except Exception as e:
    print(f"Failed to load optimized CUDA extension: {e}")
    print("Falling back to standard implementation")
    OPTIMIZED_CUDA_AVAILABLE = False

def k_nearest_neighbors_optimized(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Find k-nearest neighbors using optimized implementation.
    
    Args:
        x: (B, N, 3) tensor of point coordinates
        k: number of neighbors
        
    Returns:
        (B, N, k) tensor of indices
    """
    if OPTIMIZED_CUDA_AVAILABLE:
        return knn_optimized.knn_optimized(x, k)
    else:
        # Fall back to standard implementation
        from urban_point_cloud_analyzer.optimization.cuda import k_nearest_neighbors
        return k_nearest_neighbors(x, k)

def benchmark_knn(x: torch.Tensor, k: int, num_runs: int = 10) -> Dict:
    """
    Benchmark KNN implementations.
    
    Args:
        x: (B, N, 3) tensor of point coordinates
        k: number of neighbors
        num_runs: number of runs for benchmarking
        
    Returns:
        Dictionary with benchmark results
    """
    # Ensure tensor is on CUDA
    if not x.is_cuda and torch.cuda.is_available():
        x = x.cuda()
    
    results = {}
    
    # Benchmark standard implementation
    from urban_point_cloud_analyzer.optimization.cuda import k_nearest_neighbors
    
    # Warm-up
    _ = k_nearest_neighbors(x, k)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        _ = k_nearest_neighbors(x, k)
        torch.cuda.synchronize()
    standard_time = (time.time() - start_time) / num_runs
    
    results["standard"] = {
        "time": standard_time,
        "throughput": x.size(0) * x.size(1) / standard_time
    }
    
    # Benchmark optimized implementation if available
    if OPTIMIZED_CUDA_AVAILABLE:
        # Warm-up
        _ = knn_optimized.knn_optimized(x, k)
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            _ = knn_optimized.knn_optimized(x, k)
            torch.cuda.synchronize()
        optimized_time = (time.time() - start_time) / num_runs
        
        results["optimized"] = {
            "time": optimized_time,
            "throughput": x.size(0) * x.size(1) / optimized_time,
            "speedup": standard_time / optimized_time
        }
    
    return results