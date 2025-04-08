# urban_point_cloud_analyzer/optimization/cuda/gpu_memory_utils.py
import torch
import gc
from typing import Dict, Tuple, List, Optional

def profile_memory_usage(model: torch.nn.Module, 
                         input_shape: Tuple[int, int, int], 
                         batch_sizes: List[int]) -> Dict:
    """
    Profile memory usage for different batch sizes.
    
    Args:
        model: Model to profile
        input_shape: Input shape (batch_size, num_points, features)
        batch_sizes: List of batch sizes to test
        
    Returns:
        Dictionary with memory usage for each batch size
    """
    if not torch.cuda.is_available():
        return {batch_size: 0 for batch_size in batch_sizes}
    
    device = torch.device('cuda')
    model = model.to(device)
    
    results = {}
    
    for batch_size in batch_sizes:
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        
        # Create random input
        dummy_input = torch.randn(batch_size, input_shape[1], input_shape[2], device=device)
        
        # Warm up
        with torch.no_grad():
            model(dummy_input)
        
        # Reset peak memory
        torch.cuda.reset_peak_memory_stats()
        
        # Measure memory
        with torch.no_grad():
            model(dummy_input)
        
        # Get peak memory
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        results[batch_size] = peak_memory
        
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
    
    return results

def optimize_memory_usage(model: torch.nn.Module, 
                         input_shape: Tuple[int, int, int],
                         target_memory_mb: float = 3500,  # ~3.5GB target for 1650Ti (4GB VRAM)
                         increment: int = 1) -> int:
    """
    Find optimal batch size that fits within memory constraints.
    
    Args:
        model: Model to optimize
        input_shape: Input shape (batch_size, num_points, features)
        target_memory_mb: Target memory usage in MB
        increment: Batch size increment for testing
        
    Returns:
        Optimal batch size
    """
    if not torch.cuda.is_available():
        return 1
    
    device = torch.device('cuda')
    model = model.to(device)
    
    # Start with batch size of 1
    batch_size = 1
    
    while True:
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        
        # Create random input
        dummy_input = torch.randn(batch_size, input_shape[1], input_shape[2], device=device)
        
        try:
            # Warm up
            with torch.no_grad():
                model(dummy_input)
            
            # Reset peak memory
            torch.cuda.reset_peak_memory_stats()
            
            # Measure memory
            with torch.no_grad():
                model(dummy_input)
            
            # Get peak memory
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            
            # If we've exceeded target memory, return previous batch size
            if peak_memory > target_memory_mb:
                return max(1, batch_size - increment)
            
            # Try the next batch size
            batch_size += increment
            
        except RuntimeError as e:
            # Out of memory error
            if "CUDA out of memory" in str(e):
                return max(1, batch_size - increment)
            else:
                raise
    
    return batch_size

def enable_tf32():
    """Enable TF32 precision for faster computation on Ampere GPUs."""
    if torch.cuda.is_available():
        # Enable TF32 for matrix multiplications
        torch.backends.cuda.matmul.allow_tf32 = True
        # Enable TF32 for CUDNN
        torch.backends.cudnn.allow_tf32 = True

def optimize_for_1650ti():
    """Apply specific optimizations for NVIDIA GeForce GTX 1650Ti."""
    if torch.cuda.is_available():
        # Use deterministic algorithms to avoid non-deterministic kernels
        # that might be slower on 1650Ti
        torch.backends.cudnn.deterministic = False
        
        # Use auto-tuner to find the best algorithm
        torch.backends.cudnn.benchmark = True
        
        # Limit cache allocation
        torch.cuda.memory._set_allocator_settings('max_split_size_mb:512')