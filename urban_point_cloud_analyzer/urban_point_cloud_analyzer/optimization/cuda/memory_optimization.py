# urban_point_cloud_analyzer/optimization/cuda/memory_optimization.py
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import gc

class MemoryOptimizer:
    """
    Optimize memory usage for CUDA devices.
    Specifically optimized for GPUs with limited VRAM like the 1650Ti (4GB).
    """
    
    @staticmethod
    def free_memory():
        """Free unused memory."""
        gc.collect()
        torch.cuda.empty_cache()
    
    @staticmethod
    def get_memory_stats() -> Dict:
        """
        Get memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        if not torch.cuda.is_available():
            return {
                'allocated': 0,
                'reserved': 0,
                'total': 0,
                'free': 0
            }
        
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # MB
        free = total - allocated
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'total': total,
            'free': free
        }
    
    @staticmethod
    def print_memory_stats():
        """Print memory statistics."""
        stats = MemoryOptimizer.get_memory_stats()
        print(f"Memory stats (MB):")
        print(f"  Allocated: {stats['allocated']:.2f}")
        print(f"  Reserved:  {stats['reserved']:.2f}")
        print(f"  Total:     {stats['total']:.2f}")
        print(f"  Free:      {stats['free']:.2f}")
    
    @staticmethod
    def optimize_for_1650ti():
        """Apply specific optimizations for NVIDIA GeForce GTX 1650Ti."""
        if torch.cuda.is_available():
            # Enable benchmark mode for optimized kernels
            torch.backends.cudnn.benchmark = True
            
            # Limit memory splitting to avoid fragmentation
            # This is particularly important for GPUs with limited memory
            torch.cuda.memory._set_allocator_settings('max_split_size_mb:128')
            
            # Enable TF32 if available (not on 1650Ti, but does nothing if not supported)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    @staticmethod
    def enable_gradient_checkpointing(model: nn.Module) -> nn.Module:
        """
        Enable gradient checkpointing for memory efficiency.
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimized model
        """
        # Find all modules that support gradient checkpointing
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
            elif hasattr(module, 'checkpoint_activations'):
                module.checkpoint_activations = True
        
        return model
    
    @staticmethod
    def get_optimal_batch_size(model: nn.Module, 
                              input_shape: Tuple[int, int, int],
                              target_memory_limit: float = 0.8,  # Use 80% of available memory
                              min_batch_size: int = 1,
                              max_batch_size: int = 16) -> int:
        """
        Find optimal batch size that fits within memory constraints.
        
        Args:
            model: Model to optimize
            input_shape: Input shape (batch_size, num_points, features)
            target_memory_limit: Fraction of total memory to target (0.0-1.0)
            min_batch_size: Minimum batch size to consider
            max_batch_size: Maximum batch size to consider
            
        Returns:
            Optimal batch size
        """
        if not torch.cuda.is_available():
            return min_batch_size
        
        device = torch.device('cuda')
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        # Get total memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        target_memory = total_memory * target_memory_limit
        
        # Binary search for optimal batch size
        low = min_batch_size
        high = max_batch_size
        optimal_batch_size = min_batch_size
        
        while low <= high:
            mid = (low + high) // 2
            
            # Free memory
            MemoryOptimizer.free_memory()
            
            try:
                # Create dummy input
                dummy_input = torch.randn(mid, input_shape[1], input_shape[2], device=device)
                
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                
                # Forward pass
                with torch.no_grad():
                    model(dummy_input)
                
                # Get peak memory
                peak_memory = torch.cuda.max_memory_allocated()
                
                if peak_memory < target_memory:
                    # Can fit larger batch
                    optimal_batch_size = mid
                    low = mid + 1
                else:
                    # Need smaller batch
                    high = mid - 1
                
            except RuntimeError as e:
                # Out of memory error
                if "CUDA out of memory" in str(e):
                    high = mid - 1
                else:
                    raise
        
        return optimal_batch_size