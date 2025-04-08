# urban_point_cloud_analyzer/optimization/m1_optimizations.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
import platform
import os
import functools
import time

def is_m1_mac() -> bool:
    """
    Check if running on Apple M1 Mac.
    
    Returns:
        True if running on M1 Mac, False otherwise
    """
    if platform.system() != "Darwin":
        return False
    
    # Check if ARM64 architecture (M1)
    return platform.machine() == "arm64"

def is_mps_available() -> bool:
    """
    Check if MPS (Metal Performance Shaders) is available.
    
    Returns:
        True if MPS is available, False otherwise
    """
    if not is_m1_mac():
        return False
    
    # Check if PyTorch has MPS support
    return hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available') and torch.mps.is_available()

def get_optimal_device() -> torch.device:
    """
    Get optimal device for current hardware.
    
    Returns:
        torch.device for computation
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif is_mps_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

class M1Optimizer:
    """
    Optimize operations for M1 Mac GPUs using MPS.
    """
    
    @staticmethod
    def optimize_model(model: nn.Module, config: Optional[Dict] = None) -> nn.Module:
        """
        Optimize model for M1 Mac with robust error handling.
        
        Args:
            model: PyTorch model
            config: Optional configuration
            
        Returns:
            Optimized model
        """
        if not is_m1_mac():
            return model
        
        # Move model to MPS device if available
        device = get_optimal_device()
        model = model.to(device)
        
        # Default config
        if config is None:
            config = {}
        
        # Convert to float16 for better performance if specified - with error handling
        if config.get('half_precision', True) and device.type == 'mps':
            try:
                # Try direct half conversion
                model = model.half()
            except RuntimeError as e:
                if "Input type (float) and bias type" in str(e):
                    # Selective half conversion for parameters that support it
                    print("Applying selective half-precision conversion for M1")
                    for name, param in model.named_parameters():
                        if 'bias' not in name:  # Skip bias parameters
                            try:
                                param.data = param.data.half()
                            except Exception:
                                pass  # Skip parameters that can't be converted
                else:
                    # For other errors, log and continue
                    print(f"Warning: Half precision failed: {e}")
        
        # Enable FP16 training if available
        if device.type == 'mps' and hasattr(torch, '_dynamo'):
            # Enable TorchDynamo compiler for M1 optimizations if available
            try:
                torch._dynamo.config.use_dynamic_shapes = True
                model = torch._dynamo.optimize("inductor")(model)
            except (AttributeError, ImportError):
                pass
        
        return model
    
    @staticmethod
    def optimize_memory_usage(model: nn.Module) -> nn.Module:
        """
        Apply memory optimizations for M1 Mac's limited unified memory.
        
        Args:
            model: PyTorch model
            
        Returns:
            Memory-optimized model
        """
        if not is_m1_mac():
            return model
        
        # Enable gradient checkpointing if supported
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
        
        return model
    
    @staticmethod
    def get_optimal_batch_size(model: nn.Module, input_shape: Tuple[int, ...], 
                              memory_limit: float = 7.0) -> int:
        """
        Determine optimal batch size for M1 Mac's unified memory.
        
        Args:
            model: PyTorch model
            input_shape: Input shape (without batch dimension)
            memory_limit: Memory limit in GB (default: 7GB for 8GB M1)
            
        Returns:
            Optimal batch size
        """
        if not is_m1_mac():
            return 32  # Default batch size for non-M1
        
        # Conservative batch size for M1 Mac
        default_batch_size = 2
        
        # Try to estimate batch size if model is loaded
        try:
            device = get_optimal_device()
            model = model.to(device)
            
            # Create test inputs with batch size 1
            input_shape_with_batch = (1,) + tuple(input_shape)
            test_input = torch.randn(input_shape_with_batch).to(device)
            
            # Run forward pass to measure memory
            with torch.no_grad():
                _ = model(test_input)
            
            # Estimate memory per sample (rough approximation)
            # On M1, we need to estimate since mps doesn't expose memory stats directly
            # Based on empirical testing, use conservative estimate
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            input_size_mb = np.prod(input_shape) * 4 / (1024 * 1024)  # Assuming float32
            
            # Memory usage per batch item (with overhead factor)
            overhead_factor = 3.0
            memory_per_batch_mb = (model_size_mb + input_size_mb) * overhead_factor
            
            # Calculate batch size based on available memory
            # Convert memory_limit from GB to MB
            available_memory_mb = memory_limit * 1024
            optimal_batch_size = int(available_memory_mb / memory_per_batch_mb)
            
            # Ensure at least 1, at most 32
            optimal_batch_size = max(1, min(32, optimal_batch_size))
            
            return optimal_batch_size
            
        except Exception as e:
            print(f"Error estimating batch size: {e}")
            return default_batch_size

    @staticmethod
    def enable_mps_acceleration(module_fn, fallback_fn=None):
        """
        Decorator to enable MPS acceleration with fallback.
        
        Args:
            module_fn: Function to be accelerated
            fallback_fn: Optional fallback function
            
        Returns:
            Accelerated function with fallback
        """
        @functools.wraps(module_fn)
        def wrapper(*args, **kwargs):
            if is_mps_available():
                # Move tensor args to MPS
                mps_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        mps_args.append(arg.to('mps'))
                    else:
                        mps_args.append(arg)
                
                # Move tensor kwargs to MPS
                mps_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        mps_kwargs[k] = v.to('mps')
                    else:
                        mps_kwargs[k] = v
                
                # Run on MPS
                result = module_fn(*mps_args, **mps_kwargs)
                
                # Move result back to CPU if tensor
                if isinstance(result, torch.Tensor):
                    return result.to('cpu')
                else:
                    return result
            elif fallback_fn is not None:
                return fallback_fn(*args, **kwargs)
            else:
                return module_fn(*args, **kwargs)
        
        return wrapper
    
    @staticmethod
    def chunk_large_pointcloud(points: torch.Tensor, 
                             features: Optional[torch.Tensor] = None, 
                             max_points_per_chunk: int = 16384) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Split large point clouds into chunks for processing on M1 Mac.
        
        Args:
            points: (N, 3) tensor of point coordinates
            features: Optional (N, C) tensor of point features
            max_points_per_chunk: Maximum points per chunk
            
        Returns:
            List of (points, features) tuples for each chunk
        """
        num_points = points.shape[0]
        
        # If small enough, return as is
        if num_points <= max_points_per_chunk:
            return [(points, features)]
        
        # Split into chunks
        num_chunks = (num_points + max_points_per_chunk - 1) // max_points_per_chunk
        chunks = []
        
        for i in range(num_chunks):
            start_idx = i * max_points_per_chunk
            end_idx = min((i + 1) * max_points_per_chunk, num_points)
            
            points_chunk = points[start_idx:end_idx]
            
            if features is not None:
                features_chunk = features[start_idx:end_idx]
            else:
                features_chunk = None
            
            chunks.append((points_chunk, features_chunk))
        
        return chunks

    @staticmethod
    def merge_chunk_results(results: List[torch.Tensor], 
                           original_shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        """
        Merge results from processed chunks.
        
        Args:
            results: List of tensors from each chunk
            original_shape: Optional original shape for reshaping
            
        Returns:
            Merged result tensor
        """
        # Simply concatenate along first dimension
        merged = torch.cat(results, dim=0)
        
        # Reshape if original shape provided
        if original_shape is not None:
            merged = merged.reshape(original_shape)
        
        return merged

def benchmark_m1_optimizations(model: nn.Module, 
                              input_shape: Tuple[int, ...], 
                              num_runs: int = 10) -> Dict:
    """
    Benchmark M1-specific optimizations.
    
    Args:
        model: PyTorch model
        input_shape: Input shape
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with benchmark results
    """
    # Skip if not on M1 Mac
    if not is_m1_mac():
        return {
            'cpu_time': 0,
            'mps_time': 0,
            'mps_fp16_time': 0,
            'speedup': 0
        }
    
    # Check if MPS is available
    mps_available = is_mps_available()
    
    # Create random input
    input_data = torch.randn(*input_shape)
    
    # Benchmark on CPU
    model.to('cpu').float()
    
    # Warmup
    with torch.no_grad():
        _ = model(input_data)
    
    # Benchmark
    cpu_start = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(input_data)
    
    cpu_time = (time.time() - cpu_start) / num_runs
    
    # Benchmark on MPS if available
    if mps_available:
        # Move to MPS
        model.to('mps').float()
        input_data_mps = input_data.to('mps')
        
        # Warmup
        with torch.no_grad():
            _ = model(input_data_mps)
        
        # Benchmark
        mps_start = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(input_data_mps)
        
        mps_time = (time.time() - mps_start) / num_runs
        
        # Benchmark MPS FP16
        model.to('mps').half()
        input_data_mps = input_data.to('mps').half()
        
        # Warmup
        with torch.no_grad():
            _ = model(input_data_mps)
        
        # Benchmark
        mps_fp16_start = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(input_data_mps)
        
        mps_fp16_time = (time.time() - mps_fp16_start) / num_runs
        
        # Calculate speedup
        speedup_mps = cpu_time / mps_time
        speedup_mps_fp16 = cpu_time / mps_fp16_time
    else:
        mps_time = 0
        mps_fp16_time = 0
        speedup_mps = 0
        speedup_mps_fp16 = 0
    
    return {
        'cpu_time': cpu_time,
        'mps_time': mps_time,
        'mps_fp16_time': mps_fp16_time,
        'speedup_mps': speedup_mps,
        'speedup_mps_fp16': speedup_mps_fp16
    }