# urban_point_cloud_analyzer/optimization/hardware_optimizations.py
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Callable
import os
import platform
import functools

def detect_hardware() -> Dict:
    """
    Detect hardware features and capabilities.
    
    Returns:
        Dictionary with hardware information
    """
    hardware_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cpu_count': os.cpu_count() or 1,
        'platform': platform.system(),
        'architecture': platform.machine(),
        'is_m1_mac': platform.system() == "Darwin" and platform.machine() == "arm64",
        'mps_available': hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available') and torch.mps.is_available(),
    }
    
    if hardware_info['cuda_available']:
        # Get CUDA device properties
        device_properties = torch.cuda.get_device_properties(0)
        
        hardware_info.update({
            'cuda_device_name': device_properties.name,
            'cuda_total_memory': device_properties.total_memory / (1024 ** 3),  # Convert to GB
            'cuda_compute_capability': f"{device_properties.major}.{device_properties.minor}",
            'cuda_multi_processor_count': device_properties.multi_processor_count,
            'is_1650ti': "1650 Ti" in device_properties.name
        })
    
    return hardware_info

def optimize_for_hardware(model: nn.Module, hardware_info: Optional[Dict] = None) -> nn.Module:
    """
    Apply hardware-specific optimizations to model.
    
    Args:
        model: PyTorch model
        hardware_info: Optional hardware information dictionary
        
    Returns:
        Optimized model
    """
    # Detect hardware if not provided
    if hardware_info is None:
        hardware_info = detect_hardware()
    
    # Apply optimizations based on hardware
    if hardware_info.get('is_1650ti', False):
        # Apply 1650Ti optimizations
        from urban_point_cloud_analyzer.optimization.mixed_precision import optimize_model_for_1650ti
        model = optimize_model_for_1650ti(model)
        
    elif hardware_info.get('is_m1_mac', False):
        # Apply M1 Mac optimizations
        from urban_point_cloud_analyzer.optimization.m1_optimizations import M1Optimizer
        model = M1Optimizer.optimize_model(model)
        model = M1Optimizer.optimize_memory_usage(model)
        
    elif hardware_info.get('cuda_available', False):
        # Apply general CUDA optimizations
        model = model.cuda()
        
        # Enable mixed precision if available
        if torch.cuda.is_available() and hasattr(torch.cuda, 'amp'):
            from urban_point_cloud_analyzer.optimization.mixed_precision import apply_mixed_precision
            model = apply_mixed_precision(model)
    
    return model

def get_optimal_batch_size(model: nn.Module, 
                          input_shape: Tuple[int, ...], 
                          hardware_info: Optional[Dict] = None) -> int:
    """
    Get optimal batch size for the current hardware.
    
    Args:
        model: PyTorch model
        input_shape: Input shape (without batch dimension)
        hardware_info: Optional hardware information dictionary
        
    Returns:
        Optimal batch size
    """
    # Detect hardware if not provided
    if hardware_info is None:
        hardware_info = detect_hardware()
    
    # Get optimal batch size based on hardware
    if hardware_info.get('is_1650ti', False):
        # Conservative batch size for 1650Ti (4GB VRAM)
        # Start with a safe default
        batch_size = 4
        
        # Try to estimate better batch size if CUDA is available
        if torch.cuda.is_available():
            try:
                # Simple test with increasing batch sizes
                model = model.cuda()
                max_batch = 16  # Upper limit to try
                
                for bs in range(4, max_batch + 1, 2):
                    try:
                        # Create test input
                        test_input = torch.randn(bs, *input_shape, device='cuda')
                        
                        # Run forward pass
                        with torch.no_grad():
                            _ = model(test_input)
                            
                        # If successful, update batch size
                        batch_size = bs
                    except RuntimeError:
                        # Out of memory, use previous batch size
                        break
                    
                    # Clear memory after each test
                    torch.cuda.empty_cache()
            
            except Exception:
                # Fall back to default if estimation fails
                pass
        
        return batch_size
        
    elif hardware_info.get('is_m1_mac', False):
        # Get optimal batch size for M1 Mac
        from urban_point_cloud_analyzer.optimization.m1_optimizations import M1Optimizer
        return M1Optimizer.get_optimal_batch_size(model, input_shape)
        
    elif hardware_info.get('cuda_available', False):
        # For other CUDA devices, use memory-based estimation
        try:
            # Move model to CUDA
            model = model.cuda()
            
            # Get available memory
            total_memory = hardware_info.get('cuda_total_memory', 8.0)  # Default to 8GB
            
            # Reserve 1.5GB for system and other processes
            available_memory = total_memory - 1.5
            
            # Simple test with increasing batch sizes
            max_batch = 32  # Upper limit to try
            batch_size = 4  # Start with a safe default
            
            for bs in range(4, max_batch + 1, 4):
                try:
                    # Create test input
                    test_input = torch.randn(bs, *input_shape, device='cuda')
                    
                    # Run forward pass
                    with torch.no_grad():
                        _ = model(test_input)
                        
                    # If successful, update batch size
                    batch_size = bs
                except RuntimeError:
                    # Out of memory, use previous batch size
                    break
                
                # Clear memory after each test
                torch.cuda.empty_cache()
            
            return batch_size
            
        except Exception:
            # Fall back to default if estimation fails
            return 8
    else:
        # CPU default
        return 2

def optimize_dataloader_config(config: Dict, hardware_info: Optional[Dict] = None) -> Dict:
    """
    Optimize DataLoader configuration for the current hardware.
    
    Args:
        config: DataLoader configuration
        hardware_info: Optional hardware information dictionary
        
    Returns:
        Optimized DataLoader configuration
    """
    # Detect hardware if not provided
    if hardware_info is None:
        hardware_info = detect_hardware()
    
    # Create a copy of the config
    optimized_config = config.copy()
    
    # Set number of workers based on hardware
    if hardware_info.get('is_1650ti', False):
        # For 1650Ti, use fewer workers to avoid CPU memory pressure
        optimized_config['num_workers'] = min(4, hardware_info.get('cpu_count', 4))
        optimized_config['pin_memory'] = True
        
    elif hardware_info.get('is_m1_mac', False):
        # For M1 Mac, use fewer workers due to unified memory architecture
        optimized_config['num_workers'] = min(2, hardware_info.get('cpu_count', 4) // 2)
        optimized_config['pin_memory'] = False  # Not needed for MPS
        
    elif hardware_info.get('cuda_available', False):
        # For other CUDA devices
        optimized_config['num_workers'] = min(8, hardware_info.get('cpu_count', 4))
        optimized_config['pin_memory'] = True
        optimized_config['prefetch_factor'] = 2  # Prefetch 2 batches per worker
        
    else:
        # For CPU
        optimized_config['num_workers'] = min(2, hardware_info.get('cpu_count', 4))
        optimized_config['pin_memory'] = False
    
    return optimized_config

def hardware_specific_pipeline(fn: Callable) -> Callable:
    """
    Decorator to apply hardware-specific optimizations to a pipeline function.
    
    Args:
        fn: Pipeline function to optimize
        
    Returns:
        Optimized pipeline function
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Detect hardware
        hardware_info = detect_hardware()
        
        # Apply hardware-specific logic
        if hardware_info.get('is_1650ti', False):
            # Enable optimizations for 1650Ti
            if torch.cuda.is_available():
                # Use mixed precision
                from torch.cuda.amp import autocast
                
                with autocast():
                    return fn(*args, **kwargs)
            
        elif hardware_info.get('is_m1_mac', False):
            # Apply M1-specific logic
            from urban_point_cloud_analyzer.optimization.m1_optimizations import M1Optimizer
            
            # Check if we need to chunk large point clouds
            if 'points' in kwargs and isinstance(kwargs['points'], torch.Tensor):
                points = kwargs['points']
                features = kwargs.get('features', None)
                
                # If point cloud is large, process in chunks
                if points.shape[0] > 16384:
                    chunks = M1Optimizer.chunk_large_pointcloud(points, features)
                    
                    # Process each chunk
                    results = []
                    for chunk_points, chunk_features in chunks:
                        chunk_kwargs = kwargs.copy()
                        chunk_kwargs['points'] = chunk_points
                        if chunk_features is not None:
                            chunk_kwargs['features'] = chunk_features
                        
                        # Process chunk
                        chunk_result = fn(*args, **chunk_kwargs)
                        results.append(chunk_result)
                    
                    # Merge results
                    return M1Optimizer.merge_chunk_results(results)
        
        # No special handling needed, just call the function
        return fn(*args, **kwargs)
    
    return wrapper

def enable_sparse_operations(use_sparse: bool = True) -> Callable:
    """
    Decorator to enable sparse operations for memory efficiency.
    
    Args:
        use_sparse: Whether to use sparse operations
        
    Returns:
        Decorator function
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Only use sparse ops if specified and we're on CUDA
            if use_sparse and torch.cuda.is_available():
                # Import sparse ops
                from urban_point_cloud_analyzer.optimization.sparse_ops import (
                    create_sparse_tensor, SparseTensor
                )
                
                # Check if we have point cloud data
                if 'points' in kwargs and isinstance(kwargs['points'], torch.Tensor):
                    points = kwargs['points']
                    features = kwargs.get('features', None)
                    
                    # Convert to sparse representation
                    sparse_tensor = create_sparse_tensor(points, features)
                    
                    # Replace inputs with sparse tensor
                    sparse_kwargs = kwargs.copy()
                    sparse_kwargs['points'] = sparse_tensor
                    
                    if 'features' in sparse_kwargs:
                        del sparse_kwargs['features']  # Features are included in sparse tensor
                    
                    # Call function with sparse inputs
                    return fn(*args, **sparse_kwargs)
            
            # Standard processing without sparse ops
            return fn(*args, **kwargs)
        
        return wrapper
    
    return decorator