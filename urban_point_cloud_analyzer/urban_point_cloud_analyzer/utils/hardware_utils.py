# urban_point_cloud_analyzer/utils/hardware_utils.py
import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

def get_device_info() -> Dict:
    """
    Get information about the current device.
    
    Returns:
        Dictionary with device information
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
    }
    
    if device_info['cuda_available']:
        # Get CUDA device properties
        device_properties = torch.cuda.get_device_properties(0)
        
        device_info.update({
            'device_name': device_properties.name,
            'total_memory': device_properties.total_memory / (1024 ** 3),  # Convert to GB
            'compute_capability': f"{device_properties.major}.{device_properties.minor}",
            'multi_processor_count': device_properties.multi_processor_count
        })
        
        # Check if device is NVIDIA GeForce GTX 1650 Ti
        device_info['is_1650ti'] = "1650 Ti" in device_properties.name
    
    # Check if device is M1 Mac
    device_info['is_m1'] = "Darwin" in os.uname().sysname and "ARM64" in os.uname().machine
    
    return device_info

def get_optimal_config(device_info: Dict) -> Dict:
    """
    Get optimal configuration for the current device.
    
    Args:
        device_info: Dictionary with device information
        
    Returns:
        Dictionary with optimal configuration
    """
    config = {}
    
    # Default batch sizes
    config['train_batch_size'] = 8
    config['val_batch_size'] = 16
    config['inference_batch_size'] = 1
    
    # Default optimization settings
    config['mixed_precision'] = False
    config['gradient_checkpointing'] = False
    config['model_quantization'] = False
    config['custom_cuda_kernels'] = False
    
    # NVIDIA GeForce GTX 1650 Ti optimizations
    if device_info.get('is_1650ti', False):
        # For 1650 Ti (4GB VRAM)
        config['train_batch_size'] = 4
        config['val_batch_size'] = 8
        config['inference_batch_size'] = 1
        config['mixed_precision'] = True
        config['gradient_checkpointing'] = True
        config['custom_cuda_kernels'] = True
        config['num_workers'] = 4
    
    # M1 Mac optimizations
    elif device_info.get('is_m1', False):
        # For M1 Mac (8GB shared memory)
        config['train_batch_size'] = 2
        config['val_batch_size'] = 4
        config['inference_batch_size'] = 1
        config['model_quantization'] = True
        config['precision'] = 'int8'
        config['num_workers'] = 2
    
    # General CUDA optimizations
    elif device_info.get('cuda_available', False):
        # For other CUDA devices
        config['mixed_precision'] = True
        config['custom_cuda_kernels'] = True
        config['num_workers'] = min(8, os.cpu_count() or 4)
    
    # CPU fallback
    else:
        config['train_batch_size'] = 1
        config['val_batch_size'] = 2
        config['inference_batch_size'] = 1
        config['model_quantization'] = True
        config['precision'] = 'int8'
        config['num_workers'] = 1
    
    return config

def optimize_dataloader_for_device(dataloader_config: Dict, device_info: Dict) -> Dict:
    """
    Optimize dataloader configuration for the current device.
    
    Args:
        dataloader_config: DataLoader configuration
        device_info: Dictionary with device information
        
    Returns:
        Optimized DataLoader configuration
    """
    optimized_config = dataloader_config.copy()
    
    # Get device-specific optimal config
    optimal_config = get_optimal_config(device_info)
    
    # Update batch size
    if 'batch_size' in dataloader_config:
        split = dataloader_config.get('split', 'train')
        
        if split == 'train':
            optimized_config['batch_size'] = optimal_config['train_batch_size']
        elif split == 'val':
            optimized_config['batch_size'] = optimal_config['val_batch_size']
        else:
            optimized_config['batch_size'] = optimal_config['inference_batch_size']
    
    # Set number of workers
    optimized_config['num_workers'] = optimal_config['num_workers']
    
    # Set pin_memory for CUDA
    optimized_config['pin_memory'] = device_info.get('cuda_available', False)
    
    # Set prefetch_factor for better GPU utilization
    if device_info.get('cuda_available', False):
        optimized_config['prefetch_factor'] = 2
    
    return optimized_config

def optimize_model_for_device(model: torch.nn.Module, device_info: Dict) -> torch.nn.Module:
    """
    Optimize model for the current device.
    
    Args:
        model: Model to optimize
        device_info: Dictionary with device information
        
    Returns:
        Optimized model
    """
    # Get device-specific optimal config
    optimal_config = get_optimal_config(device_info)
    
    # Apply optimizations based on device
    if device_info.get('cuda_available', False):
        # CUDA optimizations
        model = model.cuda()
        
        # Apply mixed precision
        if optimal_config['mixed_precision']:
            from torch.cuda.amp import autocast
            model._forward = model.forward
            model.forward = lambda *args, **kwargs: autocast()(model._forward)(*args, **kwargs)
        
        # Enable gradient checkpointing for memory efficiency
        if optimal_config['gradient_checkpointing'] and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    
    elif device_info.get('mps_available', False):
        # M1 Mac optimizations
        model = model.to('mps')
    
    # Apply quantization for M1 or CPU
    if optimal_config['model_quantization'] and not device_info.get('cuda_available', False):
        # Avoid circular import
        from urban_point_cloud_analyzer.optimization.quantization import prepare_model_for_quantization
        
        model = prepare_model_for_quantization(model, {
            'enabled': True,
            'precision': optimal_config['precision']
        })
    
    return model