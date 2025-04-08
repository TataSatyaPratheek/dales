# urban_point_cloud_analyzer/tests/test_utils.py
import os
import sys
import torch
import platform
from pathlib import Path
from functools import wraps

def is_cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()

def is_1650ti():
    """Check if the GPU is an NVIDIA GeForce GTX 1650 Ti."""
    if not is_cuda_available():
        return False
    
    device_name = torch.cuda.get_device_name(0)
    return "1650 Ti" in device_name

def is_m1_mac():
    """Check if running on Apple M1 Mac."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"

def is_mps_available():
    """Check if MPS (Metal Performance Shaders) is available."""
    return hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available') and torch.mps.is_available()

def skip_if_no_cuda(func):
    """Decorator to skip test if CUDA is not available."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not is_cuda_available():
            print(f"⚠ Skipping {func.__name__} as CUDA is not available")
            return True
        return func(*args, **kwargs)
    return wrapper

def skip_on_m1(func):
    """Decorator to skip test on M1 Mac."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_m1_mac():
            print(f"⚠ Skipping {func.__name__} on M1 Mac")
            return True
        return func(*args, **kwargs)
    return wrapper

def create_small_point_cloud(num_points=1000, num_classes=8):
    """Create a small point cloud for testing."""
    import numpy as np
    
    # Create points
    points = np.random.rand(num_points, 3).astype(np.float32)
    
    # Create segmentation with all classes
    segmentation = np.zeros(num_points, dtype=np.int32)
    segment_size = num_points // num_classes
    
    for i in range(num_classes):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < num_classes - 1 else num_points
        segmentation[start_idx:end_idx] = i
    
    return points, segmentation