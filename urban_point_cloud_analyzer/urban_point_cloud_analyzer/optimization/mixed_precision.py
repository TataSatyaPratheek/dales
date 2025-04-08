# urban_point_cloud_analyzer/optimization/mixed_precision.py
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional, Tuple, Union, Callable
import functools
import time

class MixedPrecisionTrainer:
    """
    Mixed precision training implementation optimized for NVIDIA 1650Ti GPU.
    Reduces memory usage by using lower precision (float16) for appropriate operations.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 scaler: Optional[GradScaler] = None,
                 enabled: bool = True):
        """
        Initialize mixed precision trainer.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            scaler: Optional GradScaler instance
            enabled: Whether to enable mixed precision training
        """
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler if scaler is not None else GradScaler(enabled=enabled)
        self.enabled = enabled and torch.cuda.is_available()
    
    def train_step(self, 
                  data: torch.Tensor, 
                  target: torch.Tensor, 
                  loss_fn: Callable) -> Tuple[torch.Tensor, float]:
        """
        Perform a single training step with mixed precision.
        
        Args:
            data: Input data tensor
            target: Target tensor
            loss_fn: Loss function
            
        Returns:
            Tuple of (loss, accuracy)
        """
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast(enabled=self.enabled):
            output = self.model(data)
            loss = loss_fn(output, target)
        
        # Backward pass with scaler
        self.scaler.scale(loss).backward()
        
        # Step with scaler
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Calculate accuracy
        with torch.no_grad():
            pred = output.argmax(dim=1)
            correct = (pred == target).sum().item()
            total = target.numel()
            accuracy = correct / total
        
        return loss, accuracy
    
    def validate_step(self, 
                    data: torch.Tensor, 
                    target: torch.Tensor, 
                    loss_fn: Callable) -> Tuple[torch.Tensor, float]:
        """
        Perform a single validation step with mixed precision.
        
        Args:
            data: Input data tensor
            target: Target tensor
            loss_fn: Loss function
            
        Returns:
            Tuple of (loss, accuracy)
        """
        # No gradients needed for validation
        with torch.no_grad():
            # Forward pass with autocast
            with autocast(enabled=self.enabled):
                output = self.model(data)
                loss = loss_fn(output, target)
            
            # Calculate accuracy
            pred = output.argmax(dim=1)
            correct = (pred == target).sum().item()
            total = target.numel()
            accuracy = correct / total
        
        return loss, accuracy


def apply_mixed_precision(model: nn.Module) -> nn.Module:
    """
    Apply mixed precision to model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model with mixed precision applied
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        return model
    
    # Add autocast to forward method
    original_forward = model.forward
    
    @functools.wraps(original_forward)
    def forward_with_autocast(*args, **kwargs):
        with autocast():
            return original_forward(*args, **kwargs)
    
    model.forward = forward_with_autocast
    
    return model


def benchmark_mixed_precision(model: nn.Module, 
                             input_shape: Tuple[int, ...], 
                             num_runs: int = 10) -> Dict:
    """
    Benchmark mixed precision vs. full precision.
    
    Args:
        model: PyTorch model
        input_shape: Input shape
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with benchmark results
    """
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        return {
            'fp32_time': 0,
            'fp16_time': 0,
            'fp32_memory': 0,
            'fp16_memory': 0,
            'speedup': 0,
            'memory_saving': 0
        }
    
    # Create random input
    input_data = torch.randn(*input_shape, device='cuda')
    
    # Move model to GPU
    model.cuda()
    
    # Benchmark FP32
    model.float()
    torch.cuda.reset_peak_memory_stats()
    
    # Warmup
    with torch.no_grad():
        _ = model(input_data.float())
    
    # Benchmark
    torch.cuda.synchronize()
    fp32_start = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(input_data.float())
    
    torch.cuda.synchronize()
    fp32_time = (time.time() - fp32_start) / num_runs
    fp32_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    
    # Clear memory
    torch.cuda.empty_cache()
    
    # Benchmark FP16 with autocast
    torch.cuda.reset_peak_memory_stats()
    
    # Warmup
    with torch.no_grad(), autocast():
        _ = model(input_data)
    
    # Benchmark
    torch.cuda.synchronize()
    fp16_start = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad(), autocast():
            _ = model(input_data)
    
    torch.cuda.synchronize()
    fp16_time = (time.time() - fp16_start) / num_runs
    fp16_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    
    # Clear memory
    torch.cuda.empty_cache()
    
    # Calculate speedup and memory saving
    speedup = fp32_time / fp16_time
    memory_saving = (1 - fp16_memory / fp32_memory) * 100
    
    return {
        'fp32_time': fp32_time,
        'fp16_time': fp16_time,
        'fp32_memory': fp32_memory,
        'fp16_memory': fp16_memory,
        'speedup': speedup,
        'memory_saving': memory_saving
    }


def optimize_model_for_1650ti(model: nn.Module) -> nn.Module:
    """
    Apply optimizations specific to NVIDIA GeForce GTX 1650Ti.
    
    Args:
        model: PyTorch model
        
    Returns:
        Optimized model
    """
    # Don't apply if CUDA is not available
    if not torch.cuda.is_available():
        return model
    
    # Check if we're running on 1650Ti
    device_name = torch.cuda.get_device_name(0)
    is_1650ti = "1650 Ti" in device_name
    
    if not is_1650ti:
        return model
    
    # 1. Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    elif hasattr(model, 'apply'):
        # Try to find modules that support gradient checkpointing
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
    
    # 2. Apply mixed precision
    model = apply_mixed_precision(model)
    
    # 3. Configure CUDA for memory efficiency
    # Limit memory splitting to avoid fragmentation
    if hasattr(torch.cuda, 'memory') and hasattr(torch.cuda.memory, '_set_allocator_settings'):
        torch.cuda.memory._set_allocator_settings('max_split_size_mb:128')
    
    # 4. Use cudnn benchmark for faster convolutions if feasible for model size
    torch.backends.cudnn.benchmark = True
    
    # 5. Disable cudnn determinism for better performance
    torch.backends.cudnn.deterministic = False
    
    return model