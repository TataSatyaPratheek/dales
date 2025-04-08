# urban_point_cloud_analyzer/optimization/quantization/quantization_apple.py
import torch
import torch.nn as nn
from typing import Dict, Optional

class M1Optimizer:
    """
    Optimize models for Apple M1 Macs.
    """
    
    @staticmethod
    def optimize_model_for_m1(model: nn.Module, config: Dict) -> nn.Module:
        """
        Apply M1-specific optimizations.
        
        Args:
            model: Model to optimize
            config: Configuration dictionary
            
        Returns:
            Optimized model
        """
        # Check if running on MPS (Metal Performance Shaders)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Move model to MPS device
            model = model.to('mps')
            print("Model moved to MPS device")
            
            # Convert model to float16 for better performance on M1
            if config.get('precision', 'float32') == 'float16':
                model = model.half()
                print("Model converted to float16")
        
        # Apply memory-efficient optimizations
        model = M1Optimizer.apply_memory_optimizations(model)
        
        return model
    
    @staticmethod
    def apply_memory_optimizations(model: nn.Module) -> nn.Module:
        """
        Apply memory-efficient optimizations.
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimized model
        """
        # Fuse batch normalization with convolutions where possible
        model = torch.quantization.fuse_modules(model, [['conv', 'bn']], inplace=True)
        
        # Enable gradient checkpointing if available
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
        
        return model
    
    @staticmethod
    def get_optimal_chunk_size(points: torch.Tensor, model: nn.Module) -> int:
        """
        Determine optimal chunk size for processing large point clouds on M1.
        
        Args:
            points: Input point cloud
            model: Model to use
            
        Returns:
            Optimal chunk size
        """
        # Default chunk size
        default_chunk_size = 4096
        
        # If not MPS, return default
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            return default_chunk_size
        
        # Get available memory (estimate - MPS doesn't expose memory stats directly)
        # For M1 with 8GB RAM, reserve about 4GB for ML tasks
        available_memory = 4 * 1024 * 1024 * 1024  # 4GB in bytes
        
        # Estimate memory per point
        # Each point has coordinates and features, plus model overhead
        bytes_per_point = 4 * (3 + model.in_channels)  # float32 * (xyz + features)
        
        # Add 50% overhead for model parameters and intermediates
        bytes_per_point *= 1.5
        
        # Calculate maximum points
        max_points = int(available_memory / bytes_per_point)
        
        # Limit to actual point count
        max_points = min(max_points, points.shape[0] if points.dim() == 2 else points.shape[1])
        
        # Round to power of 2 for better memory alignment
        chunk_size = 2 ** int(np.log2(max_points))
        
        # Ensure it's not too small
        chunk_size = max(chunk_size, 1024)
        
        return chunk_size