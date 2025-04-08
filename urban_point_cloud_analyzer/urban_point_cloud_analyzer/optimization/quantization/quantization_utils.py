# urban_point_cloud_analyzer/optimization/quantization/quantization_utils.py
import torch
import torch.nn as nn
import torch.quantization as quantization
from typing import Dict, List, Optional, Tuple, Union

def is_quantization_supported() -> bool:
    """Check if quantization is supported on this platform and PyTorch build."""
    try:
        import torch.quantization
        import torch.nn as nn
        
        # Create a simple model and try to prepare it for quantization
        model = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1),
            nn.ReLU()
        )
        
        # Get default QConfig
        qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model.qconfig = qconfig
        
        # Try to prepare
        prepared_model = torch.quantization.prepare(model)
        
        # If we got here, basic quantization support works
        return True
    except Exception as e:
        # Check for specific quantization-related errors
        error_msg = str(e)
        if ("quantized::conv" in error_msg or 
            "NoQEngine" in error_msg or
            "quantization" in error_msg):
            # This is an expected error when quantization isn't supported
            return False
        # Unexpected error - let it propagate
        raise

def prepare_model_for_quantization(model: nn.Module, config: Dict) -> nn.Module:
    """
    Prepare a model for quantization using updated API with robust error handling.
    
    Args:
        model: Model to be quantized
        config: Quantization configuration
        
    Returns:
        Model prepared for quantization
    """
    # Skip if quantization is not supported
    if not is_quantization_supported():
        print("Quantization not supported on this platform, skipping preparation")
        return model
        
    precision = config.get('precision', 'int8')
    
    # Create quantization configuration safely
    try:
        # Try standard API
        if precision == 'int8':
            qconfig = torch.quantization.QConfig(
                activation=torch.quantization.default_observer,
                weight=torch.quantization.default_per_channel_weight_observer.with_args(
                    dtype=torch.qint8,
                    quant_min=-128,
                    quant_max=127
                )
            )
        elif precision == 'float16':
            qconfig = torch.quantization.float_qparams_weight_only_qconfig
        else:
            raise ValueError(f"Unsupported precision: {precision}")
        
        # Set quantization configuration
        model.qconfig = qconfig
        
        # Prepare model for quantization
        prepared_model = torch.quantization.prepare(model)
        
        return prepared_model
    except Exception as e:
        print(f"Warning: Failed to prepare model for quantization: {e}")
        return model
    
def quantize_model(model: nn.Module, config: Dict, calibration_loader: Optional = None) -> nn.Module:
    """
    Quantize a model.
    
    Args:
        model: Model to be quantized
        config: Quantization configuration
        calibration_loader: DataLoader for calibration data
        
    Returns:
        Quantized model
    """
    # Skip if quantization is disabled
    if not config.get('enabled', False):
        return model
    
    precision = config.get('precision', 'int8')
    
    # Create a copy of the model for quantization
    # This is because quantization modifies the model in-place
    model_to_quantize = type(model)(*model.__init_args__, **model.__init_kwargs__)
    model_to_quantize.load_state_dict(model.state_dict())
    
    # Prepare model for quantization
    prepared_model = prepare_model_for_quantization(model_to_quantize, config)
    
    # Calibrate model if calibration loader is provided
    if calibration_loader is not None:
        # Run calibration data through the model
        with torch.no_grad():
            for batch in calibration_loader:
                prepared_model(batch['points'])
    
    # Convert model to quantized version
    quantized_model = torch.quantization.convert(prepared_model)
    
    return quantized_model

def optimize_for_inference(model: nn.Module, device: torch.device) -> nn.Module:
    """
    Optimize model for inference.
    
    Args:
        model: Model to be optimized
        device: Device to optimize for
        
    Returns:
        Optimized model
    """
    # Fuse batch normalization layers with convolutions
    model = torch.quantization.fuse_modules(model, ['conv', 'bn', 'relu'])
    
    # If on GPU, use CUDA graphs for faster inference
    if device.type == 'cuda':
        # Trace model with JIT for better optimization
        example_input = torch.randn(1, 16384, 3, device=device)
        traced_model = torch.jit.trace(model, example_input)
        
        return traced_model
    
    # If on CPU (like M1), use MPS backend if available
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return model.to('mps')
    
    return model

def get_optimal_batch_size(model: nn.Module, 
                          input_size: Tuple[int, int, int], 
                          target_memory_usage: float = 0.8, 
                          device: torch.device = torch.device('cuda')) -> int:
    """
    Find optimal batch size for the given model and hardware.
    
    Args:
        model: Model to analyze
        input_size: Size of input tensor (batch_size, num_points, features)
        target_memory_usage: Target GPU memory usage (0.0-1.0)
        device: Device to analyze for
        
    Returns:
        Optimal batch size
    """
    if device.type != 'cuda':
        # For CPU or MPS, use a conservative batch size
        return 4
    
    # Get total GPU memory
    total_memory = torch.cuda.get_device_properties(device).total_memory
    target_memory = total_memory * target_memory_usage
    
    # Estimate memory per sample
    model = model.to(device)
    
    # Start with batch size of 1
    batch_size = 1
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, input_size[1], input_size[2], device=device)
    
    # Warm up
    with torch.no_grad():
        model(dummy_input)
    
    # Record memory usage
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(dummy_input)
    
    memory_per_sample = torch.cuda.max_memory_allocated()
    
    # Calculate optimal batch size
    optimal_batch_size = int(target_memory / memory_per_sample)
    
    # Ensure batch size is at least 1
    return max(1, optimal_batch_size)

def quantize_for_deployment(model: nn.Module, 
                           target_device: str = 'cpu',
                           precision: str = 'int8',
                           example_input: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Convenience function to quantize a model for deployment with robust error handling.
    
    Args:
        model: PyTorch model
        target_device: Target device ('cpu', 'm1')
        precision: Quantization precision ('int8', 'fp16', 'dynamic')
        example_input: Example input for tracing (required for some optimizations)
        
    Returns:
        Deployment-ready quantized model
    """
    # Skip if quantization is not supported
    if not is_quantization_supported():
        print("Quantization not supported on this platform, returning original model")
        return model
    
    # Determine best quantization strategy based on target device
    if target_device == 'm1':
        # For M1, fp16 works well with Metal acceleration
        precision = 'fp16'
    
    try:
        # Create quantizer
        from urban_point_cloud_analyzer.optimization.quantization.model_quantization import ModelQuantizer
        quantizer = ModelQuantizer(precision=precision)
        
        # Basic quantization without calibration
        quantized_model = quantizer.quantize_model(model)
        
        # For CPU deployment, try converting to TorchScript if example input is provided
        if target_device == 'cpu' and example_input is not None:
            try:
                # Trace the model
                traced_model = torch.jit.trace(quantized_model, example_input)
                # Optimize for inference
                traced_model = torch.jit.optimize_for_inference(traced_model)
                return traced_model
            except Exception as e:
                print(f"Warning: Failed to convert to TorchScript: {e}")
                # Fall back to standard quantized model
                return quantized_model
        
        return quantized_model
    
    except Exception as e:
        print(f"Warning: Failed to quantize model: {e}")
        # Return original model as fallback
        return model