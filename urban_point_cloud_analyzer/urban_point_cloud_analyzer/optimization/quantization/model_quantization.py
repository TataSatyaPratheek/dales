# urban_point_cloud_analyzer/optimization/quantization/model_quantization.py
import torch
import torch.nn as nn
import torch.quantization as quantization
from typing import Dict, List, Optional, Tuple, Union
import copy
import time
import numpy as np

class ModelQuantizer:
    """
    Quantize models for efficient inference on CPU and resource-constrained devices.
    Specifically optimized for deploying on M1 MacBook Air with 8GB RAM.
    """
    
    def __init__(self, precision: str = 'int8', calibration_method: str = 'histogram'):
        """
        Initialize model quantizer.
        
        Args:
            precision: Quantization precision ('int8', 'fp16', or 'dynamic')
            calibration_method: Calibration method for static quantization ('minmax', 'histogram')
        """
        self.precision = precision
        self.calibration_method = calibration_method
        
        # Set up observers based on calibration method
        if calibration_method == 'histogram':
            self.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        else:  # minmax
            self.qconfig = torch.quantization.QConfig(
                activation=torch.quantization.MinMaxObserver.with_args(
                    dtype=torch.quint8
                ),
                weight=torch.quantization.MinMaxObserver.with_args(
                    dtype=torch.qint8
                )
            )
    
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """
        Prepare model for quantization.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model prepared for quantization
        """
        # Create a copy of the model to avoid modifying the original
        model_fp32 = copy.deepcopy(model)
        
        # Move model to CPU for quantization
        model_fp32 = model_fp32.cpu()
        
        # Set model to eval mode
        model_fp32.eval()
        
        if self.precision == 'int8':
            # Static int8 quantization
            # Fuse modules for better quantization
            model_fp32 = self._fuse_modules(model_fp32)
            
            # Set qconfig for the model
            model_fp32.qconfig = self.qconfig
            
            # Prepare model for calibration
            model_fp32 = torch.quantization.prepare(model_fp32)
            
        elif self.precision == 'dynamic':
            # Dynamic quantization (no preparation needed)
            pass
            
        elif self.precision == 'fp16':
            # FP16 half-precision (no preparation needed for standard conversion)
            pass
            
        else:
            raise ValueError(f"Unsupported precision: {self.precision}")
        
        return model_fp32
    
    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """
        Fuse modules for more efficient quantization.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model with fused modules
        """
        # Find fusable module patterns
        fusable_patterns = []
        
        # Find Conv-BN-ReLU patterns
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                # Check if the sequence contains Conv-BN-ReLU pattern
                for i in range(len(module) - 2):
                    if (isinstance(module[i], nn.Conv2d) and 
                        isinstance(module[i+1], nn.BatchNorm2d) and 
                        isinstance(module[i+2], nn.ReLU)):
                        fusable_patterns.append([f"{name}.{i}", f"{name}.{i+1}", f"{name}.{i+2}"])
        
        # Find Conv-BN pattern
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                for i in range(len(module) - 1):
                    if (isinstance(module[i], nn.Conv2d) and 
                        isinstance(module[i+1], nn.BatchNorm2d)):
                        # Check if this Conv-BN is already part of a Conv-BN-ReLU pattern
                        pattern = [f"{name}.{i}", f"{name}.{i+1}"]
                        extended_pattern = pattern + [f"{name}.{i+2}"]
                        if extended_pattern not in fusable_patterns:
                            fusable_patterns.append(pattern)
        
        # Fuse modules if patterns found
        if fusable_patterns:
            model = torch.quantization.fuse_modules(model, fusable_patterns)
        
        return model
    
    def calibrate_model(self, model: nn.Module, calibration_loader: torch.utils.data.DataLoader) -> nn.Module:
        """
        Calibrate model for static quantization.
        
        Args:
            model: Model prepared for quantization
            calibration_loader: DataLoader with calibration data
            
        Returns:
            Calibrated model
        """
        if self.precision != 'int8':
            # No calibration needed for dynamic or fp16
            return model
        
        # Run calibration
        with torch.no_grad():
            for batch in calibration_loader:
                # Adapt this to your data format
                if isinstance(batch, dict):
                    # Handle dictionary format (typical for point cloud data)
                    if 'points' in batch:
                        points = batch['points']
                        # Add features if available
                        features = batch.get('features', None)
                        
                        if features is not None:
                            # Call model with both points and features
                            model(points, features)
                        else:
                            # Call model with just points
                            model(points)
                elif isinstance(batch, tuple) and len(batch) >= 1:
                    # Handle tuple format (data, target)
                    data = batch[0]
                    model(data)
                else:
                    # Handle tensor format
                    model(batch)
        
        return model
    
    def convert_model(self, model: nn.Module) -> nn.Module:
        """
        Convert model to quantized version.
        
        Args:
            model: Prepared and calibrated model
            
        Returns:
            Quantized model
        """
        if self.precision == 'int8':
            # Static int8 quantization
            quantized_model = torch.quantization.convert(model)
            
        elif self.precision == 'dynamic':
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {nn.Linear, nn.LSTM, nn.LSTMCell, nn.RNN, nn.RNNCell, nn.GRUCell},
                dtype=torch.qint8
            )
            
        elif self.precision == 'fp16':
            # FP16 half-precision
            quantized_model = model.half()
            
        else:
            raise ValueError(f"Unsupported precision: {self.precision}")
        
        return quantized_model
    
    def quantize_model(self, model: nn.Module, calibration_loader: Optional[torch.utils.data.DataLoader] = None) -> nn.Module:
        """
        Full quantization pipeline: prepare, calibrate, convert.
        
        Args:
            model: PyTorch model
            calibration_loader: Optional DataLoader with calibration data
            
        Returns:
            Quantized model
        """
        # Prepare model
        prepared_model = self.prepare_model(model)
        
        # Calibrate model if needed
        if self.precision == 'int8' and calibration_loader is not None:
            calibrated_model = self.calibrate_model(prepared_model, calibration_loader)
        else:
            calibrated_model = prepared_model
        
        # Convert model
        quantized_model = self.convert_model(calibrated_model)
        
        return quantized_model
    
    def benchmark_model(self, 
                      fp32_model: nn.Module, 
                      quantized_model: nn.Module, 
                      input_data: Union[torch.Tensor, Tuple[torch.Tensor, ...]], 
                      num_runs: int = 50,
                      warmup_runs: int = 10) -> Dict:
        """
        Benchmark FP32 vs quantized model.
        
        Args:
            fp32_model: Original FP32 model
            quantized_model: Quantized model
            input_data: Input data for benchmarking (tensor or tuple of tensors)
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Dictionary with benchmark results
        """
        results = {}
        
        # Move models to CPU for fair comparison
        fp32_model = fp32_model.cpu().eval()
        quantized_model = quantized_model.cpu().eval()
        
        # Ensure input is on CPU
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu()
        elif isinstance(input_data, tuple):
            input_data = tuple(t.cpu() if isinstance(t, torch.Tensor) else t for t in input_data)
        
        # Warmup FP32 model
        with torch.no_grad():
            for _ in range(warmup_runs):
                if isinstance(input_data, tuple):
                    fp32_model(*input_data)
                else:
                    fp32_model(input_data)
        
        # Benchmark FP32 model
        fp32_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                if isinstance(input_data, tuple):
                    fp32_model(*input_data)
                else:
                    fp32_model(input_data)
                fp32_times.append(time.time() - start_time)
        
        # Calculate FP32 stats
        fp32_mean = np.mean(fp32_times) * 1000  # ms
        fp32_std = np.std(fp32_times) * 1000  # ms
        fp32_median = np.median(fp32_times) * 1000  # ms
        
        # Warmup quantized model
        with torch.no_grad():
            for _ in range(warmup_runs):
                if isinstance(input_data, tuple):
                    quantized_model(*input_data)
                else:
                    quantized_model(input_data)
        
        # Benchmark quantized model
        quantized_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                if isinstance(input_data, tuple):
                    quantized_model(*input_data)
                else:
                    quantized_model(input_data)
                quantized_times.append(time.time() - start_time)
        
        # Calculate quantized stats
        quantized_mean = np.mean(quantized_times) * 1000  # ms
        quantized_std = np.std(quantized_times) * 1000  # ms
        quantized_median = np.median(quantized_times) * 1000  # ms
        
        # Calculate speedup
        speedup = fp32_mean / quantized_mean
        
        # Check model sizes
        def get_model_size(model):
            torch.save(model.state_dict(), "temp_model.pt")
            size_mb = os.path.getsize("temp_model.pt") / (1024 * 1024)
            os.remove("temp_model.pt")
            return size_mb
        
        try:
            import os
            fp32_size = get_model_size(fp32_model)
            quantized_size = get_model_size(quantized_model)
            size_reduction = (1 - quantized_size / fp32_size) * 100
        except:
            fp32_size = 0
            quantized_size = 0
            size_reduction = 0
        
        # Compile results
        results = {
            'fp32_mean_ms': fp32_mean,
            'fp32_std_ms': fp32_std,
            'fp32_median_ms': fp32_median,
            'quantized_mean_ms': quantized_mean,
            'quantized_std_ms': quantized_std,
            'quantized_median_ms': quantized_median,
            'speedup': speedup,
            'fp32_size_mb': fp32_size,
            'quantized_size_mb': quantized_size,
            'size_reduction_percent': size_reduction,
            'precision': self.precision
        }
        
        return results


def quantize_for_deployment(model: nn.Module, 
                           target_device: str = 'cpu',
                           precision: str = 'int8',
                           example_input: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Convenience function to quantize a model for deployment.
    
    Args:
        model: PyTorch model
        target_device: Target device ('cpu', 'm1')
        precision: Quantization precision ('int8', 'fp16', 'dynamic')
        example_input: Example input for tracing (required for some optimizations)
        
    Returns:
        Deployment-ready quantized model
    """
    # Determine best quantization strategy based on target device
    if target_device == 'm1':
        # For M1, fp16 works well with Metal acceleration
        precision = 'fp16'
    
    # Create quantizer
    quantizer = ModelQuantizer(precision=precision)
    
    # Basic quantization without calibration
    quantized_model = quantizer.quantize_model(model)
    
    # For CPU deployment, convert to TorchScript if example input is provided
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


def create_calibration_loader(dataset: torch.utils.data.Dataset, 
                             num_samples: int = 100, 
                             batch_size: int = 10) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for calibration with a subset of the dataset.
    
    Args:
        dataset: Full dataset
        num_samples: Number of samples to use for calibration
        batch_size: Batch size for calibration
        
    Returns:
        DataLoader with calibration samples
    """
    # Create subset of dataset
    if len(dataset) > num_samples:
        # Randomly select indices
        indices = torch.randperm(len(dataset))[:num_samples]
        calibration_dataset = torch.utils.data.Subset(dataset, indices)
    else:
        calibration_dataset = dataset
    
    # Create DataLoader
    calibration_loader = torch.utils.data.DataLoader(
        calibration_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Use single process for predictable behavior
    )
    
    return calibration_loader