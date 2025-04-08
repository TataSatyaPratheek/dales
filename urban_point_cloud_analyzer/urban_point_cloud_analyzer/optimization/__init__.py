# urban_point_cloud_analyzer/optimization/__init__.py
from .hardware_optimizations import (
    detect_hardware,
    optimize_for_hardware,
    get_optimal_batch_size,
    optimize_dataloader_config,
    hardware_specific_pipeline,
    enable_sparse_operations
)

try:
    from .mixed_precision import (
        MixedPrecisionTrainer,
        apply_mixed_precision,
        benchmark_mixed_precision,
        optimize_model_for_1650ti
    )
except ImportError:
    # Define placeholder functions/classes
    class DummyMixedPrecisionTrainer:
        def __init__(self, *args, **kwargs):
            pass
        
        def train_step(self, *args, **kwargs):
            raise NotImplementedError("Mixed precision training not available")
        
        def validate_step(self, *args, **kwargs):
            raise NotImplementedError("Mixed precision validation not available")
    
    MixedPrecisionTrainer = DummyMixedPrecisionTrainer
    
    def apply_mixed_precision(model):
        return model
    
    def benchmark_mixed_precision(*args, **kwargs):
        return {}
    
    def optimize_model_for_1650ti(model):
        return model

try:
    from .m1_optimizations import (
        is_m1_mac,
        is_mps_available,
        get_optimal_device,
        M1Optimizer,
        benchmark_m1_optimizations
    )
except ImportError:
    # Define placeholder functions/classes
    def is_m1_mac():
        return False
    
    def is_mps_available():
        return False
    
    def get_optimal_device():
        return torch.device('cpu')
    
    class DummyM1Optimizer:
        @staticmethod
        def optimize_model(model, config=None):
            return model
        
        @staticmethod
        def optimize_memory_usage(model):
            return model
        
        @staticmethod
        def get_optimal_batch_size(*args, **kwargs):
            return 2
        
        @staticmethod
        def chunk_large_pointcloud(points, features=None, max_points_per_chunk=16384):
            return [(points, features)]
        
        @staticmethod
        def merge_chunk_results(results, original_shape=None):
            return results[0] if results else None
        
        @staticmethod
        def enable_mps_acceleration(fn, fallback_fn=None):
            return fn
    
    M1Optimizer = DummyM1Optimizer
    
    def benchmark_m1_optimizations(*args, **kwargs):
        return {}

try:
    from .sparse_ops import (
        create_sparse_tensor,
        SparseTensor,
        SparseConvolution,
        memory_usage_comparison
    )
except ImportError:
    # Define placeholder functions/classes
    def create_sparse_tensor(points, features=None):
        return points
    
    class DummySparseTensor:
        def __init__(self, indices, values=None):
            self.indices = indices
            self.values = values if values is not None else torch.ones(indices.shape[0])
    
    SparseTensor = DummySparseTensor
    
    class DummySparseConvolution(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.conv = nn.Conv1d(args[0], args[1], 1)
        
        def forward(self, points, features):
            return self.conv(features.transpose(1, 2)).transpose(1, 2)
    
    SparseConvolution = DummySparseConvolution
    
    def memory_usage_comparison(*args, **kwargs):
        return {'sparse_memory': 0, 'dense_memory': 0, 'memory_saving_percent': 0}

# Create a simplified optimization API
class OptimizationManager:
    """
    Main class for managing hardware-specific optimizations.
    """
    
    def __init__(self, config=None):
        """
        Initialize optimization manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.hardware_info = detect_hardware()
        self.device = get_optimal_device()
        
        # Log detected hardware
        print(f"Detected hardware: {self.hardware_info}")
        print(f"Using device: {self.device}")
    
    def optimize_model(self, model):
        """
        Apply all relevant optimizations to the model.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Optimized model
        """
        # First apply hardware-specific optimizations
        model = optimize_for_hardware(model, self.hardware_info)
        
        # Move to the appropriate device
        model = model.to(self.device)
        
        return model
    
    def get_trainer(self, model, optimizer):
        """
        Get appropriate trainer for the current hardware.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            
        Returns:
            Training helper appropriate for the hardware
        """
        if self.hardware_info.get('cuda_available', False):
            return MixedPrecisionTrainer(model, optimizer)
        elif self.hardware_info.get('is_m1_mac', False):
            # Return a trainer that uses MPS acceleration
            return self._create_m1_trainer(model, optimizer)
        else:
            # Return a basic trainer for CPU
            return self._create_basic_trainer(model, optimizer)
    
    def _create_m1_trainer(self, model, optimizer):
        """Create a trainer optimized for M1 Mac."""
        # Create a wrapper around MixedPrecisionTrainer with MPS optimizations
        class M1Trainer:
            def __init__(self, model, optimizer):
                self.model = M1Optimizer.optimize_model(model)
                self.optimizer = optimizer
            
            def train_step(self, data, target, loss_fn):
                # Move data to MPS if available
                if is_mps_available():
                    data = data.to('mps')
                    target = target.to('mps')
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                loss = loss_fn(output, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Calculate accuracy
                with torch.no_grad():
                    pred = output.argmax(dim=1)
                    correct = (pred == target).sum().item()
                    total = target.numel()
                    accuracy = correct / total
                
                return loss, accuracy
            
            def validate_step(self, data, target, loss_fn):
                # Move data to MPS if available
                if is_mps_available():
                    data = data.to('mps')
                    target = target.to('mps')
                
                # No gradients needed for validation
                with torch.no_grad():
                    output = self.model(data)
                    loss = loss_fn(output, target)
                    
                    # Calculate accuracy
                    pred = output.argmax(dim=1)
                    correct = (pred == target).sum().item()
                    total = target.numel()
                    accuracy = correct / total
                
                return loss, accuracy
        
        return M1Trainer(model, optimizer)
    
    def _create_basic_trainer(self, model, optimizer):
        """Create a basic trainer for CPU."""
        class BasicTrainer:
            def __init__(self, model, optimizer):
                self.model = model
                self.optimizer = optimizer
            
            def train_step(self, data, target, loss_fn):
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                loss = loss_fn(output, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Calculate accuracy
                with torch.no_grad():
                    pred = output.argmax(dim=1)
                    correct = (pred == target).sum().item()
                    total = target.numel()
                    accuracy = correct / total
                
                return loss, accuracy
            
            def validate_step(self, data, target, loss_fn):
                # No gradients needed for validation
                with torch.no_grad():
                    output = self.model(data)
                    loss = loss_fn(output, target)
                    
                    # Calculate accuracy
                    pred = output.argmax(dim=1)
                    correct = (pred == target).sum().item()
                    total = target.numel()
                    accuracy = correct / total
                
                return loss, accuracy
        
        return BasicTrainer(model, optimizer)
    
    def optimize_dataloader_config(self, dataloader_config):
        """
        Optimize DataLoader configuration for current hardware.
        
        Args:
            dataloader_config: DataLoader configuration dictionary
            
        Returns:
            Optimized DataLoader configuration
        """
        return optimize_dataloader_config(dataloader_config, self.hardware_info)
    
    def get_optimal_batch_size(self, model, input_shape):
        """
        Get optimal batch size for the current hardware.
        
        Args:
            model: PyTorch model
            input_shape: Input shape (without batch dimension)
            
        Returns:
            Optimal batch size
        """
        return get_optimal_batch_size(model, input_shape, self.hardware_info)
    
    @staticmethod
    def enable_hardware_optimizations(fn):
        """
        Decorator to apply hardware-specific optimizations to a function.
        
        Args:
            fn: Function to optimize
            
        Returns:
            Optimized function
        """
        return hardware_specific_pipeline(fn)
    
    @staticmethod
    def enable_sparse_ops(fn):
        """
        Decorator to enable sparse operations for a function.
        
        Args:
            fn: Function to optimize
            
        Returns:
            Optimized function
        """
        return enable_sparse_operations()(fn)
    
    def run_benchmark(self, model, input_shape):
        """
        Run benchmarks for the current hardware.
        
        Args:
            model: PyTorch model
            input_shape: Input shape (including batch dimension)
            
        Returns:
            Dictionary with benchmark results
        """
        results = {}
        
        # Get hardware-specific benchmarks
        if self.hardware_info.get('is_1650ti', False):
            # Benchmark mixed precision for NVIDIA GPU
            mixed_precision_results = benchmark_mixed_precision(model, input_shape)
            results.update(mixed_precision_results)
            
            # Benchmark sparse operations if available
            try:
                sparse_results = memory_usage_comparison(
                    input_shape[0], input_shape[2], 32  # Example values
                )
                results.update(sparse_results)
            except ImportError:
                pass
                
        elif self.hardware_info.get('is_m1_mac', False):
            # Benchmark M1-specific optimizations
            m1_results = benchmark_m1_optimizations(model, input_shape)
            results.update(m1_results)
        
        return results

# Create a global optimization manager
optimization_manager = None

def get_optimization_manager(config=None):
    """
    Get or create the global optimization manager.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        OptimizationManager instance
    """
    global optimization_manager
    if optimization_manager is None:
        optimization_manager = OptimizationManager(config)
    return optimization_manager

# Export public API
__all__ = [
    'detect_hardware',
    'optimize_for_hardware',
    'get_optimal_batch_size',
    'optimize_dataloader_config',
    'hardware_specific_pipeline',
    'enable_sparse_operations',
    'MixedPrecisionTrainer',
    'apply_mixed_precision',
    'benchmark_mixed_precision',
    'optimize_model_for_1650ti',
    'is_m1_mac',
    'is_mps_available',
    'get_optimal_device',
    'M1Optimizer',
    'benchmark_m1_optimizations',
    'create_sparse_tensor',
    'SparseTensor',
    'SparseConvolution',
    'memory_usage_comparison',
    'OptimizationManager',
    'get_optimization_manager'
]