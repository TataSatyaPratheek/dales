# urban_point_cloud_analyzer/optimization/batch_size_optimization.py
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Callable
import time
import numpy as np
import gc

class BatchSizeOptimizer:
    """
    Optimizer for finding the optimal batch size for different hardware.
    Specifically tuned for 1650Ti (4GB VRAM) and M1 MacBook Air (8GB RAM).
    """
    
    def __init__(self, 
                model: nn.Module, 
                input_shape: Tuple[int, ...],
                device: Optional[torch.device] = None,
                max_memory_usage: float = 0.8,  # Use 80% of available memory
                min_batch_size: int = 1,
                max_batch_size: int = 128):
        """
        Initialize batch size optimizer.
        
        Args:
            model: PyTorch model
            input_shape: Input shape (without batch dimension)
            device: Device to use (defaults to cuda if available, then mps, then cpu)
            max_memory_usage: Maximum fraction of available memory to use
            min_batch_size: Minimum batch size to consider
            max_batch_size: Maximum batch size to consider
        """
        self.model = model
        self.input_shape = input_shape
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available') and torch.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        self.max_memory_usage = max_memory_usage
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        
        # Move model to device
        self.model = self.model.to(self.device)
    
    def _get_available_memory(self) -> float:
        """
        Get available memory in bytes.
        
        Returns:
            Available memory in bytes
        """
        if self.device.type == 'cuda':
            # For CUDA, get device properties
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            allocated_memory = torch.cuda.memory_allocated(self.device)
            available_memory = total_memory - allocated_memory
            
            return available_memory
        elif self.device.type == 'mps':
            # For MPS (Metal), there's no direct API to get memory
            # Use a conservative estimate of 6GB for M1 with 8GB RAM
            # This accounts for memory shared with the system
            return 6 * 1024 * 1024 * 1024
        else:
            # For CPU, use a conservative estimate of 2GB
            return 2 * 1024 * 1024 * 1024
    
    def binary_search_batch_size(self) -> int:
        """
        Use binary search to find the largest batch size that fits in memory.
        
        Returns:
            Optimal batch size
        """
        # Clear memory before starting
        self._clear_memory()
        
        # Set model to eval mode
        self.model.eval()
        
        # Available memory adjusted by max_memory_usage
        available_memory = self._get_available_memory() * self.max_memory_usage
        
        # Binary search
        low = self.min_batch_size
        high = self.max_batch_size
        optimal_batch_size = low
        
        while low <= high:
            mid = (low + high) // 2
            
            try:
                # Try this batch size
                test_input = torch.randn(mid, *self.input_shape, device=self.device)
                
                if self.device.type == 'cuda':
                    # Record memory before forward pass
                    torch.cuda.reset_peak_memory_stats(self.device)
                
                # Run forward pass
                with torch.no_grad():
                    self.model(test_input)
                
                if self.device.type == 'cuda':
                    # Check memory usage
                    memory_used = torch.cuda.max_memory_allocated(self.device)
                    
                    if memory_used < available_memory:
                        # This batch size works, try larger
                        optimal_batch_size = mid
                        low = mid + 1
                    else:
                        # This batch size uses too much memory, try smaller
                        high = mid - 1
                else:
                    # If not CUDA, we can't measure memory directly
                    # So if it completes without error, assume it's fine
                    optimal_batch_size = mid
                    low = mid + 1
                
                # Clear memory after test
                del test_input
                self._clear_memory()
                
            except RuntimeError as e:
                # Likely out of memory error
                high = mid - 1
                
                # Clear memory after error
                self._clear_memory()
            
            except Exception as e:
                print(f"Unexpected error: {e}")
                break
        
        return optimal_batch_size
    
    def _clear_memory(self):
        """Clear GPU/MPS memory."""
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def gradual_increase_batch_size(self) -> int:
        """
        Gradually increase batch size until memory error occurs.
        More reliable but slower than binary search.
        
        Returns:
            Optimal batch size
        """
        # Clear memory before starting
        self._clear_memory()
        
        # Set model to eval mode
        self.model.eval()
        
        # Start with minimum batch size
        batch_size = self.min_batch_size
        optimal_batch_size = batch_size
        
        while batch_size <= self.max_batch_size:
            try:
                # Try this batch size
                test_input = torch.randn(batch_size, *self.input_shape, device=self.device)
                
                # Run forward pass
                with torch.no_grad():
                    self.model(test_input)
                
                # This batch size works
                optimal_batch_size = batch_size
                
                # Clear memory after test
                del test_input
                self._clear_memory()
                
                # Try larger batch size
                batch_size += max(1, batch_size // 4)  # Increase by 25% each time
                
            except RuntimeError as e:
                # Likely out of memory error
                break
            
            except Exception as e:
                print(f"Unexpected error: {e}")
                break
        
        return optimal_batch_size
    
    def measure_throughput(self, batch_sizes: List[int], 
                         num_iterations: int = 10, 
                         warmup_iterations: int = 3) -> Dict[int, float]:
        """
        Measure throughput (samples/second) for different batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            num_iterations: Number of iterations for each batch size
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary mapping batch size to throughput
        """
        # Clear memory before starting
        self._clear_memory()
        
        # Set model to eval mode
        self.model.eval()
        
        # Store results
        throughput_results = {}
        
        for batch_size in batch_sizes:
            try:
                # Create test input
                test_input = torch.randn(batch_size, *self.input_shape, device=self.device)
                
                # Warmup
                for _ in range(warmup_iterations):
                    with torch.no_grad():
                        self.model(test_input)
                
                # Measure time
                start_time = time.time()
                
                for _ in range(num_iterations):
                    with torch.no_grad():
                        self.model(test_input)
                
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                
                # Calculate throughput
                total_samples = batch_size * num_iterations
                throughput = total_samples / elapsed_time
                
                throughput_results[batch_size] = throughput
                
                # Clear memory after test
                del test_input
                self._clear_memory()
                
            except RuntimeError:
                # Skip this batch size
                pass
            
            except Exception as e:
                print(f"Unexpected error for batch size {batch_size}: {e}")
        
        return throughput_results
    
    def find_optimal_throughput_batch_size(self) -> int:
        """
        Find batch size with highest throughput.
        
        Returns:
            Batch size with highest throughput
        """
        # First find the maximum batch size that fits in memory
        max_memory_batch_size = self.gradual_increase_batch_size()
        
        # If max batch size is small, just return it
        if max_memory_batch_size <= 8:
            return max_memory_batch_size
        
        # Generate batch sizes to test
        batch_sizes = []
        current = self.min_batch_size
        while current <= max_memory_batch_size:
            batch_sizes.append(current)
            current *= 2
        
        # Add max memory batch size if not already included
        if max_memory_batch_size not in batch_sizes:
            batch_sizes.append(max_memory_batch_size)
        
        # Sort batch sizes
        batch_sizes.sort()
        
        # Measure throughput
        throughput_results = self.measure_throughput(batch_sizes)
        
        if not throughput_results:
            # If no results, return the minimum batch size
            return self.min_batch_size
        
        # Find batch size with highest throughput
        optimal_batch_size = max(throughput_results.items(), key=lambda x: x[1])[0]
        
        return optimal_batch_size
    
    def find_optimal_batch_size(self, criterion: str = 'memory') -> Dict:
        """
        Find optimal batch size based on specified criterion.
        
        Args:
            criterion: Criterion for optimization ('memory', 'throughput', or 'both')
            
        Returns:
            Dictionary with optimal batch size and related information
        """
        if criterion == 'memory':
            # Find largest batch size that fits in memory
            optimal_batch_size = self.binary_search_batch_size()
            
            return {
                'optimal_batch_size': optimal_batch_size,
                'criterion': 'memory',
                'device': str(self.device)
            }
            
        elif criterion == 'throughput':
            # Find batch size with highest throughput
            optimal_batch_size = self.find_optimal_throughput_batch_size()
            
            return {
                'optimal_batch_size': optimal_batch_size,
                'criterion': 'throughput',
                'device': str(self.device)
            }
            
        elif criterion == 'both':
            # Find largest batch size that fits in memory
            memory_batch_size = self.binary_search_batch_size()
            
            # Find batch size with highest throughput (up to memory_batch_size)
            self.max_batch_size = memory_batch_size
            throughput_batch_size = self.find_optimal_throughput_batch_size()
            
            return {
                'memory_batch_size': memory_batch_size,
                'throughput_batch_size': throughput_batch_size,
                'optimal_batch_size': throughput_batch_size,  # Prefer throughput
                'criterion': 'both',
                'device': str(self.device)
            }
            
        else:
            raise ValueError(f"Unknown criterion: {criterion}")


def get_optimal_batch_size_for_hardware(model: nn.Module, 
                                       input_shape: Tuple[int, ...],
                                       hardware_type: str = 'auto') -> int:
    """
    Get an optimal batch size for a specific hardware type.
    
    Args:
        model: PyTorch model
        input_shape: Input shape (without batch dimension)
        hardware_type: Hardware type ('1650ti', 'm1', 'auto')
        
    Returns:
        Optimal batch size
    """
    # Detect hardware if auto
    if hardware_type == 'auto':
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if '1650 Ti' in device_name:
                hardware_type = '1650ti'
            else:
                hardware_type = 'cuda'
        elif hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available') and torch.mps.is_available():
            hardware_type = 'm1'
        else:
            hardware_type = 'cpu'
    
    # Set device based on hardware type
    if hardware_type == '1650ti' or hardware_type == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif hardware_type == 'm1':
        device = torch.device('mps' if hasattr(torch, 'mps') and torch.mps.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    # Customize optimizer settings based on hardware
    if hardware_type == '1650ti':
        optimizer = BatchSizeOptimizer(
            model, input_shape, device, 
            max_memory_usage=0.7,  # Use 70% of memory for 1650Ti (4GB VRAM)
            max_batch_size=32
        )
        
        # Find optimal batch size prioritizing throughput
        results = optimizer.find_optimal_batch_size(criterion='both')
        return results['optimal_batch_size']
        
    elif hardware_type == 'm1':
        optimizer = BatchSizeOptimizer(
            model, input_shape, device, 
            max_memory_usage=0.6,  # Use 60% of memory for M1 (unified memory)
            max_batch_size=16
        )
        
        # For M1, prioritize memory efficiency
        results = optimizer.find_optimal_batch_size(criterion='memory')
        return results['optimal_batch_size']
        
    elif hardware_type == 'cuda':
        optimizer = BatchSizeOptimizer(
            model, input_shape, device, 
            max_memory_usage=0.8,  # Use 80% of memory for general CUDA
            max_batch_size=64
        )
        
        # For CUDA, prioritize throughput
        results = optimizer.find_optimal_batch_size(criterion='throughput')
        return results['optimal_batch_size']
        
    else:  # CPU
        # For CPU, use a small static batch size
        return 4


class AutoBatchSizeIntegration:
    """
    Integration helper for automatic batch size selection in training loops.
    """
    
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...]):
        """
        Initialize auto batch size integration.
        
        Args:
            model: PyTorch model
            input_shape: Input shape (without batch dimension)
        """
        self.model = model
        self.input_shape = input_shape
        
        # Detect hardware and get optimal batch size
        self.batch_size = get_optimal_batch_size_for_hardware(model, input_shape)
    
    def create_dataloader(self, 
                         dataset: torch.utils.data.Dataset, 
                         is_training: bool = True,
                         **kwargs) -> torch.utils.data.DataLoader:
        """
        Create a DataLoader with optimal batch size.
        
        Args:
            dataset: PyTorch dataset
            is_training: Whether this is a training dataloader
            **kwargs: Additional arguments for DataLoader
            
        Returns:
            DataLoader with optimal batch size
        """
        # Set batch size
        kwargs['batch_size'] = self.batch_size
        
        # Set shuffle for training
        if 'shuffle' not in kwargs:
            kwargs['shuffle'] = is_training
        
        # Set pin_memory for CUDA
        if torch.cuda.is_available() and 'pin_memory' not in kwargs:
            kwargs['pin_memory'] = True
        
        # Create DataLoader
        return torch.utils.data.DataLoader(dataset, **kwargs)
    
    def get_batch_size(self) -> int:
        """Get the optimal batch size."""
        return self.batch_size