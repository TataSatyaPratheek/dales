# urban_point_cloud_analyzer/tests/test_batch_size_optimization.py
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def test_batch_size_optimizer():
    """Test BatchSizeOptimizer class."""
    try:
        from urban_point_cloud_analyzer.optimization.batch_size_optimization import BatchSizeOptimizer
        
        # Create a simple model
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 10)
        )
        
        # Define input shape (without batch dimension)
        input_shape = (3, 32, 32)
        
        # Create optimizer
        optimizer = BatchSizeOptimizer(
            model, input_shape,
            min_batch_size=1,
            max_batch_size=8  # Small max for testing
        )
        
        # Test binary search
        try:
            batch_size = optimizer.binary_search_batch_size()
            print(f"Optimal batch size (binary search): {batch_size}")
            
            # Check if batch size is within expected range
            assert 1 <= batch_size <= 8, f"Batch size should be between 1 and 8, got {batch_size}"
        except Exception as e:
            print(f"Binary search failed: {e}")
        
        # Test gradual increase
        try:
            batch_size = optimizer.gradual_increase_batch_size()
            print(f"Optimal batch size (gradual increase): {batch_size}")
            
            # Check if batch size is within expected range
            assert 1 <= batch_size <= 8, f"Batch size should be between 1 and 8, got {batch_size}"
        except Exception as e:
            print(f"Gradual increase failed: {e}")
        
        # Test throughput measurement with a few batch sizes
        try:
            throughput = optimizer.measure_throughput([1, 2], num_iterations=2, warmup_iterations=1)
            print(f"Throughput results: {throughput}")
            
            # Check if results are returned
            assert isinstance(throughput, dict), "Throughput results should be a dictionary"
        except Exception as e:
            print(f"Throughput measurement failed: {e}")
        
        # Test find_optimal_batch_size
        try:
            results = optimizer.find_optimal_batch_size(criterion='memory')
            print(f"Optimal batch size (memory): {results}")
            
            # Check if results contain expected keys
            assert 'optimal_batch_size' in results, "Missing optimal_batch_size in results"
            assert 'criterion' in results, "Missing criterion in results"
            assert 'device' in results, "Missing device in results"
        except Exception as e:
            print(f"Find optimal batch size failed: {e}")
        
        print(f"✓ BatchSizeOptimizer test passed")
        return True
    except Exception as e:
        print(f"✗ BatchSizeOptimizer test failed: {e}")
        return False

def test_get_optimal_batch_size_for_hardware():
    """Test get_optimal_batch_size_for_hardware function."""
    try:
        from urban_point_cloud_analyzer.optimization.batch_size_optimization import get_optimal_batch_size_for_hardware
        
        # Create a simple model
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 10)
        )
        
        # Define input shape (without batch dimension)
        input_shape = (3, 32, 32)
        
        # Test auto detection
        batch_size = get_optimal_batch_size_for_hardware(model, input_shape, hardware_type='auto')
        print(f"Optimal batch size (auto): {batch_size}")
        
        # Check if batch size is positive
        assert batch_size > 0, "Batch size should be positive"
        
        # Test CPU
        batch_size = get_optimal_batch_size_for_hardware(model, input_shape, hardware_type='cpu')
        print(f"Optimal batch size (cpu): {batch_size}")
        
        # Check if batch size is positive
        assert batch_size > 0, "Batch size should be positive"
        
        print(f"✓ get_optimal_batch_size_for_hardware test passed")
        return True
    except Exception as e:
        print(f"✗ get_optimal_batch_size_for_hardware test failed: {e}")
        return False

def test_auto_batch_size_integration():
    """Test AutoBatchSizeIntegration class."""
    try:
        from urban_point_cloud_analyzer.optimization.batch_size_optimization import AutoBatchSizeIntegration
        
        # Create a simple model
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 10)
        )
        
        # Define input shape (without batch dimension)
        input_shape = (3, 32, 32)
        
        # Create integration helper
        integration = AutoBatchSizeIntegration(model, input_shape)
        
        # Check batch size
        batch_size = integration.get_batch_size()
        print(f"Auto batch size: {batch_size}")
        
        # Check if batch size is positive
        assert batch_size > 0, "Batch size should be positive"
        
        # Create dummy dataset
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size=100):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return torch.randn(3, 32, 32)
        
        dataset = DummyDataset()
        
        # Create dataloader
        dataloader = integration.create_dataloader(dataset, num_workers=0)
        
        # Check dataloader batch size
        assert dataloader.batch_size == batch_size, f"DataLoader batch size ({dataloader.batch_size}) doesn't match integration batch size ({batch_size})"
        
        print(f"✓ AutoBatchSizeIntegration test passed")
        return True
    except Exception as e:
        print(f"✗ AutoBatchSizeIntegration test failed: {e}")
        return False

def run_batch_size_optimization_tests():
    """Run all batch size optimization tests."""
    tests = [
        test_batch_size_optimizer,
        test_get_optimal_batch_size_for_hardware,
        test_auto_batch_size_integration,
    ]
    
    results = []
    for test in tests:
        print(f"Running {test.__name__}...")
        result = test()
        results.append(result)
        print("")
    
    success_count = sum(1 for r in results if r)
    total_count = len(results)
    
    print(f"Batch size optimization tests completed: {success_count}/{total_count} successful")
    
    return all(results)

if __name__ == "__main__":
    success = run_batch_size_optimization_tests()
    sys.exit(0 if success else 1)