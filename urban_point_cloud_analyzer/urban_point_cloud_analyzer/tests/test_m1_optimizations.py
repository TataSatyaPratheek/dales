# urban_point_cloud_analyzer/tests/test_m1_optimizations.py
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def test_is_m1_mac():
    """Test is_m1_mac function."""
    try:
        from urban_point_cloud_analyzer.optimization.m1_optimizations import is_m1_mac
        
        # This should run without errors
        result = is_m1_mac()
        print(f"Running on M1 Mac: {result}")
        
        print(f"✓ is_m1_mac test passed")
        return True
    except Exception as e:
        print(f"✗ is_m1_mac test failed: {e}")
        return False

def test_is_mps_available():
    """Test is_mps_available function."""
    try:
        from urban_point_cloud_analyzer.optimization.m1_optimizations import is_mps_available
        
        # This should run without errors
        result = is_mps_available()
        print(f"MPS available: {result}")
        
        print(f"✓ is_mps_available test passed")
        return True
    except Exception as e:
        print(f"✗ is_mps_available test failed: {e}")
        return False

def test_get_optimal_device():
    """Test get_optimal_device function."""
    try:
        from urban_point_cloud_analyzer.optimization.m1_optimizations import get_optimal_device
        
        # This should run without errors
        device = get_optimal_device()
        print(f"Optimal device: {device}")
        
        print(f"✓ get_optimal_device test passed")
        return True
    except Exception as e:
        print(f"✗ get_optimal_device test failed: {e}")
        return False

def test_m1_optimizer():
    """Test M1Optimizer class."""
    try:
        from urban_point_cloud_analyzer.optimization.m1_optimizations import (
            M1Optimizer, is_m1_mac, is_mps_available, get_optimal_device
        )
        
        # Create a simple model for testing
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
        
        # Test optimize_model
        optimized_model = M1Optimizer.optimize_model(model)
        assert isinstance(optimized_model, nn.Module), "Result should be a nn.Module"
        
        # Test optimize_memory_usage
        memory_optimized_model = M1Optimizer.optimize_memory_usage(model)
        assert isinstance(memory_optimized_model, nn.Module), "Result should be a nn.Module"
        
        # Test get_optimal_batch_size
        input_shape = (3, 32, 32)
        batch_size = M1Optimizer.get_optimal_batch_size(model, input_shape)
        print(f"Optimal batch size: {batch_size}")
        
        # Test chunk_large_pointcloud with simple data
        points = torch.rand(10000, 3)
        features = torch.rand(10000, 16)
        
        chunks = M1Optimizer.chunk_large_pointcloud(points, features, max_points_per_chunk=4096)
        assert len(chunks) > 0, "Should create at least one chunk"
        
        # Test merge_chunk_results with dummy data
        chunk_results = [torch.rand(1000, 10) for _ in range(5)]
        merged = M1Optimizer.merge_chunk_results(chunk_results)
        assert merged.shape == (5000, 10), f"Expected shape (5000, 10), got {merged.shape}"
        
        print(f"✓ M1Optimizer test passed")
        return True
    except Exception as e:
        print(f"✗ M1Optimizer test failed: {e}")
        return False

def test_enable_mps_acceleration():
    """Test enable_mps_acceleration decorator."""
    try:
        from urban_point_cloud_analyzer.optimization.m1_optimizations import M1Optimizer
        
        # Define a simple function to accelerate
        def simple_function(tensor):
            return tensor * 2
        
        # Apply the decorator
        accelerated_function = M1Optimizer.enable_mps_acceleration(simple_function)
        
        # Test with a tensor
        input_tensor = torch.rand(10, 10)
        output_tensor = accelerated_function(input_tensor)
        
        # Check result
        expected_output = input_tensor * 2
        assert torch.allclose(output_tensor, expected_output), "Output should be input * 2"
        
        print(f"✓ enable_mps_acceleration test passed")
        return True
    except Exception as e:
        print(f"✗ enable_mps_acceleration test failed: {e}")
        return False

def test_benchmark_m1_optimizations():
    """Test benchmark_m1_optimizations function."""
    try:
        from urban_point_cloud_analyzer.optimization.m1_optimizations import (
            benchmark_m1_optimizations, is_m1_mac
        )
        
        # Create a simple model for benchmarking
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
        
        # Run benchmark with small model and few runs
        input_shape = (1, 3, 32, 32)
        results = benchmark_m1_optimizations(model, input_shape, num_runs=2)
        
        # Check if results are returned (will be empty if not on M1)
        assert isinstance(results, dict), "Results should be a dictionary"
        
        if is_m1_mac():
            # Print results if on M1 Mac
            print(f"CPU time: {results.get('cpu_time', 0):.6f} s")
            print(f"MPS time: {results.get('mps_time', 0):.6f} s")
            print(f"MPS FP16 time: {results.get('mps_fp16_time', 0):.6f} s")
            print(f"MPS speedup: {results.get('speedup_mps', 0):.2f}x")
            print(f"MPS FP16 speedup: {results.get('speedup_mps_fp16', 0):.2f}x")
        
        print(f"✓ benchmark_m1_optimizations test passed")
        return True
    except Exception as e:
        print(f"✗ benchmark_m1_optimizations test failed: {e}")
        return False

def run_m1_optimization_tests():
    """Run all M1 optimization tests."""
    tests = [
        test_is_m1_mac,
        test_is_mps_available,
        test_get_optimal_device,
        test_m1_optimizer,
        test_enable_mps_acceleration,
        test_benchmark_m1_optimizations,
    ]
    
    results = []
    for test in tests:
        print(f"Running {test.__name__}...")
        result = test()
        results.append(result)
        print("")
    
    success_count = sum(1 for r in results if r)
    total_count = len(results)
    
    print(f"M1 optimization tests completed: {success_count}/{total_count} successful")
    
    return all(results)

if __name__ == "__main__":
    success = run_m1_optimization_tests()
    sys.exit(0 if success else 1)