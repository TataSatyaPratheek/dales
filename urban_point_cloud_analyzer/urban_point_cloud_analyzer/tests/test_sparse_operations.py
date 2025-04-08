# urban_point_cloud_analyzer/tests/test_sparse_operations.py
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def test_sparse_tensor_creation():
    """Test sparse tensor creation from point cloud data."""
    try:
        from urban_point_cloud_analyzer.optimization.sparse_ops import create_sparse_tensor
        
        # Create a simple point cloud with features
        points = np.random.rand(1000, 3).astype(np.float32)
        features = np.random.rand(1000, 32).astype(np.float32)
        
        # Create sparse tensor
        sparse_tensor = create_sparse_tensor(points, features)
        
        # Check if it's the right type (either sparse or dense tensor is fine)
        assert isinstance(sparse_tensor, torch.Tensor), "Result should be a tensor"
        
        print(f"✓ Sparse tensor creation test passed")
        return True
    except Exception as e:
        print(f"✗ Sparse tensor creation test failed: {e}")
        return False

def test_sparse_convolution():
    """Test sparse convolution operation."""
    try:
        from urban_point_cloud_analyzer.optimization.sparse_ops import SparseConvolution
        
        # Skip test if CUDA is not available
        if not torch.cuda.is_available():
            print("⚠ Skipping sparse convolution test as CUDA is not available")
            return True
        
        # Create a simple test case
        in_channels = 32
        out_channels = 64
        kernel_size = 3
        
        # Create sparse convolution layer
        conv = SparseConvolution(in_channels, out_channels, kernel_size)
        conv = conv.cuda()
        
        # Create input
        points = torch.rand(1, 1000, 3, device='cuda')
        features = torch.rand(1, 1000, in_channels, device='cuda')
        
        # Forward pass
        output = conv(points, features)
        
        # Check output shape
        assert output.shape == (1, 1000, out_channels), f"Expected shape (1, 1000, {out_channels}), got {output.shape}"
        
        print(f"✓ Sparse convolution test passed")
        return True
    except Exception as e:
        print(f"✗ Sparse convolution test failed: {e}")
        return False

def test_sparse_ops_memory_efficiency():
    """Test memory efficiency of sparse operations."""
    try:
        from urban_point_cloud_analyzer.optimization.sparse_ops import (
            create_sparse_tensor, SparseConvolution, memory_usage_comparison
        )
        
        # Skip test if CUDA is not available
        if not torch.cuda.is_available():
            print("⚠ Skipping memory efficiency test as CUDA is not available")
            return True
        
        # Compare memory usage
        num_points = 100000  # Large number of points
        in_channels = 32
        out_channels = 64
        
        memory_stats = memory_usage_comparison(num_points, in_channels, out_channels)
        
        # Check if sparse operations use less memory
        assert memory_stats['sparse_memory'] < memory_stats['dense_memory'], \
            "Sparse operations should use less memory than dense operations"
        
        # Print memory savings
        memory_saving = (1 - memory_stats['sparse_memory'] / memory_stats['dense_memory']) * 100
        print(f"Sparse operations use {memory_saving:.2f}% less memory than dense operations")
        
        print(f"✓ Memory efficiency test passed")
        return True
    except Exception as e:
        print(f"✗ Memory efficiency test failed: {e}")
        return False

def run_sparse_ops_tests():
    """Run all sparse operations tests."""
    tests = [
        test_sparse_tensor_creation,
        test_sparse_convolution,
        test_sparse_ops_memory_efficiency,
    ]
    
    results = []
    for test in tests:
        print(f"Running {test.__name__}...")
        result = test()
        results.append(result)
        print("")
    
    success_count = sum(1 for r in results if r)
    total_count = len(results)
    
    print(f"Sparse operations tests completed: {success_count}/{total_count} successful")
    
    return all(results)

if __name__ == "__main__":
    success = run_sparse_ops_tests()
    sys.exit(0 if success else 1)