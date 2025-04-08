# urban_point_cloud_analyzer/tests/test_mixed_precision.py
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def test_mixed_precision_trainer():
    """Test mixed precision trainer."""
    try:
        from urban_point_cloud_analyzer.optimization.mixed_precision import MixedPrecisionTrainer
        
        # Skip test if CUDA is not available
        if not torch.cuda.is_available():
            print("⚠ Skipping mixed precision test as CUDA is not available")
            return True
        
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
        ).cuda()
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create mixed precision trainer
        trainer = MixedPrecisionTrainer(model, optimizer)
        
        # Create dummy data
        data = torch.randn(8, 3, 32, 32).cuda()
        target = torch.randint(0, 10, (8,)).cuda()
        
        # Define loss function
        loss_fn = nn.CrossEntropyLoss()
        
        # Test train step
        loss, acc = trainer.train_step(data, target, loss_fn)
        
        # Check if loss and accuracy are valid
        assert loss > 0, "Loss should be positive"
        assert 0 <= acc <= 1, "Accuracy should be between 0 and 1"
        
        # Test validate step
        val_loss, val_acc = trainer.validate_step(data, target, loss_fn)
        
        # Check if validation loss and accuracy are valid
        assert val_loss > 0, "Validation loss should be positive"
        assert 0 <= val_acc <= 1, "Validation accuracy should be between 0 and 1"
        
        print(f"✓ Mixed precision trainer test passed")
        return True
    except Exception as e:
        print(f"✗ Mixed precision trainer test failed: {e}")
        return False

def test_apply_mixed_precision():
    """Test apply_mixed_precision function."""
    try:
        from urban_point_cloud_analyzer.optimization.mixed_precision import apply_mixed_precision
        
        # Skip test if CUDA is not available
        if not torch.cuda.is_available():
            print("⚠ Skipping apply_mixed_precision test as CUDA is not available")
            return True
        
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
        ).cuda()
        
        # Apply mixed precision
        model = apply_mixed_precision(model)
        
        # Test forward pass
        data = torch.randn(8, 3, 32, 32).cuda()
        
        # This should complete without errors
        output = model(data)
        
        # Check output shape
        assert output.shape == (8, 10), f"Expected shape (8, 10), got {output.shape}"
        
        print(f"✓ apply_mixed_precision test passed")
        return True
    except Exception as e:
        print(f"✗ apply_mixed_precision test failed: {e}")
        return False

def test_benchmark_mixed_precision():
    """Test benchmark_mixed_precision function."""
    try:
        from urban_point_cloud_analyzer.optimization.mixed_precision import benchmark_mixed_precision
        
        # Skip test if CUDA is not available
        if not torch.cuda.is_available():
            print("⚠ Skipping benchmark_mixed_precision test as CUDA is not available")
            return True
        
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
        ).cuda()
        
        # Benchmark
        input_shape = (8, 3, 32, 32)
        results = benchmark_mixed_precision(model, input_shape, num_runs=3)
        
        # Check if results are valid
        assert 'fp32_time' in results, "Missing fp32_time in results"
        assert 'fp16_time' in results, "Missing fp16_time in results"
        assert 'speedup' in results, "Missing speedup in results"
        
        # Print results
        print(f"FP32 time: {results['fp32_time']:.6f} s")
        print(f"FP16 time: {results['fp16_time']:.6f} s")
        print(f"Speedup: {results['speedup']:.2f}x")
        print(f"Memory saving: {results['memory_saving']:.2f}%")
        
        print(f"✓ benchmark_mixed_precision test passed")
        return True
    except Exception as e:
        print(f"✗ benchmark_mixed_precision test failed: {e}")
        return False

def test_optimize_model_for_1650ti():
    """Test optimize_model_for_1650ti function."""
    try:
        from urban_point_cloud_analyzer.optimization.mixed_precision import optimize_model_for_1650ti
        
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
        
        # Apply optimizations
        optimized_model = optimize_model_for_1650ti(model)
        
        # This should complete without errors
        assert isinstance(optimized_model, nn.Module), "Result should be a nn.Module"
        
        if torch.cuda.is_available():
            # Test forward pass
            data = torch.randn(8, 3, 32, 32).cuda()
            optimized_model = optimized_model.cuda()
            
            # This should complete without errors
            output = optimized_model(data)
            
            # Check output shape
            assert output.shape == (8, 10), f"Expected shape (8, 10), got {output.shape}"
        
        print(f"✓ optimize_model_for_1650ti test passed")
        return True
    except Exception as e:
        print(f"✗ optimize_model_for_1650ti test failed: {e}")
        return False

def run_mixed_precision_tests():
    """Run all mixed precision tests."""
    tests = [
        test_mixed_precision_trainer,
        test_apply_mixed_precision,
        test_benchmark_mixed_precision,
        test_optimize_model_for_1650ti,
    ]
    
    results = []
    for test in tests:
        print(f"Running {test.__name__}...")
        result = test()
        results.append(result)
        print("")
    
    success_count = sum(1 for r in results if r)
    total_count = len(results)
    
    print(f"Mixed precision tests completed: {success_count}/{total_count} successful")
    
    return all(results)

if __name__ == "__main__":
    success = run_mixed_precision_tests()
    sys.exit(0 if success else 1)