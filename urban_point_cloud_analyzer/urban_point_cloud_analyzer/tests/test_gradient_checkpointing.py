# urban_point_cloud_analyzer/tests/test_gradient_checkpointing.py
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import copy
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def test_enable_gradient_checkpointing():
    """Test enable_gradient_checkpointing function."""
    try:
        from urban_point_cloud_analyzer.optimization.gradient_checkpointing import enable_gradient_checkpointing
        
        # Create a model with Sequentials
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 10)
        )
        
        # Apply gradient checkpointing
        checkpointed_model = enable_gradient_checkpointing(model)
        
        # Check if model is still a nn.Module
        assert isinstance(checkpointed_model, nn.Module), "Result should be a nn.Module"
        
        # Test forward pass to ensure it still works
        input_tensor = torch.randn(1, 3, 32, 32)
        
        # Run forward pass
        with torch.no_grad():
            output = checkpointed_model(input_tensor)
        
        # Check output shape
        assert output.shape == (1, 10), f"Expected shape (1, 10), got {output.shape}"
        
        # Now test with gradients to see if checkpointing works
        if torch.cuda.is_available():
            # Move to GPU to better test memory benefits
            model = model.cuda()
            checkpointed_model = checkpointed_model.cuda()
            input_tensor = input_tensor.cuda()
            
            # Run with original model
            input_tensor.requires_grad = True
            output = model(input_tensor)
            loss = output.mean()
            loss.backward()
            
            # Run with checkpointed model
            input_tensor.grad = None
            output = checkpointed_model(input_tensor)
            loss = output.mean()
            loss.backward()
            
            # Verify gradients still flow
            assert input_tensor.grad is not None, "No gradients from checkpointed model"
        
        print(f"✓ enable_gradient_checkpointing test passed")
        return True
    except Exception as e:
        print(f"✗ enable_gradient_checkpointing test failed: {e}")
        return False

def test_create_checkpoint_wrapper():
    """Test create_checkpoint_wrapper function."""
    try:
        from urban_point_cloud_analyzer.optimization.gradient_checkpointing import create_checkpoint_wrapper
        
        # Create a wrapper for a Conv2d
        CheckpointedConv2d = create_checkpoint_wrapper(nn.Conv2d)
        
        # Create a normal Conv2d and a wrapped Conv2d
        conv = nn.Conv2d(3, 16, 3, padding=1)
        checkpointed_conv = CheckpointedConv2d(3, 16, 3, padding=1)
        
        # Test forward pass with both
        input_tensor = torch.randn(1, 3, 32, 32)
        
        with torch.no_grad():
            output_regular = conv(input_tensor)
            output_checkpointed = checkpointed_conv(input_tensor)
        
        # Check that outputs match
        assert output_regular.shape == output_checkpointed.shape, "Outputs should have same shape"
        
        # Test with gradients
        if torch.cuda.is_available():
            # Move to GPU
            conv = conv.cuda()
            checkpointed_conv = checkpointed_conv.cuda()
            input_tensor = input_tensor.cuda().requires_grad_(True)
            
            # Run with original module
            output_regular = conv(input_tensor)
            loss_regular = output_regular.mean()
            loss_regular.backward()
            grad_regular = input_tensor.grad.clone()
            
            # Reset gradients
            input_tensor.grad = None
            
            # Run with checkpointed module
            output_checkpointed = checkpointed_conv(input_tensor)
            loss_checkpointed = output_checkpointed.mean()
            loss_checkpointed.backward()
            grad_checkpointed = input_tensor.grad.clone()
            
            # Check that gradients are close
            assert torch.allclose(grad_regular, grad_checkpointed), "Gradients should match"
        
        print(f"✓ create_checkpoint_wrapper test passed")
        return True
    except Exception as e:
        print(f"✗ create_checkpoint_wrapper test failed: {e}")
        return False

def test_replace_modules_with_checkpointed():
    """Test replace_modules_with_checkpointed function."""
    try:
        from urban_point_cloud_analyzer.optimization.gradient_checkpointing import replace_modules_with_checkpointed
        
        # Create a model with different module types
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(16)
                self.relu1 = nn.ReLU()
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(32)
                self.relu2 = nn.ReLU()
                self.fc = nn.Linear(32 * 32 * 32, 10)
            
            def forward(self, x):
                x = self.relu1(self.bn1(self.conv1(x)))
                x = self.relu2(self.bn2(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        model = TestModel()
        
        # Replace Conv2d modules with checkpointed versions
        modified_model = replace_modules_with_checkpointed(model, [nn.Conv2d])
        
        # Check if Conv2d modules have been replaced
        assert not isinstance(modified_model.conv1, nn.Conv2d), "Conv2d should be replaced"
        assert isinstance(modified_model.bn1, nn.BatchNorm2d), "BatchNorm2d should not be replaced"
        
        # Test forward pass
        input_tensor = torch.randn(1, 3, 32, 32)
        
        with torch.no_grad():
            output = modified_model(input_tensor)
        
        # Check output shape
        assert output.shape == (1, 10), f"Expected shape (1, 10), got {output.shape}"
        
        print(f"✓ replace_modules_with_checkpointed test passed")
        return True
    except Exception as e:
        print(f"✗ replace_modules_with_checkpointed test failed: {e}")
        return False

def test_measure_memory_savings():
    """Test measure_memory_savings function."""
    try:
        from urban_point_cloud_analyzer.optimization.gradient_checkpointing import measure_memory_savings
        
        # Skip test if CUDA is not available
        if not torch.cuda.is_available():
            print("⚠ Skipping memory savings test as CUDA is not available")
            return True
        
        # Create a model with some depth for meaningful checkpointing
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
        # Measure memory savings with a small batch
        input_shape = (2, 3, 32, 32)
        results = measure_memory_savings(model, input_shape)
        
        # Check results
        assert 'no_checkpoint_memory_mb' in results, "Missing no_checkpoint_memory_mb in results"
        assert 'with_checkpoint_memory_mb' in results, "Missing with_checkpoint_memory_mb in results"
        assert 'memory_savings_mb' in results, "Missing memory_savings_mb in results"
        assert 'memory_savings_percent' in results, "Missing memory_savings_percent in results"
        
        # Print results
        print(f"Memory usage without checkpointing: {results['no_checkpoint_memory_mb']:.2f} MB")
        print(f"Memory usage with checkpointing: {results['with_checkpoint_memory_mb']:.2f} MB")
        print(f"Memory savings: {results['memory_savings_mb']:.2f} MB ({results['memory_savings_percent']:.2f}%)")
        
        print(f"✓ measure_memory_savings test passed")
        return True
    except Exception as e:
        print(f"✗ measure_memory_savings test failed: {e}")
        return False

def test_is_gradient_checkpointing_effective():
    """Test is_gradient_checkpointing_effective function."""
    try:
        from urban_point_cloud_analyzer.optimization.gradient_checkpointing import is_gradient_checkpointing_effective
        
        # Skip test if CUDA is not available
        if not torch.cuda.is_available():
            print("⚠ Skipping effectiveness test as CUDA is not available")
            return True
        
        # Create a model with some depth for meaningful checkpointing
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 10)
        )
        
        # Analyze effectiveness
        input_shape = (2, 3, 32, 32)
        results = is_gradient_checkpointing_effective(model, input_shape)
        
        # Check results
        assert 'memory_savings_percent' in results, "Missing memory_savings_percent in results"
        assert 'throughput_slowdown' in results, "Missing throughput_slowdown in results"
        assert 'is_effective' in results, "Missing is_effective in results"
        assert 'recommendation' in results, "Missing recommendation in results"
        
        # Print results
        print(f"Memory savings: {results['memory_savings_percent']:.2f}%")
        print(f"Throughput slowdown: {results['throughput_slowdown']:.2f}x")
        print(f"Is effective: {results['is_effective']}")
        print(f"Recommendation: {results['recommendation']}")
        
        print(f"✓ is_gradient_checkpointing_effective test passed")
        return True
    except Exception as e:
        print(f"✗ is_gradient_checkpointing_effective test failed: {e}")
        return False

def test_checkpointed_module():
    """Test CheckpointedModule class."""
    try:
        from urban_point_cloud_analyzer.optimization.gradient_checkpointing import CheckpointedModule
        
        # Create a custom module that inherits from CheckpointedModule
        class CustomModule(CheckpointedModule):
            def __init__(self):
                super(CustomModule, self).__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                return self._forward_with_checkpointing(self._forward_impl, x)
            
            def _forward_impl(self, x):
                return self.relu(self.conv(x))
        
        # Create an instance
        module = CustomModule()
        
        # Test without checkpointing
        input_tensor = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output_normal = module(input_tensor)
        
        # Enable checkpointing
        module.gradient_checkpointing_enable()
        
        # Test with checkpointing
        with torch.no_grad():
            output_checkpointed = module(input_tensor)
        
        # Check that outputs match
        assert torch.allclose(output_normal, output_checkpointed), "Outputs should match"
        
        # Test with gradients if CUDA is available
        if torch.cuda.is_available():
            # Move to GPU
            module = module.cuda()
            input_tensor = input_tensor.cuda().requires_grad_(True)
            
            # Run with checkpointing
            output = module(input_tensor)
            loss = output.mean()
            loss.backward()
            
            # Verify gradients flow
            assert input_tensor.grad is not None, "No gradients from checkpointed module"
        
        print(f"✓ CheckpointedModule test passed")
        return True
    except Exception as e:
        print(f"✗ CheckpointedModule test failed: {e}")
        return False

def run_gradient_checkpointing_tests():
    """Run all gradient checkpointing tests."""
    tests = [
        test_enable_gradient_checkpointing,
        test_create_checkpoint_wrapper,
        test_replace_modules_with_checkpointed,
        test_measure_memory_savings,
        test_is_gradient_checkpointing_effective,
        test_checkpointed_module,
    ]
    
    results = []
    for test in tests:
        print(f"Running {test.__name__}...")
        result = test()
        results.append(result)
        print("")
    
    success_count = sum(1 for r in results if r)
    total_count = len(results)
    
    print(f"Gradient checkpointing tests completed: {success_count}/{total_count} successful")
    
    return all(results)

if __name__ == "__main__":
    success = run_gradient_checkpointing_tests()
    sys.exit(0 if success else 1)