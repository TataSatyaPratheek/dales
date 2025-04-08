# urban_point_cloud_analyzer/tests/test_training.py
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def test_loss_functions():
    """Test segmentation loss functions."""
    try:
        from urban_point_cloud_analyzer.training.loss_functions import get_loss_function, SegmentationLoss, FocalLoss
        
        # Create dummy predictions and targets
        batch_size = 2
        num_points = 64
        num_classes = 8
        
        predictions = torch.rand(batch_size, num_classes, num_points)
        targets = torch.randint(0, num_classes, (batch_size, num_points))
        
        # Test segmentation loss
        seg_loss_config = {'segmentation': 'cross_entropy'}
        seg_loss = get_loss_function(seg_loss_config)
        
        loss_value = seg_loss(predictions, targets)
        assert isinstance(loss_value, torch.Tensor), "Loss should be a tensor"
        assert loss_value.dim() == 0, "Loss should be a scalar"
        
        # Test focal loss
        focal_loss_config = {
            'segmentation': 'focal',
            'focal_gamma': 2.0
        }
        focal_loss = get_loss_function(focal_loss_config)
        
        loss_value = focal_loss(predictions, targets)
        assert isinstance(loss_value, torch.Tensor), "Loss should be a tensor"
        assert loss_value.dim() == 0, "Loss should be a scalar"
        
        print(f"✓ Loss functions test passed")
        return True
    except Exception as e:
        print(f"✗ Loss functions test failed: {e}")
        return False

def test_optimizers():
    """Test optimizer creation."""
    try:
        from urban_point_cloud_analyzer.training.optimizers import get_optimizer
        
        # Create dummy model parameters
        model = torch.nn.Linear(10, 10)
        
        # Test Adam optimizer
        adam_config = {'name': 'adam', 'lr': 0.001, 'weight_decay': 0.0001}
        adam = get_optimizer(model.parameters(), adam_config)
        
        assert isinstance(adam, torch.optim.Adam), "Should be Adam optimizer"
        
        # Test SGD optimizer
        sgd_config = {'name': 'sgd', 'lr': 0.01, 'momentum': 0.9}
        sgd = get_optimizer(model.parameters(), sgd_config)
        
        assert isinstance(sgd, torch.optim.SGD), "Should be SGD optimizer"
        
        print(f"✓ Optimizers test passed")
        return True
    except Exception as e:
        print(f"✗ Optimizers test failed: {e}")
        return False

def test_schedulers():
    """Test scheduler creation."""
    try:
        from urban_point_cloud_analyzer.training.schedulers import get_scheduler
        
        # Create dummy optimizer
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Test step scheduler
        step_config = {'name': 'step', 'step_size': 10, 'gamma': 0.1}
        step_scheduler = get_scheduler(optimizer, step_config)
        
        assert isinstance(step_scheduler, torch.optim.lr_scheduler.StepLR), "Should be StepLR scheduler"
        
        # Test cosine scheduler
        cosine_config = {'name': 'cosine', 'epochs': 100, 'min_lr': 0.00001}
        cosine_scheduler = get_scheduler(optimizer, cosine_config)
        
        assert isinstance(cosine_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR), "Should be CosineAnnealingLR scheduler"
        
        print(f"✓ Schedulers test passed")
        return True
    except Exception as e:
        print(f"✗ Schedulers test failed: {e}")
        return False

def run_training_tests():
    """Run all training component tests."""
    tests = [
        test_loss_functions,
        test_optimizers,
        test_schedulers,
    ]
    
    results = []
    for test in tests:
        print(f"Running {test.__name__}...")
        result = test()
        results.append(result)
        print("")
    
    success_count = sum(1 for r in results if r)
    total_count = len(results)
    
    print(f"Training tests completed: {success_count}/{total_count} successful")
    
    return all(results)

if __name__ == "__main__":
    success = run_training_tests()
    sys.exit(0 if success else 1)