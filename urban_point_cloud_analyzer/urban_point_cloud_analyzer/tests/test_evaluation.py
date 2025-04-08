# urban_point_cloud_analyzer/tests/test_evaluation.py
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def test_segmentation_metrics():
    """Test segmentation evaluation metrics."""
    try:
        from urban_point_cloud_analyzer.evaluation.metrics import (
            calculate_iou, calculate_accuracy, calculate_precision_recall_f1, evaluate_segmentation
        )
        
        # Create dummy predictions and targets
        batch_size = 2
        num_points = 100
        num_classes = 8
        
        # Create perfect predictions for the first half of points
        # and wrong predictions for the second half
        pred = np.zeros(num_points, dtype=np.int32)
        target = np.zeros(num_points, dtype=np.int32)
        
        for i in range(num_classes):
            start = i * (num_points // num_classes)
            end = (i + 1) * (num_points // num_classes)
            
            # Set correct targets
            target[start:end] = i
            
            # Make half correct, half incorrect predictions
            mid = (start + end) // 2
            pred[start:mid] = i  # Correct
            pred[mid:end] = (i + 1) % num_classes  # Incorrect
        
        # Test IoU
        class_iou, mean_iou = calculate_iou(pred, target, num_classes)
        
        assert len(class_iou) == num_classes, f"Expected {num_classes} class IoUs, got {len(class_iou)}"
        assert 0 <= mean_iou <= 1, f"Mean IoU should be between 0 and 1, got {mean_iou}"
        
        # Test accuracy
        accuracy = calculate_accuracy(pred, target)
        assert 0 <= accuracy <= 1, f"Accuracy should be between 0 and 1, got {accuracy}"
        
        # Test precision, recall, F1
        precision, recall, f1 = calculate_precision_recall_f1(pred, target, num_classes)
        
        assert len(precision) == num_classes, f"Expected {num_classes} precision values, got {len(precision)}"
        assert len(recall) == num_classes, f"Expected {num_classes} recall values, got {len(recall)}"
        assert len(f1) == num_classes, f"Expected {num_classes} F1 values, got {len(f1)}"
        
        # Test full evaluation
        # Convert to torch tensors for the evaluation function
        pred_tensor = torch.tensor(pred).unsqueeze(0)  # Add batch dimension
        target_tensor = torch.tensor(target).unsqueeze(0)  # Add batch dimension
        
        metrics = evaluate_segmentation(pred_tensor, target_tensor, num_classes)
        
        assert 'accuracy' in metrics, "Missing accuracy in metrics"
        assert 'mean_iou' in metrics, "Missing mean_iou in metrics"
        assert 'class_iou' in metrics, "Missing class_iou in metrics"
        
        print(f"✓ Segmentation metrics test passed")
        return True
    except Exception as e:
        print(f"✗ Segmentation metrics test failed: {e}")
        return False

def run_evaluation_tests():
    """Run all evaluation metric tests."""
    tests = [
        test_segmentation_metrics,
        # Add more evaluation tests here
    ]
    
    results = []
    for test in tests:
        print(f"Running {test.__name__}...")
        result = test()
        results.append(result)
        print("")
    
    success_count = sum(1 for r in results if r)
    total_count = len(results)
    
    print(f"Evaluation tests completed: {success_count}/{total_count} successful")
    
    return all(results)

if __name__ == "__main__":
    success = run_evaluation_tests()
    sys.exit(0 if success else 1)