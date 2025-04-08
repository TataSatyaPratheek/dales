# urban_point_cloud_analyzer/evaluation/metrics.py
import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional

def calculate_iou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> Tuple[np.ndarray, float]:
    """
    Calculate Intersection over Union (IoU) for each class and mean IoU.
    
    Args:
        pred: (N,) array of predicted labels
        target: (N,) array of target labels
        num_classes: Number of classes
        
    Returns:
        Tuple of (class_iou, mean_iou)
    """
    # Initialize IoU for each class
    class_iou = np.zeros(num_classes)
    
    # Calculate IoU for each class
    for i in range(num_classes):
        # True positives: prediction and target are class i
        tp = np.sum((pred == i) & (target == i))
        
        # False positives: prediction is class i but target is not
        fp = np.sum((pred == i) & (target != i))
        
        # False negatives: prediction is not class i but target is
        fn = np.sum((pred != i) & (target == i))
        
        # Calculate IoU
        if tp + fp + fn == 0:
            # Class not present in prediction or target
            class_iou[i] = np.nan
        else:
            class_iou[i] = tp / (tp + fp + fn)
    
    # Calculate mean IoU (exclude NaN values)
    valid_classes = ~np.isnan(class_iou)
    if np.sum(valid_classes) == 0:
        mean_iou = 0.0
    else:
        mean_iou = np.mean(class_iou[valid_classes])
    
    return class_iou, mean_iou

def calculate_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate overall accuracy.
    
    Args:
        pred: (N,) array of predicted labels
        target: (N,) array of target labels
        
    Returns:
        Accuracy as a float
    """
    return np.mean(pred == target)

def calculate_precision_recall_f1(pred: np.ndarray, target: np.ndarray, num_classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate precision, recall, and F1 score for each class.
    
    Args:
        pred: (N,) array of predicted labels
        target: (N,) array of target labels
        num_classes: Number of classes
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    
    for i in range(num_classes):
        # True positives: prediction and target are class i
        tp = np.sum((pred == i) & (target == i))
        
        # False positives: prediction is class i but target is not
        fp = np.sum((pred == i) & (target != i))
        
        # False negatives: prediction is not class i but target is
        fn = np.sum((pred != i) & (target == i))
        
        # Calculate precision
        if tp + fp == 0:
            precision[i] = 0.0
        else:
            precision[i] = tp / (tp + fp)
        
        # Calculate recall
        if tp + fn == 0:
            recall[i] = 0.0
        else:
            recall[i] = tp / (tp + fn)
        
        # Calculate F1 score
        if precision[i] + recall[i] == 0:
            f1[i] = 0.0
        else:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    
    return precision, recall, f1

def evaluate_segmentation(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict:
    """
    Evaluate semantic segmentation predictions.
    
    Args:
        pred: (B, C, N) or (B, N) tensor of predictions
        target: (B, N) tensor of target labels
        num_classes: Number of classes
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Convert to numpy for easier processing
    if pred.dim() == 3:  # (B, C, N)
        pred = torch.argmax(pred, dim=1)
    
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Flatten batch dimension
    pred_flat = pred_np.reshape(-1)
    target_flat = target_np.reshape(-1)
    
    # Calculate metrics
    class_iou, mean_iou = calculate_iou(pred_flat, target_flat, num_classes)
    accuracy = calculate_accuracy(pred_flat, target_flat)
    precision, recall, f1 = calculate_precision_recall_f1(pred_flat, target_flat, num_classes)
    
    # Combine metrics
    metrics = {
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'class_iou': {i: class_iou[i] for i in range(num_classes)},
        'precision': {i: precision[i] for i in range(num_classes)},
        'recall': {i: recall[i] for i in range(num_classes)},
        'f1': {i: f1[i] for i in range(num_classes)}
    }
    
    return metrics