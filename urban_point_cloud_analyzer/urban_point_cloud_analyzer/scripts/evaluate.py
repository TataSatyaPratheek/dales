# urban_point_cloud_analyzer/scripts/evaluate.py
#!/usr/bin/env python3
"""
Evaluation script for Urban Point Cloud Analyzer
"""
import argparse
import os
import sys
import yaml
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from urban_point_cloud_analyzer.data.loaders import DALESDataset
from urban_point_cloud_analyzer.data.preprocessing import PointCloudPreprocessor
from urban_point_cloud_analyzer.models import get_model
from urban_point_cloud_analyzer.evaluation.metrics import evaluate_segmentation
from urban_point_cloud_analyzer.utils.logger import setup_logger
from urban_point_cloud_analyzer.utils.hardware_utils import get_device_info, get_optimal_config, optimize_model_for_device, optimize_dataloader_for_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Urban Point Cloud Analyzer")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to test data directory")
    parser.add_argument("--output_dir", type=str, default="evaluation", help="Output directory")
    parser.add_argument("--batch_size", type=int, help="Batch size for evaluation")
    parser.add_argument("--disable_cuda", action="store_true", help="Disable CUDA even if available")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup logger
    logger = setup_logger(output_dir / "evaluation.log")
    logger.info(f"Config: {config}")
    
    # Get hardware info and optimal config
    device_info = get_device_info()
    logger.info(f"Device info: {device_info}")
    
    # Override CUDA availability if requested
    if args.disable_cuda:
        device_info['cuda_available'] = False
    
    # Get optimal config for hardware
    optimal_config = get_optimal_config(device_info)
    logger.info(f"Optimal config: {optimal_config}")
    
    # Setup device
    device = torch.device("cuda" if device_info['cuda_available'] else 
                         "mps" if device_info.get('mps_available', False) else 
                         "cpu")
    logger.info(f"Using device: {device}")
    
    # Get batch size
    batch_size = args.batch_size if args.batch_size else optimal_config['val_batch_size']
    logger.info(f"Using batch size: {batch_size}")
    
    # Create test dataset
    logger.info("Creating test dataset...")
    
    # Setup preprocessor
    preprocessor = PointCloudPreprocessor(config['data'])
    
    # Create test dataset
    test_dataset = DALESDataset(
        root_dir=args.data_dir,
        split='test',
        config=config['data'],
        transforms=None,  # No augmentation for testing
        preprocessor=preprocessor,
        cache=True,
        num_points=config['model']['backbone'].get('num_points', 16384),
        memory_efficient=device_info.get('is_m1', False)  # Use memory-efficient mode for M1
    )
    
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Optimize dataloader configuration
    test_loader_config = {
        'batch_size': batch_size,
        'shuffle': False,
        'drop_last': False,
        'collate_fn': DALESDataset.collate_fn,
        'split': 'test'
    }
    
    test_loader_config = optimize_dataloader_for_device(test_loader_config, device_info)
    
    # Create dataloader
    test_loader = DataLoader(test_dataset, **test_loader_config)
    
    # Create model
    logger.info("Creating model...")
    model = get_model(config['model'])
    
    # Load model weights
    logger.info(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Optimize model for device
    model = optimize_model_for_device(model, device_info)
    model.eval()
    
    # Initialize metrics
    all_metrics = []
    
    # Evaluate
    logger.info("Starting evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Get data
            points = batch['points'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(points)
            
            # Calculate metrics
            metrics = evaluate_segmentation(
                outputs, labels, 
                num_classes=config['model']['segmentation']['num_classes']
            )
            
            all_metrics.append(metrics)
    
    # Calculate overall metrics
    overall_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
        'mean_iou': np.mean([m['mean_iou'] for m in all_metrics]),
        'class_iou': {},
        'precision': {},
        'recall': {},
        'f1': {}
    }
    
    num_classes = config['model']['segmentation']['num_classes']
    
    for i in range(num_classes):
        overall_metrics['class_iou'][i] = np.mean([m['class_iou'][i] for m in all_metrics if i in m['class_iou']])
        overall_metrics['precision'][i] = np.mean([m['precision'][i] for m in all_metrics if i in m['precision']])
        overall_metrics['recall'][i] = np.mean([m['recall'][i] for m in all_metrics if i in m['recall']])
        overall_metrics['f1'][i] = np.mean([m['f1'][i] for m in all_metrics if i in m['f1']])
    
    # Log results
    logger.info("Evaluation Results:")
    logger.info(f"  Accuracy: {overall_metrics['accuracy']:.4f}")
    logger.info(f"  Mean IoU: {overall_metrics['mean_iou']:.4f}")
    
    # Log class-wise metrics
    class_names = {
        0: 'Ground',
        1: 'Vegetation',
        2: 'Buildings',
        3: 'Water',
        4: 'Car',
        5: 'Truck',
        6: 'Powerline',
        7: 'Fence'
    }
    
    logger.info("Class-wise Metrics:")
    for i in range(num_classes):
        logger.info(f"  Class {i} ({class_names.get(i, 'Unknown')}):")
        logger.info(f"    IoU: {overall_metrics['class_iou'][i]:.4f}")
        logger.info(f"    Precision: {overall_metrics['precision'][i]:.4f}")
        logger.info(f"    Recall: {overall_metrics['recall'][i]:.4f}")
        logger.info(f"    F1: {overall_metrics['f1'][i]:.4f}")
    
    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()