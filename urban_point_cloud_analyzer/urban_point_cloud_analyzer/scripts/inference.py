# urban_point_cloud_analyzer/scripts/inference.py
#!/usr/bin/env python3
"""
Inference script for Urban Point Cloud Analyzer
"""
import argparse
import os
import sys
import yaml
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import laspy

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from urban_point_cloud_analyzer.data.loaders import DALESDataset
from urban_point_cloud_analyzer.data.preprocessing import PointCloudPreprocessor
from urban_point_cloud_analyzer.models import get_model
from urban_point_cloud_analyzer.utils.logger import setup_logger
from urban_point_cloud_analyzer.utils.hardware_utils import get_device_info, get_optimal_config, optimize_model_for_device
from urban_point_cloud_analyzer.optimization.quantization import optimize_for_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for Urban Point Cloud Analyzer")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to point cloud file or directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--batch_size", type=int, help="Batch size for inference")
    parser.add_argument("--disable_cuda", action="store_true", help="Disable CUDA even if available")
    parser.add_argument("--use_sliding_window", action="store_true", help="Use sliding window inference for large point clouds")
    parser.add_argument("--window_size", type=int, default=20, help="Window size in meters for sliding window inference")
    parser.add_argument("--window_stride", type=int, default=10, help="Window stride in meters for sliding window inference")
    return parser.parse_args()


def load_point_cloud(file_path):
    """Load point cloud from LAS/LAZ file."""
    # Read LAS/LAZ file
    las = laspy.read(file_path)
    
    # Extract points
    points = np.vstack([
        las.x, las.y, las.z
    ]).T
    
    # Extract intensity if available
    if hasattr(las, 'intensity'):
        intensity = np.array(las.intensity).reshape(-1, 1)
        points = np.hstack([points, intensity])
    
    return points, las


def sliding_window_inference(model, points, window_size, window_stride, batch_size, device, num_classes=8):
    """
    Run inference using sliding window approach for large point clouds.
    
    Args:
        model: Trained model
        points: (N, 3+) array of points
        window_size: Window size in meters
        window_stride: Window stride in meters
        batch_size: Batch size for inference
        device: Device to run inference on
        num_classes: Number of classes
        
    Returns:
        Segmentation predictions for all points
    """
    # Calculate bounds
    min_bound = np.min(points[:, :3], axis=0)
    max_bound = np.max(points[:, :3], axis=0)
    
    # Initialize predictions array
    all_preds = np.zeros(len(points), dtype=np.int64)
    all_counts = np.zeros(len(points), dtype=np.int32)
    
    # Create KD-tree for efficient point lookup
    from scipy.spatial import cKDTree
    kdtree = cKDTree(points[:, :3])
    
    # Generate windows
    x_windows = np.arange(min_bound[0], max_bound[0] + window_stride, window_stride)
    y_windows = np.arange(min_bound[1], max_bound[1] + window_stride, window_stride)
    
    # Process each window
    total_windows = len(x_windows) * len(y_windows)
    with tqdm(total=total_windows, desc="Processing windows") as pbar:
        for x_min in x_windows:
            for y_min in y_windows:
                # Define window bounds
                x_max = x_min + window_size
                y_max = y_min + window_size
                
                # Query points in window
                window_min = np.array([x_min, y_min, min_bound[2]])
                window_max = np.array([x_max, y_max, max_bound[2]])
                
                within_range = kdtree.query_ball_box(
                    np.array([(window_min + window_max) / 2]),
                    np.array([(window_max - window_min) / 2])
                )[0]
                
                if len(within_range) == 0:
                    pbar.update(1)
                    continue
                
                # Get window points
                window_points = points[within_range]
                
                # Skip if too few points
                if len(window_points) < 100:
                    pbar.update(1)
                    continue
                
                # Normalize window points
                window_points_copy = window_points.copy()
                window_center = (window_min + window_max) / 2
                window_points_copy[:, :3] = window_points_copy[:, :3] - window_center
                
                # Process window in batches
                window_preds = []
                
                for i in range(0, len(window_points_copy), batch_size):
                    batch_points = window_points_copy[i:i+batch_size]
                    batch_tensor = torch.from_numpy(batch_points).float().to(device)
                    if len(batch_tensor.shape) == 2:
                        batch_tensor = batch_tensor.unsqueeze(0)  # Add batch dimension
                    
                    with torch.no_grad():
                        # Forward pass
                        outputs = model(batch_tensor)
                        
                        # Get predictions
                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                        window_preds.append(preds)
                
                # Combine batch predictions
                if len(window_preds) > 0:
                    window_preds = np.concatenate(window_preds)
                    
                    # Update predictions
                    all_preds[within_range] += window_preds
                    all_counts[within_range] += 1
                
                pbar.update(1)
    
    # Average predictions (majority voting)
    valid_mask = all_counts > 0
    all_preds[valid_mask] = all_preds[valid_mask] // all_counts[valid_mask]
    
    return all_preds


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup logger
    logger = setup_logger(output_dir / "inference.log")
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
    batch_size = args.batch_size if args.batch_size else optimal_config['inference_batch_size']
    logger.info(f"Using batch size: {batch_size}")
    
    # Create model
    logger.info("Creating model...")
    model = get_model(config['model'])
    
    # Load model weights
    logger.info(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Optimize model for inference
    model = optimize_model_for_device(model, device_info)
    model = optimize_for_inference(model, device)
    model.eval()
    
    # Check if data_path is a file or directory
    data_path = Path(args.data_path)
    
    if data_path.is_file():
        # Single file inference
        files = [data_path]
    else:
        # Directory inference
        files = list(data_path.glob('*.la[sz]'))  # LAS/LAZ files
    
    logger.info(f"Found {len(files)} point cloud files")
    
    # Process each file
    for file_path in tqdm(files, desc="Processing files"):
        logger.info(f"Processing {file_path}")
        
        # Load point cloud
        try:
            points, las_file = load_point_cloud(file_path)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue
        
        logger.info(f"Loaded {len(points)} points")
        
        # Create preprocessor
        preprocessor = PointCloudPreprocessor(config['data'])
        
        # Preprocess point cloud
        try:
            processed_points, _ = preprocessor.process(points)
        except Exception as e:
            logger.error(f"Error preprocessing {file_path}: {e}")
            # Fall back to raw points if preprocessing fails
            processed_points = points
        
        # Run inference
        start_time = time.time()
        
        if args.use_sliding_window:
            # Sliding window inference for large point clouds
            logger.info("Using sliding window inference")
            predictions = sliding_window_inference(
                model, processed_points, 
                args.window_size, args.window_stride,
                batch_size, device, 
                num_classes=config['model']['segmentation']['num_classes']
            )
        else:
            # Standard inference
            # Convert to tensor
            points_tensor = torch.from_numpy(processed_points).float().to(device)
            if len(points_tensor.shape) == 2:
                points_tensor = points_tensor.unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():
                # Forward pass
                outputs = model(points_tensor)
                
                # Get predictions
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                
                # Remove batch dimension if present
                if len(predictions.shape) > 1:
                    predictions = predictions[0]
        
        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f} seconds")
        
        # Save results
        output_file = output_dir / f"{file_path.stem}_segmented.las"
        
        # Create a new LAS file
        new_las = laspy.create(point_format=las_file.header.point_format)
        
        # Copy header
        for dim in las_file.point_format.dimensions:
            if dim.name in las_file.point_format.dimension_names:
                new_las[dim.name] = las_file[dim.name]
        
        # Add classification
        new_las.classification = predictions
        
        # Write output file
        new_las.write(output_file)
        logger.info(f"Saved results to {output_file}")
    
    logger.info("Inference completed!")


if __name__ == "__main__":
    main()