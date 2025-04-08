# urban_point_cloud_analyzer/scripts/visualize.py
#!/usr/bin/env python3
"""
Visualization script for Urban Point Cloud Analyzer
"""
import argparse
import os
import sys
import yaml
import json
import numpy as np
from pathlib import Path
import laspy

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from urban_point_cloud_analyzer.visualization.visualizer import PointCloudVisualizer
from urban_point_cloud_analyzer.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Urban Point Cloud Analyzer results")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--point_cloud", type=str, required=True, help="Path to point cloud file")
    parser.add_argument("--segmentation", type=str, help="Path to segmentation file")
    parser.add_argument("--metrics", type=str, help="Path to metrics JSON file")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Output directory")
    parser.add_argument("--view_mode", type=str, default="predictions", 
                        choices=["predictions", "labels", "error", "side_by_side"], help="Visualization mode")
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
    logger = setup_logger(output_dir / "visualization.log")
    logger.info(f"Config: {config}")
    
    # Create visualizer
    visualizer = PointCloudVisualizer(config)
    
    # Handle point cloud visualization
    if args.point_cloud:
        point_cloud_path = Path(args.point_cloud)
        logger.info(f"Visualizing point cloud: {point_cloud_path}")
        
        try:
            # Load point cloud
            las = laspy.read(point_cloud_path)
            
            # Extract points
            points = np.vstack([
                las.x, las.y, las.z
            ]).T
            
            # Extract labels if available
            labels = None
            if hasattr(las, 'classification'):
                labels = np.array(las.classification)
            
            # Load segmentation if provided
            predictions = None
            if args.segmentation:
                seg_path = Path(args.segmentation)
                logger.info(f"Loading segmentation: {seg_path}")
                
                if seg_path.suffix.lower() in ['.las', '.laz']:
                    # Load segmentation from LAS/LAZ file
                    seg_las = laspy.read(seg_path)
                    predictions = np.array(seg_las.classification)
                elif seg_path.suffix.lower() == '.npy':
                    # Load segmentation from NumPy file
                    predictions = np.load(seg_path)
                else:
                    logger.error(f"Unsupported segmentation file format: {seg_path.suffix}")
            
            # Create output filename
            output_file = output_dir / f"{point_cloud_path.stem}_{args.view_mode}.png"
            
            # Visualize point cloud
            visualizer.visualize_point_cloud(
                points=points,
                labels=labels,
                predictions=predictions,
                view_mode=args.view_mode,
                filename=str(output_file)
            )
            
            logger.info(f"Visualization saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error visualizing point cloud: {e}")
    
    # Handle metrics visualization
    if args.metrics:
        metrics_path = Path(args.metrics)
        logger.info(f"Visualizing metrics: {metrics_path}")
        
        try:
            # Load metrics
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Create output filename
            output_file = output_dir / f"{metrics_path.stem}_metrics.png"
            
            # Visualize metrics
            visualizer.visualize_metrics(
                metrics=metrics,
                filename=str(output_file)
            )
            
            logger.info(f"Metrics visualization saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error visualizing metrics: {e}")
    
    logger.info("Visualization completed!")


if __name__ == "__main__":
    main()