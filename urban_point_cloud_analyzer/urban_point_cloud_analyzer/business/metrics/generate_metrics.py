# urban_point_cloud_analyzer/scripts/generate_metrics.py
#!/usr/bin/env python3
"""
Generate urban metrics from segmented point clouds
"""
import argparse
import os
import sys
import yaml
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import laspy

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from urban_point_cloud_analyzer.business.metrics.urban_metrics import calculate_urban_metrics, generate_urban_analysis_report
from urban_point_cloud_analyzer.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Generate urban metrics from segmented point clouds")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to segmented point cloud file or directory")
    parser.add_argument("--output_dir", type=str, default="metrics", help="Output directory")
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
    logger = setup_logger(output_dir / "metrics.log")
    logger.info(f"Config: {config}")
    
    # Check if data_path is a file or directory
    data_path = Path(args.data_path)
    
    if data_path.is_file():
        # Single file processing
        files = [data_path]
    else:
        # Directory processing
        files = list(data_path.glob('*.la[sz]'))  # LAS/LAZ files
    
    logger.info(f"Found {len(files)} point cloud files")
    
    # Process each file
    for file_path in tqdm(files, desc="Processing files"):
        logger.info(f"Processing {file_path}")
        
        # Load point cloud with segmentation
        try:
            las = laspy.read(file_path)
            
            # Extract coordinates
            points = np.vstack([
                las.x, las.y, las.z
            ]).T
            
            # Extract segmentation labels
            if hasattr(las, 'classification'):
                labels = np.array(las.classification)
            else:
                logger.warning(f"No classification found in {file_path}")
                continue
            
            logger.info(f"Loaded {len(points)} points")
            
            # Calculate urban metrics
            metrics = calculate_urban_metrics(points, labels)
            
            # Generate report
            report = generate_urban_analysis_report(metrics)
            
            # Save metrics as JSON
            metrics_file = output_dir / f"{file_path.stem}_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save report as text
            report_file = output_dir / f"{file_path.stem}_report.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            
            logger.info(f"Saved metrics to {metrics_file}")
            logger.info(f"Saved report to {report_file}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    logger.info("Metrics generation completed!")


if __name__ == "__main__":
    main()