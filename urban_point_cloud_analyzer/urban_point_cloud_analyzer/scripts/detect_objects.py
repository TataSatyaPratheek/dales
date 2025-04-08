# urban_point_cloud_analyzer/scripts/detect_objects.py
#!/usr/bin/env python3
"""
Object detection script for Urban Point Cloud Analyzer
"""
import argparse
import os
import sys
import yaml
import json
from pathlib import Path
import numpy as np
import laspy

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from urban_point_cloud_analyzer.models.detection import detect_objects
from urban_point_cloud_analyzer.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Detect objects in point clouds")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to segmented point cloud file or directory")
    parser.add_argument("--output_dir", type=str, default="objects", help="Output directory")
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
    logger = setup_logger(output_dir / "object_detection.log")
    logger.info(f"Config: {config}")
    
    # Check if data_path is a file or directory
    data_path = Path(args.data_path)
    
    if data_path.is_file():
        # Single file processing
        files = [data_path]
    else:
        # Directory processing
        files = list(data_path.glob('*_segmented.la[sz]'))  # Segmented LAS/LAZ files
    
    logger.info(f"Found {len(files)} segmented point cloud files")
    
    # Process each file
    for file_path in files:
        logger.info(f"Processing {file_path}")
        
        try:
            # Load segmented point cloud
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
            
            # Detect objects
            logger.info("Detecting objects...")
            objects = detect_objects(points, labels, config['model']['detection'])
            
            logger.info(f"Detected {len(objects)} objects")
            
            # Save objects as JSON
            objects_file = output_dir / f"{file_path.stem}_objects.json"
            
            # Convert objects to serializable format
            serializable_objects = []
            for obj in objects:
                serializable_obj = {
                    'class_id': int(obj['class_id']),
                    'class_name': obj['class_name'],
                    'center': obj['center'].tolist(),
                    'dimensions': obj['dimensions'].tolist(),
                    'confidence': float(obj['confidence']),
                    'attributes': {k: float(v) for k, v in obj['attributes'].items()}
                }
                serializable_objects.append(serializable_obj)
            
            # Save to JSON
            with open(objects_file, 'w') as f:
                json.dump(serializable_objects, f, indent=2)
            
            logger.info(f"Saved objects to {objects_file}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    logger.info("Object detection completed!")


if __name__ == "__main__":
    main()