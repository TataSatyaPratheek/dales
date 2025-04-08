# urban_point_cloud_analyzer/scripts/analyze_urban_area.py
#!/usr/bin/env python3
"""
Comprehensive urban analysis script for Urban Point Cloud Analyzer
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
from urban_point_cloud_analyzer.business.metrics.integrated_metrics import IntegratedUrbanAnalyzer
from urban_point_cloud_analyzer.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive urban analysis")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to segmented point cloud file or directory")
    parser.add_argument("--objects_path", type=str, help="Path to pre-detected objects JSON file (optional)")
    parser.add_argument("--output_dir", type=str, default="analysis", help="Output directory")
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
    logger = setup_logger(output_dir / "urban_analysis.log")
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
    
    # Create urban analyzer
    analyzer = IntegratedUrbanAnalyzer(config.get('business', {}))
    
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
            
            # Load or detect objects
            objects = None
            
            if args.objects_path:
                # Try to load pre-detected objects
                objects_path = Path(args.objects_path)
                if objects_path.exists():
                    logger.info(f"Loading objects from {objects_path}")
                    with open(objects_path, 'r') as f:
                        objects = json.load(f)
            
            if objects is None:
                # Detect objects
                logger.info("Detecting objects...")
                objects = detect_objects(points, labels, config['model']['detection'])
                logger.info(f"Detected {len(objects)} objects")
            
            # Perform urban analysis
            logger.info("Performing urban analysis...")
            metrics = analyzer.analyze(points, labels, objects)
            
            # Generate report
            logger.info("Generating urban analysis report...")
            report = analyzer.generate_report(metrics)
            
            # Save metrics as JSON
            metrics_file = output_dir / f"{file_path.stem}_advanced_metrics.json"
            
            # Convert numpy values to Python native types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                                    np.uint8, np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(i) for i in obj]
                else:
                    return obj
            
            serializable_metrics = convert_to_serializable(metrics)
            
            # Save to JSON
            with open(metrics_file, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
            
            # Save report as text
            report_file = output_dir / f"{file_path.stem}_advanced_report.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            
            logger.info(f"Saved advanced metrics to {metrics_file}")
            logger.info(f"Saved advanced report to {report_file}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    logger.info("Urban analysis completed!")


if __name__ == "__main__":
    main()