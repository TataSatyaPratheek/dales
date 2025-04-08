# urban_point_cloud_analyzer/models/detection/__init__.py
from .object_detector import PointCloudObjectDetector
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

def detect_objects(points: np.ndarray, 
                 segmentation: np.ndarray, 
                 config: Dict) -> List[Dict]:
    """
    Detect objects in a segmented point cloud.
    
    Args:
        points: (N, 3+) array of point coordinates and features
        segmentation: (N,) array of segmentation labels
        config: Configuration dictionary
        
    Returns:
        List of detected objects
    """
    detector = PointCloudObjectDetector(config)
    return detector.detect_objects(points, segmentation)

__all__ = ['PointCloudObjectDetector', 'detect_objects']