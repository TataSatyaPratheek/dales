# urban_point_cloud_analyzer/models/detection/object_detector.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import open3d as o3d

class PointCloudObjectDetector:
    """
    Object detector for urban infrastructure in point clouds.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize object detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.min_confidence = config.get('min_confidence', 0.5)
        self.nms_threshold = config.get('nms_threshold', 0.3)
        self.min_points = config.get('min_points', 50)
        self.max_points = config.get('max_points', 100000)
        
        # Object classes (subset of segmentation classes)
        self.object_classes = {
            4: 'Car',
            5: 'Truck',
            6: 'Powerline',
            7: 'Fence'
        }
    
    def detect_objects(self, 
                      points: np.ndarray, 
                      segmentation: np.ndarray) -> List[Dict]:
        """
        Detect objects in a segmented point cloud.
        
        Args:
            points: (N, 3+) array of point coordinates and features
            segmentation: (N,) array of segmentation labels
            
        Returns:
            List of detected objects
        """
        objects = []
        
        # Process each object class
        for class_id, class_name in self.object_classes.items():
            # Get points of this class
            class_mask = segmentation == class_id
            class_points = points[class_mask]
            
            if len(class_points) < self.min_points:
                continue
            
            # Cluster points into individual objects
            clusters = self._cluster_points(class_points)
            
            # Process each cluster
            for i, cluster in enumerate(clusters):
                if len(cluster) < self.min_points:
                    continue
                
                # Calculate object properties
                properties = self._calculate_object_properties(cluster, class_id)
                
                # Add to objects list
                objects.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'points': cluster,
                    'center': properties['center'],
                    'dimensions': properties['dimensions'],
                    'orientation': properties['orientation'],
                    'confidence': properties['confidence'],
                    'attributes': properties['attributes']
                })
        
        # Apply non-maximum suppression
        objects = self._apply_nms(objects)
        
        return objects
    
    def _cluster_points(self, points: np.ndarray) -> List[np.ndarray]:
        """
        Cluster points into individual objects.
        
        Args:
            points: (N, 3+) array of point coordinates
            
        Returns:
            List of point clusters
        """
        # Limit number of points for processing efficiency
        if len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # Use DBSCAN clustering
        eps = 0.5  # 0.5 meters
        min_points = self.min_points
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
        
        # Group points by cluster
        clusters = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            
            # Get points in this cluster
            cluster_mask = labels == label
            cluster_points = points[cluster_mask]
            
            clusters.append(cluster_points)
        
        return clusters
    
    def _calculate_object_properties(self, points: np.ndarray, class_id: int) -> Dict:
        """
        Calculate object properties.
        
        Args:
            points: (N, 3+) array of point coordinates
            class_id: Object class ID
            
        Returns:
            Dictionary of object properties
        """
        # Calculate center
        center = np.mean(points[:, :3], axis=0)
        
        # Calculate oriented bounding box
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # Get axis-aligned bounding box
        aabb = pcd.get_axis_aligned_bounding_box()
        min_bound = np.asarray(aabb.min_bound)
        max_bound = np.asarray(aabb.max_bound)
        
        # Calculate dimensions
        dimensions = max_bound - min_bound
        
        # Try to calculate oriented bounding box (may fail for small clusters)
        try:
            obb = pcd.get_oriented_bounding_box()
            orientation = np.asarray(obb.R)  # Rotation matrix
            dimensions = np.asarray(obb.extent)
        except Exception:
            # Fall back to axis-aligned orientation
            orientation = np.eye(3)
        
        # Calculate confidence based on number of points
        confidence = min(1.0, len(points) / 1000)
        
        # Calculate class-specific attributes
        attributes = {}
        
        if class_id == 4:  # Car
            # Estimate car attributes
            attributes['height'] = dimensions[2]
            attributes['length'] = max(dimensions[0], dimensions[1])
            attributes['width'] = min(dimensions[0], dimensions[1])
            attributes['volume'] = dimensions[0] * dimensions[1] * dimensions[2]
            
        elif class_id == 5:  # Truck
            # Estimate truck attributes
            attributes['height'] = dimensions[2]
            attributes['length'] = max(dimensions[0], dimensions[1])
            attributes['width'] = min(dimensions[0], dimensions[1])
            attributes['volume'] = dimensions[0] * dimensions[1] * dimensions[2]
            
        elif class_id == 6:  # Powerline
            # Estimate powerline attributes
            attributes['length'] = np.linalg.norm(max_bound - min_bound)
            attributes['height'] = np.mean(points[:, 2])
            
        elif class_id == 7:  # Fence
            # Estimate fence attributes
            attributes['length'] = max(dimensions[0], dimensions[1])
            attributes['height'] = dimensions[2]
        
        return {
            'center': center,
            'dimensions': dimensions,
            'orientation': orientation,
            'confidence': confidence,
            'attributes': attributes
        }
    
    def _apply_nms(self, objects: List[Dict]) -> List[Dict]:
        """
        Apply non-maximum suppression to objects.
        
        Args:
            objects: List of detected objects
            
        Returns:
            List of objects after NMS
        """
        if not objects:
            return []
        
        # Group objects by class
        objects_by_class = {}
        for obj in objects:
            class_id = obj['class_id']
            if class_id not in objects_by_class:
                objects_by_class[class_id] = []
            objects_by_class[class_id].append(obj)
        
        # Apply NMS to each class
        filtered_objects = []
        
        for class_id, class_objects in objects_by_class.items():
            # Sort by confidence
            class_objects = sorted(class_objects, key=lambda x: x['confidence'], reverse=True)
            
            # Apply NMS
            while class_objects:
                # Take most confident object
                current_obj = class_objects[0]
                filtered_objects.append(current_obj)
                
                # Remove current object
                class_objects = class_objects[1:]
                
                # Remove overlapping objects
                non_overlapping = []
                for obj in class_objects:
                    iou = self._calculate_iou(current_obj, obj)
                    if iou < self.nms_threshold:
                        non_overlapping.append(obj)
                
                class_objects = non_overlapping
        
        return filtered_objects
    
    def _calculate_iou(self, obj1: Dict, obj2: Dict) -> float:
        """
        Calculate IoU between two objects.
        
        Args:
            obj1: First object
            obj2: Second object
            
        Returns:
            IoU value
        """
        # Calculate as distance-based IoU
        center1 = obj1['center']
        center2 = obj2['center']
        
        # Calculate distance between centers
        distance = np.linalg.norm(center1 - center2)
        
        # Calculate average dimensions
        dims1 = obj1['dimensions']
        dims2 = obj2['dimensions']
        avg_size = (np.mean(dims1) + np.mean(dims2)) / 2
        
        # Convert to IoU-like metric
        iou = 1.0 - min(1.0, distance / (avg_size + 1e-6))
        
        return iou