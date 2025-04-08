# urban_point_cloud_analyzer/data/augmentation/point_cloud_transforms.py
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union

class PointCloudTransforms:
    """Class for point cloud data augmentation."""
    
    def __init__(self, config: Dict):
        """
        Initialize the transforms with configuration parameters.
        
        Args:
            config: Dictionary containing augmentation parameters
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.rotation_range = config.get('rotation_range', 15)
        self.flip_probability = config.get('flip_probability', 0.5)
        self.dropout_probability = config.get('dropout_probability', 0.1)
        self.noise_sigma = config.get('noise_sigma', 0.01)
    
    def random_rotation(self, points: np.ndarray) -> np.ndarray:
        """
        Apply random rotation around z-axis.
        
        Args:
            points: (N, 3+) array of point coordinates and features
            
        Returns:
            Randomly rotated point cloud
        """
        if not self.enabled:
            return points
        
        # Random angle in degrees, convert to radians
        theta = np.random.uniform(-self.rotation_range, self.rotation_range) * np.pi / 180
        
        # Rotation matrix around z-axis
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        
        # Apply rotation to xyz coordinates
        rotated_xyz = np.dot(points[:, :3], R.T)
        
        # Replace xyz coordinates
        result = points.copy()
        result[:, :3] = rotated_xyz
        
        # Rotate normals if present (assuming normals are at indices 6-8)
        if points.shape[1] >= 9:
            rotated_normals = np.dot(points[:, 6:9], R.T)
            result[:, 6:9] = rotated_normals
        
        return result
    
    def random_scaling(self, points: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        Apply random scaling.
        
        Args:
            points: (N, 3+) array of point coordinates and features
            scale_range: Tuple of (min_scale, max_scale)
            
        Returns:
            Randomly scaled point cloud
        """
        if not self.enabled:
            return points
        
        # Random scale factor
        scale = np.random.uniform(scale_range[0], scale_range[1])
        
        # Apply scaling to xyz coordinates
        result = points.copy()
        result[:, :3] = points[:, :3] * scale
        
        return result
    
    def random_jitter(self, points: np.ndarray) -> np.ndarray:
        """
        Add random noise to point coordinates.
        
        Args:
            points: (N, 3+) array of point coordinates and features
            
        Returns:
            Jittered point cloud
        """
        if not self.enabled or self.noise_sigma <= 0:
            return points
        
        # Generate random noise
        noise = np.random.normal(0, self.noise_sigma, size=points[:, :3].shape)
        
        # Apply noise to xyz coordinates
        result = points.copy()
        result[:, :3] = points[:, :3] + noise
        
        return result
    
    def random_dropout(self, points: np.ndarray) -> np.ndarray:
        """
        Randomly drop points.
        
        Args:
            points: (N, 3+) array of point coordinates and features
            
        Returns:
            Point cloud with randomly dropped points
        """
        if not self.enabled or self.dropout_probability <= 0:
            return points
        
        # Generate dropout mask
        mask = np.random.random(size=points.shape[0]) > self.dropout_probability
        
        # Apply mask
        return points[mask]
    
    def random_flip(self, points: np.ndarray) -> np.ndarray:
        """
        Randomly flip points along x or y axis.
        
        Args:
            points: (N, 3+) array of point coordinates and features
            
        Returns:
            Randomly flipped point cloud
        """
        if not self.enabled:
            return points
        
        result = points.copy()
        
        # Flip x-axis
        if np.random.random() < self.flip_probability:
            result[:, 0] = -result[:, 0]
            # Flip normals in x direction if present
            if points.shape[1] >= 9:
                result[:, 6] = -result[:, 6]
        
        # Flip y-axis
        if np.random.random() < self.flip_probability:
            result[:, 1] = -result[:, 1]
            # Flip normals in y direction if present
            if points.shape[1] >= 9:
                result[:, 7] = -result[:, 7]
        
        return result
    
    def apply(self, points: np.ndarray, labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply all transformations to point cloud.
        
        Args:
            points: (N, 3+) array of point coordinates and features
            labels: Optional (N,) array of point labels
            
        Returns:
            Tuple of (transformed_points, transformed_labels)
        """
        if not self.enabled:
            return points, labels
        
        # Apply transformations in sequence
        transformed_points = self.random_rotation(points)
        transformed_points = self.random_scaling(transformed_points)
        transformed_points = self.random_jitter(transformed_points)
        transformed_points = self.random_flip(transformed_points)
        
        # Handle point dropout specially as it changes the number of points
        if np.random.random() < 0.5:  # Apply dropout with 50% chance
            dropout_mask = np.random.random(size=transformed_points.shape[0]) > self.dropout_probability
            transformed_points = transformed_points[dropout_mask]
            if labels is not None:
                labels = labels[dropout_mask]
        
        return transformed_points, labels
    
    def apply_batch(self, points_batch: torch.Tensor, labels_batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply transformations to a batch of point clouds.
        
        Args:
            points_batch: (B, N, 3+) tensor of point clouds
            labels_batch: Optional (B, N) tensor of point labels
            
        Returns:
            Tuple of (transformed_points_batch, transformed_labels_batch)
        """
        if not self.enabled:
            return points_batch, labels_batch
        
        # Convert to numpy for processing
        is_tensor = isinstance(points_batch, torch.Tensor)
        if is_tensor:
            points_list = points_batch.cpu().numpy()
            labels_list = labels_batch.cpu().numpy() if labels_batch is not None else None
        else:
            points_list = points_batch
            labels_list = labels_batch
        
        # Process each point cloud in the batch
        transformed_points = []
        transformed_labels = [] if labels_list is not None else None
        
        for i in range(len(points_list)):
            points = points_list[i]
            labels = labels_list[i] if labels_list is not None else None
            
            # Apply transformations
            t_points, t_labels = self.apply(points, labels)
            
            transformed_points.append(t_points)
            if transformed_labels is not None:
                transformed_labels.append(t_labels)
        
        # Convert back to tensor if needed
        if is_tensor:
            transformed_points = torch.from_numpy(np.array(transformed_points)).to(points_batch.device)
            if transformed_labels is not None:
                transformed_labels = torch.from_numpy(np.array(transformed_labels)).to(labels_batch.device)
        
        return transformed_points, transformed_labels