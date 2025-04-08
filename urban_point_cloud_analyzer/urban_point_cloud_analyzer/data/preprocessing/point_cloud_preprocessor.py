# urban_point_cloud_analyzer/data/preprocessing/point_cloud_preprocessor.py
import numpy as np
import open3d as o3d
from pathlib import Path
import torch
from typing import Dict, List, Optional, Tuple, Union

class PointCloudPreprocessor:
    """Base class for point cloud preprocessing operations."""
    
    def __init__(self, config: Dict):
        """
        Initialize the preprocessor with configuration parameters.
        
        Args:
            config: Dictionary containing preprocessing parameters
        """
        self.config = config
        self.voxel_size = config.get('voxel_size', 0.05)
    
    def clean_point_cloud(self, points: np.ndarray, min_neighbors: int = 10, 
                          radius: float = 0.1) -> np.ndarray:
        """
        Remove outliers and noise from point cloud.
        
        Args:
            points: (N, 3+) array of point coordinates and features
            min_neighbors: Minimum number of neighbors for a point to be kept
            radius: Radius for neighbor search
            
        Returns:
            Cleaned point cloud as numpy array
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # Statistical outlier removal
        cleaned_pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=min_neighbors, 
            std_ratio=2.0
        )
        
        # Get indices of remaining points
        cleaned_points = np.asarray(cleaned_pcd.points)
        
        # Find correspondence between original and cleaned points
        indices = []
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        for point in cleaned_points:
            _, idx, _ = pcd_tree.search_knn_vector_3d(point, 1)
            indices.append(idx[0])
        
        return points[indices]
    
    def downsample_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """
        Downsample point cloud using voxel grid downsampling.
        
        Args:
            points: (N, 3+) array of point coordinates and features
            
        Returns:
            Downsampled point cloud as numpy array
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        if points.shape[1] > 3:  # If there are features (like colors or normals)
            pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6] / 255.0) if points.shape[1] >= 6 else None
            pcd.normals = o3d.utility.Vector3dVector(points[:, 6:9]) if points.shape[1] >= 9 else None
        
        # Voxel grid downsampling
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        # Convert back to numpy array
        downsampled_points = np.asarray(downsampled_pcd.points)
        
        # Reconstruct features if present
        features = []
        if hasattr(downsampled_pcd, 'colors') and downsampled_pcd.colors:
            features.append(np.asarray(downsampled_pcd.colors) * 255.0)
        if hasattr(downsampled_pcd, 'normals') and downsampled_pcd.normals:
            features.append(np.asarray(downsampled_pcd.normals))
        
        if features:
            downsampled_points = np.hstack([downsampled_points] + features)
        
        return downsampled_points
    
    def estimate_normals(self, points: np.ndarray, k: int = 30) -> np.ndarray:
        """
        Estimate normals for point cloud.
        
        Args:
            points: (N, 3+) array of point coordinates and features
            k: Number of neighbors for normal estimation
            
        Returns:
            Point cloud with normals as numpy array
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
        )
        
        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(k=k)
        
        # Concatenate normals with original points
        normals = np.asarray(pcd.normals)
        
        # Keep original features if present
        if points.shape[1] > 3:
            return np.hstack([points, normals])
        else:
            return np.hstack([points, normals])
    
    def extract_ground_plane(self, points: np.ndarray, 
                            distance_threshold: float = 0.1, 
                            max_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract ground plane from point cloud using RANSAC.
        
        Args:
            points: (N, 3+) array of point coordinates and features
            distance_threshold: Maximum distance from plane to be considered inlier
            max_iterations: Maximum number of RANSAC iterations
            
        Returns:
            Tuple of (ground_points, non_ground_points) as numpy arrays
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # Extract plane using RANSAC
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=max_iterations
        )
        
        # Split into ground and non-ground points
        ground_indices = np.array(inliers)
        non_ground_indices = np.setdiff1d(np.arange(len(points)), ground_indices)
        
        return points[ground_indices], points[non_ground_indices]
    
    def chunk_point_cloud(self, points: np.ndarray, chunk_size: float, 
                          overlap: float = 0.1) -> List[np.ndarray]:
        """
        Divide large point cloud into manageable overlapping chunks.
        
        Args:
            points: (N, 3+) array of point coordinates and features
            chunk_size: Size of each chunk in meters
            overlap: Amount of overlap between chunks as a fraction of chunk_size
            
        Returns:
            List of point cloud chunks as numpy arrays
        """
        # Calculate point cloud bounds
        min_bound = np.min(points[:, :3], axis=0)
        max_bound = np.max(points[:, :3], axis=0)
        
        # Calculate number of chunks in each dimension
        dims = max_bound - min_bound
        overlap_size = chunk_size * overlap
        steps = [max(1, int(np.ceil(dim / (chunk_size - overlap_size)))) for dim in dims[:2]]
        
        chunks = []
        for i in range(steps[0]):
            for j in range(steps[1]):
                # Calculate chunk bounds with overlap
                x_min = min_bound[0] + i * (chunk_size - overlap_size)
                x_max = min(max_bound[0], x_min + chunk_size)
                y_min = min_bound[1] + j * (chunk_size - overlap_size)
                y_max = min(max_bound[1], y_min + chunk_size)
                
                # Extract points in this chunk
                mask = ((points[:, 0] >= x_min) & (points[:, 0] < x_max) & 
                        (points[:, 1] >= y_min) & (points[:, 1] < y_max))
                
                if np.sum(mask) > 0:  # Only add chunk if it contains points
                    chunks.append(points[mask])
        
        return chunks
    
    def compute_geometric_features(self, points: np.ndarray, k: int = 30) -> np.ndarray:
        """
        Compute geometric features like eigenvalues and moments.
        
        Args:
            points: (N, 3+) array of point coordinates and features
            k: Number of neighbors for feature computation
            
        Returns:
            Point cloud with geometric features as numpy array
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # Compute geometric features
        features = []
        
        # Build KDTree for neighbor search
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        eigenvalues = np.zeros((len(points), 3))
        
        for i, point in enumerate(points[:, :3]):
            # Find k nearest neighbors
            _, idx, _ = pcd_tree.search_knn_vector_3d(point, k)
            neighbors = np.asarray(pcd.points)[idx]
            
            # Compute covariance matrix
            centroid = np.mean(neighbors, axis=0)
            centered = neighbors - centroid
            cov = np.dot(centered.T, centered) / k
            
            # Compute eigenvalues
            eigvals, _ = np.linalg.eigh(cov)
            eigenvalues[i] = np.sort(eigvals)[::-1]  # Sort in descending order
        
        # Compute eigenvalue-based features
        # Linearity, planarity, sphericity
        linearity = (eigenvalues[:, 0] - eigenvalues[:, 1]) / (eigenvalues[:, 0] + 1e-6)
        planarity = (eigenvalues[:, 1] - eigenvalues[:, 2]) / (eigenvalues[:, 0] + 1e-6)
        sphericity = eigenvalues[:, 2] / (eigenvalues[:, 0] + 1e-6)
        
        # Stack features
        features = np.column_stack([linearity, planarity, sphericity])
        
        # Concatenate with original points
        return np.hstack([points, features])
    
    def process(self, points: np.ndarray, labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply full preprocessing pipeline to point cloud.
        
        Args:
            points: (N, 3+) array of point coordinates and features
            labels: Optional (N,) array of point labels
            
        Returns:
            Tuple of (processed_points, processed_labels) as numpy arrays
        """
        # Clean point cloud
        cleaned_points = self.clean_point_cloud(points)
        cleaned_labels = labels[np.isin(points, cleaned_points, axis=0).all(axis=1)] if labels is not None else None
        
        # Downsample
        downsampled_points = self.downsample_point_cloud(cleaned_points)
        # Match labels to downsampled points
        if cleaned_labels is not None:
            # Find nearest neighbors for label assignment
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cleaned_points[:, :3])
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            
            downsampled_labels = np.zeros(len(downsampled_points), dtype=cleaned_labels.dtype)
            for i, point in enumerate(downsampled_points[:, :3]):
                _, idx, _ = pcd_tree.search_knn_vector_3d(point, 1)
                downsampled_labels[i] = cleaned_labels[idx[0]]
        else:
            downsampled_labels = None
        
        # Estimate normals
        points_with_normals = self.estimate_normals(downsampled_points)
        
        # Compute geometric features
        processed_points = self.compute_geometric_features(points_with_normals)
        
        return processed_points, downsampled_labels