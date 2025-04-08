# urban_point_cloud_analyzer/data/loaders/dales_dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import laspy
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple, Union, Callable
import pickle

from urban_point_cloud_analyzer.data.preprocessing import PointCloudPreprocessor
from urban_point_cloud_analyzer.data.augmentation import PointCloudTransforms

class DALESDataset(Dataset):
    """
    Dataset class for DALES point cloud data
    """
    def __init__(self, 
                 root_dir: Union[str, Path], 
                 split: str = 'train', 
                 config: Dict = None,
                 transforms: Optional[PointCloudTransforms] = None, 
                 preprocessor: Optional[PointCloudPreprocessor] = None,
                 cache: bool = True,
                 num_points: int = 16384,
                 memory_efficient: bool = False):
        """
        Args:
            root_dir: Directory with DALES data
            split: 'train', 'val', or 'test'
            config: Configuration dictionary
            transforms: Optional transformations
            preprocessor: Optional preprocessor
            cache: Whether to cache processed data
            num_points: Number of points to sample from each point cloud
            memory_efficient: Whether to use memory-efficient mode
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.config = config if config is not None else {}
        self.transforms = transforms
        self.preprocessor = preprocessor
        self.cache_dir = self.root_dir / 'cache'
        self.cache = cache
        self.num_points = num_points
        self.memory_efficient = memory_efficient
        
        # Class mapping
        self.classes = {
            0: 'Ground',
            1: 'Vegetation',
            2: 'Buildings',
            3: 'Water',
            4: 'Car',
            5: 'Truck',
            6: 'Powerline',
            7: 'Fence'
        }
        
        # Get file list based on split
        self.files = self._get_file_list()
        
        # Create cache directory if needed
        if self.cache:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_file_list(self) -> List[Path]:
        """Get list of files for the specified split"""
        # All LAS files in the dataset
        all_files = list(self.root_dir.glob('*.las'))
        
        # Determine train/val/test split based on config or default
        train_ratio = self.config.get('train_split', 0.7)
        val_ratio = self.config.get('val_split', 0.15)
        
        # Make sure we have a consistent split
        random.seed(42)
        all_files.sort()  # Sort for reproducibility
        random.shuffle(all_files)
        
        num_files = len(all_files)
        train_end = int(num_files * train_ratio)
        val_end = train_end + int(num_files * val_ratio)
        
        if self.split == 'train':
            return all_files[:train_end]
        elif self.split == 'val':
            return all_files[train_end:val_end]
        elif self.split == 'test':
            return all_files[val_end:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def _load_file(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load LAS file and extract points and labels.
        
        Args:
            file_path: Path to LAS file
            
        Returns:
            Tuple of (points, labels) as numpy arrays
        """
        try:
            # Check cache first
            if self.cache:
                cache_path = self.cache_dir / f"{file_path.stem}.pkl"
                if cache_path.exists():
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                    return data['points'], data['labels']
            
            # Load LAS file
            las = laspy.read(file_path)
            
            # Extract coordinates
            points = np.vstack([
                las.x, las.y, las.z
            ]).T
            
            # Extract intensity if available
            if hasattr(las, 'intensity'):
                intensity = np.array(las.intensity).reshape(-1, 1)
                points = np.hstack([points, intensity])
            
            # Extract labels (DALES uses classification field for labels)
            labels = np.array(las.classification)
            
            # Apply preprocessing if available
            if self.preprocessor:
                points, labels = self.preprocessor.process(points, labels)
            
            # Cache processed data
            if self.cache:
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'points': points,
                        'labels': labels
                    }, f)
            
            return points, labels
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return empty arrays in case of error
            return np.zeros((0, 3)), np.zeros(0, dtype=np.int32)
    
    def _sample_points(self, points: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample fixed number of points from point cloud.
        
        Args:
            points: (N, D) array of point coordinates and features
            labels: (N,) array of point labels
            
        Returns:
            Tuple of (sampled_points, sampled_labels) as numpy arrays
        """
        num_points_in_cloud = points.shape[0]
        
        if num_points_in_cloud <= self.num_points:
            # If too few points, duplicate some
            indices = np.random.choice(num_points_in_cloud, self.num_points, replace=True)
        else:
            # If too many points, subsample
            indices = np.random.choice(num_points_in_cloud, self.num_points, replace=False)
        
        return points[indices], labels[indices]
    
    def __len__(self) -> int:
        """Return number of point clouds in the dataset"""
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Load point cloud and labels.
        
        Args:
            idx: Index of point cloud to load
            
        Returns:
            Dictionary with 'points' and 'labels' keys
        """
        file_path = self.files[idx]
        
        # Load points and labels
        if self.memory_efficient:
            # In memory-efficient mode, just load the file without preprocessing
            # Preprocessing will be done on the fly later
            las = laspy.read(file_path)
            points = np.vstack([
                las.x, las.y, las.z
            ]).T
            
            # Extract intensity if available
            if hasattr(las, 'intensity'):
                intensity = np.array(las.intensity).reshape(-1, 1)
                points = np.hstack([points, intensity])
            
            labels = np.array(las.classification)
        else:
            # Regular mode with preprocessing
            points, labels = self._load_file(file_path)
        
        # Sample fixed number of points
        points, labels = self._sample_points(points, labels)
        
        # Apply transforms if available
        if self.transforms:
            points, labels = self.transforms.apply(points, labels)
        
        # Convert to torch tensors
        points_tensor = torch.from_numpy(points).float()
        labels_tensor = torch.from_numpy(labels).long()
        
        return {
            'points': points_tensor,
            'labels': labels_tensor,
            'filename': file_path.stem
        }
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Custom collate function for batching point clouds of different sizes.
        
        Args:
            batch: List of dictionaries from __getitem__
            
        Returns:
            Batched data as a dictionary
        """
        points = [item['points'] for item in batch]
        labels = [item['labels'] for item in batch]
        filenames = [item['filename'] for item in batch]
        
        # Find max number of points
        max_points = max([p.shape[0] for p in points])
        
        # Pad all point clouds to the same size
        points_batch = []
        labels_batch = []
        for p, l in zip(points, labels):
            num_points = p.shape[0]
            padded_points = torch.zeros((max_points, p.shape[1]), dtype=p.dtype)
            padded_labels = torch.zeros(max_points, dtype=l.dtype)
            
            padded_points[:num_points] = p
            padded_labels[:num_points] = l
            
            points_batch.append(padded_points)
            labels_batch.append(padded_labels)
        
        # Stack into tensors
        points_tensor = torch.stack(points_batch, dim=0)
        labels_tensor = torch.stack(labels_batch, dim=0)
        
        return {
            'points': points_tensor,
            'labels': labels_tensor,
            'filenames': filenames
        }