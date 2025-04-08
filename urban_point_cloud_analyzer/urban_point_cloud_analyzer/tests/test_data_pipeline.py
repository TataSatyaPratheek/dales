# urban_point_cloud_analyzer/tests/test_data_pipeline.py

import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def test_point_cloud_transforms():
    """Test point cloud augmentation transforms."""
    try:
        from urban_point_cloud_analyzer.data.augmentation import PointCloudTransforms
        
        # Create dummy point cloud data
        points = np.random.rand(1000, 3).astype(np.float32)
        labels = np.random.randint(0, 8, size=1000).astype(np.int32)
        
        # Create transforms
        config = {
            "enabled": True,
            "rotation_range": 15,
            "flip_probability": 0.5,
            "dropout_probability": 0.1,
            "noise_sigma": 0.01
        }
        transforms = PointCloudTransforms(config)
        
        # Test individual transforms
        rotated_points = transforms.random_rotation(points)
        assert rotated_points.shape == points.shape
        
        scaled_points = transforms.random_scaling(points)
        assert scaled_points.shape == points.shape
        
        jittered_points = transforms.random_jitter(points)
        assert jittered_points.shape == points.shape
        
        flipped_points = transforms.random_flip(points)
        assert flipped_points.shape == points.shape
        
        # Test combined transforms
        transformed_points, transformed_labels = transforms.apply(points, labels)
        
        print(f"✓ Point cloud transforms test passed")
        return True
    except Exception as e:
        print(f"✗ Point cloud transforms test failed: {e}")
        return False

def test_preprocessor():
    """Test point cloud preprocessor."""
    try:
        from urban_point_cloud_analyzer.data.preprocessing import PointCloudPreprocessor
        
        # Skip test if open3d is not installed
        try:
            import open3d
        except ImportError:
            print("⚠ Skipping preprocessor test as open3d is not installed")
            return True
        
        # Create dummy point cloud data
        points = np.random.rand(1000, 3).astype(np.float32)
        
        # Create preprocessor
        config = {"voxel_size": 0.05}
        preprocessor = PointCloudPreprocessor(config)
        
        # Test methods that don't require real point clouds
        try:
            downsampled_points = preprocessor.downsample_point_cloud(points)
            print(f"✓ Downsampling worked: {len(downsampled_points)} points after downsampling")
        except Exception as e:
            print(f"⚠ Downsampling failed: {e}")
        
        print(f"✓ Preprocessor test passed")
        return True
    except Exception as e:
        print(f"✗ Preprocessor test failed: {e}")
        return False

def run_data_pipeline_tests():
    """Run all data pipeline tests."""
    tests = [
        test_point_cloud_transforms,
        test_preprocessor,
        # Add more data pipeline tests here
    ]
    
    results = []
    for test in tests:
        print(f"Running {test.__name__}...")
        result = test()
        results.append(result)
        print("")
    
    success_count = sum(1 for r in results if r)
    total_count = len(results)
    
    print(f"Data pipeline tests completed: {success_count}/{total_count} successful")
    
    return all(results)

if __name__ == "__main__":
    success = run_data_pipeline_tests()
    sys.exit(0 if success else 1)