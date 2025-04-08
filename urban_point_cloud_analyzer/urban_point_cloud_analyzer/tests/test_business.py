# urban_point_cloud_analyzer/tests/test_business.py
import os
import sys
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def test_urban_metrics():
    """Test basic urban metrics calculation."""
    try:
        from urban_point_cloud_analyzer.business.metrics.urban_metrics import (
            calculate_convex_hull_area, calculate_urban_metrics
        )
        
        # Create a simple test point cloud
        num_points = 1000
        # Create a square area
        x = np.random.uniform(0, 100, num_points)
        y = np.random.uniform(0, 100, num_points)
        z = np.random.uniform(0, 10, num_points)
        
        points = np.column_stack([x, y, z])
        
        # Create segmentation with all classes
        segmentation = np.zeros(num_points, dtype=np.int32)
        segment_size = num_points // 8
        
        for i in range(8):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < 7 else num_points
            segmentation[start_idx:end_idx] = i
        
        # Test convex hull area
        area = calculate_convex_hull_area(points[:, 0:2])
        assert area > 0, "Convex hull area should be positive"
        
        # Test full metrics calculation
        metrics = calculate_urban_metrics(points, segmentation)
        
        assert 'total_area_m2' in metrics, "Missing total_area_m2 in metrics"
        assert 'building_density' in metrics, "Missing building_density in metrics"
        assert 'green_coverage' in metrics, "Missing green_coverage in metrics"
        
        # Check if densities are within reasonable range
        assert 0 <= metrics['building_density'] <= 1, f"Building density should be between 0 and 1, got {metrics['building_density']}"
        assert 0 <= metrics['green_coverage'] <= 1, f"Green coverage should be between 0 and 1, got {metrics['green_coverage']}"
        
        print(f"✓ Urban metrics test passed")
        return True
    except Exception as e:
        print(f"✗ Urban metrics test failed: {e}")
        return False

def test_advanced_metrics():
    """Test advanced urban metrics calculation with proper error handling."""
    try:
        from urban_point_cloud_analyzer.business.metrics.advanced_metrics import (
            calculate_road_connectivity, calculate_accessibility_metrics
        )
        import numpy as np
        
        # Create a properly structured test point cloud that won't cause warnings
        num_points = 1000
        
        # Create a square area with points for all necessary classes
        x = np.random.uniform(0, 100, num_points)
        y = np.random.uniform(0, 100, num_points)
        z = np.random.uniform(0, 10, num_points)
        
        points = np.column_stack([x, y, z])
        
        # Create segmentation with ground (0) and buildings (2)
        segmentation = np.zeros(num_points, dtype=np.int32)
        
        # First 60% is ground (roads)
        segmentation[:int(num_points*0.6)] = 0
        
        # 30% is buildings
        segmentation[int(num_points*0.6):int(num_points*0.9)] = 2
        
        # 10% is vegetation
        segmentation[int(num_points*0.9):] = 1
        
        # Ensure minimum number of points in each necessary class
        assert np.sum(segmentation == 0) >= 100, "Not enough ground points"
        assert np.sum(segmentation == 2) >= 100, "Not enough building points"
        
        # Test road connectivity metrics
        road_metrics = calculate_road_connectivity(points, segmentation)
        
        assert 'connectivity_score' in road_metrics, "Missing connectivity_score in road metrics"
        assert 'road_density' in road_metrics, "Missing road_density in road metrics"
        
        # Test accessibility metrics
        accessibility_metrics = calculate_accessibility_metrics(points, segmentation)
        
        assert 'building_to_road_accessibility' in accessibility_metrics, "Missing building_to_road_accessibility"
        
        print(f"✓ Advanced metrics test passed")
        return True
    except Exception as e:
        print(f"✗ Advanced metrics test failed: {e}")
        return False

# urban_point_cloud_analyzer/tests/test_business.py
def test_integrated_metrics():
    """Test integrated urban analysis with added safeguards."""
    try:
        from urban_point_cloud_analyzer.business.metrics.integrated_metrics import IntegratedUrbanAnalyzer
        import numpy as np
        
        # Create a simple test point cloud with sufficient points in all classes
        num_points = 1000
        # Create a square area with more variety
        x = np.random.uniform(0, 100, num_points)
        y = np.random.uniform(0, 100, num_points)
        z = np.random.uniform(0, 10, num_points)
        
        points = np.column_stack([x, y, z])
        
        # Create segmentation with all classes
        segmentation = np.zeros(num_points, dtype=np.int32)
        segment_size = num_points // 8
        
        for i in range(8):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < 7 else num_points
            segmentation[start_idx:end_idx] = i
        
        # Ensure we have a good number of points in each class to avoid empty arrays
        for i in range(8):
            class_count = np.sum(segmentation == i)
            assert class_count >= 100, f"Not enough points in class {i}: {class_count}"
        
        # Create analyzer
        analyzer = IntegratedUrbanAnalyzer()
        
        # Test analysis with error suppression
        with np.errstate(invalid='ignore', divide='ignore'):  # Suppress numpy warnings
            metrics = analyzer.analyze(points, segmentation)
        
        assert 'urban_quality_score' in metrics, "Missing urban_quality_score in metrics"
        assert 'road' in metrics, "Missing road metrics"
        assert 'accessibility' in metrics, "Missing accessibility metrics"
        
        # Test report generation
        report = analyzer.generate_report(metrics)
        assert isinstance(report, str), "Report should be a string"
        assert len(report) > 0, "Report should not be empty"
        
        print(f"✓ Integrated metrics test passed")
        return True
    except Exception as e:
        print(f"✗ Integrated metrics test failed: {e}")
        return False

def run_business_tests():
    """Run all business metrics tests."""
    tests = [
        test_urban_metrics,
        test_advanced_metrics,
        test_integrated_metrics,
    ]
    
    results = []
    for test in tests:
        print(f"Running {test.__name__}...")
        result = test()
        results.append(result)
        print("")
    
    success_count = sum(1 for r in results if r)
    total_count = len(results)
    
    print(f"Business metrics tests completed: {success_count}/{total_count} successful")
    
    return all(results)

if __name__ == "__main__":
    success = run_business_tests()
    sys.exit(0 if success else 1)