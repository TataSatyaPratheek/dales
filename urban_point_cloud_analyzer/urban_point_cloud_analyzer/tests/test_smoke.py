# urban_point_cloud_analyzer/tests/test_smoke.py

import os
import sys
import importlib
import numpy as np
import torch
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported."""
    modules = [
        "urban_point_cloud_analyzer",
        "urban_point_cloud_analyzer.data.loaders",
        "urban_point_cloud_analyzer.data.preprocessing",
        "urban_point_cloud_analyzer.data.augmentation",
        "urban_point_cloud_analyzer.models",
        "urban_point_cloud_analyzer.models.backbones",
        "urban_point_cloud_analyzer.models.segmentation",
        "urban_point_cloud_analyzer.optimization.cuda",
        "urban_point_cloud_analyzer.optimization.quantization",
        "urban_point_cloud_analyzer.utils"
    ]
    
    success = True
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"✓ Module {module_name} imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import {module_name}: {e}")
            success = False
    
    return success

def test_model_creation():
    """Test that models can be created."""
    try:
        from urban_point_cloud_analyzer.models import get_model
        
        config = {
            "type": "pointnet2_seg",
            "backbone": {
                "use_normals": True
            },
            "segmentation": {
                "num_classes": 8,
                "dropout": 0.5
            }
        }
        
        try:
            model = get_model(config)
            print(f"✓ Model created successfully: {type(model).__name__}")
            return True
        except ImportError as e:
            if "PointNet++" in str(e):
                print(f"⚠ Skipping model creation as PointNet++ ops not found")
                print(f"✓ Model creation test skipped")
                return True
            else:
                raise
                
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return False
    
def run_smoke_tests():
    """Run all smoke tests."""
    tests = [
        test_imports,
        test_model_creation,
        # Add more smoke tests here
    ]
    
    results = []
    for test in tests:
        print(f"Running {test.__name__}...")
        result = test()
        results.append(result)
        print("")
    
    success_count = sum(1 for r in results if r)
    total_count = len(results)
    
    print(f"Smoke tests completed: {success_count}/{total_count} successful")
    
    return all(results)

if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)