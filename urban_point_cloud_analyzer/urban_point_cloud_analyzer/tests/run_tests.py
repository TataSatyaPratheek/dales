# urban_point_cloud_analyzer/tests/run_tests.py

import os
import sys
import subprocess
import gc
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

def run_test(test_script):
    """Run a test script and return whether it succeeded."""
    print(f"\n=== Running {test_script.name} ===")
    result = subprocess.run([sys.executable, str(test_script)], check=False)
    return result.returncode == 0

def clear_cuda_cache():
    """Clear CUDA cache if PyTorch is available."""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    """Run all tests."""
    # Get the directory containing this script
    script_dir = Path(__file__).resolve().parent

    # Find all test scripts in the directory
    test_scripts = list(script_dir.glob("test_*.py"))

    # Run all tests
    results = []

    for test_script in test_scripts:
        if test_script.exists():
            result = run_test(test_script)
            results.append((test_script.name, result))
        else:
            print(f"Test script {test_script} not found!")
            results.append((test_script.name, False))

        # Clear CUDA cache and perform garbage collection
        clear_cuda_cache()
        gc.collect()

    # Print summary
    print("\n=== Test Results Summary ===")
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{name}: {status}")

    # Return success if all tests passed
    return all(result for _, result in results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)