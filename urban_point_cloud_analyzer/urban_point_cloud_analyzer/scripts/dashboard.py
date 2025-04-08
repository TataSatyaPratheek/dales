# urban_point_cloud_analyzer/scripts/dashboard.py
#!/usr/bin/env python3
"""
Dashboard for Urban Point Cloud Analyzer
"""
import argparse
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from urban_point_cloud_analyzer.ui.dashboard import UrbanDashboard
from urban_point_cloud_analyzer.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Launch Urban Point Cloud Analyzer Dashboard")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing results")
    parser.add_argument("--port", type=int, default=8050, help="Port to run dashboard on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup results directory
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return 1
    
    # Setup logger
    logger = setup_logger(results_dir / "dashboard.log")
    logger.info(f"Starting dashboard with results from: {results_dir}")
    
    # Create and run dashboard
    try:
        dashboard = UrbanDashboard(results_dir)
        logger.info(f"Dashboard initialized. Running on port {args.port}")
        dashboard.run_server(debug=args.debug, port=args.port)
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())