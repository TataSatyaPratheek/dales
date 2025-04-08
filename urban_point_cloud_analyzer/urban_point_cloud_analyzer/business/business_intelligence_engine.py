# urban_point_cloud_analyzer/business/business_intelligence_engine.py
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import os
import json
from pathlib import Path
import torch
from datetime import datetime

from urban_point_cloud_analyzer.business.metrics.urban_metrics import calculate_urban_metrics
from urban_point_cloud_analyzer.business.metrics.advanced_metrics import (
    calculate_road_connectivity,
    calculate_accessibility_metrics,
    calculate_urban_density_metrics,
    calculate_environmental_metrics
)
from urban_point_cloud_analyzer.business.metrics.integrated_metrics import IntegratedUrbanAnalyzer
from urban_point_cloud_analyzer.business.decision_support.decision_support_system import DecisionSupportSystem
from urban_point_cloud_analyzer.business.roi.roi_calculator import ROICalculator
from urban_point_cloud_analyzer.business.reports.reporting_system import ComprehensiveReportingSystem
from urban_point_cloud_analyzer.models.detection import detect_objects


class BusinessIntelligenceEngine:
    """
    Comprehensive business intelligence engine that integrates all BI components
    for urban planning applications. Provides a unified interface for metrics
    calculation, decision support, ROI calculation, and reporting.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the business intelligence engine.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize all subsystems
        self.metrics_analyzer = IntegratedUrbanAnalyzer(self.config.get('metrics', {}))
        self.decision_support = DecisionSupportSystem(self.config.get('decision_support', {}))
        self.roi_calculator = ROICalculator(self.config.get('roi', {}))
        self.reporting_system = ComprehensiveReportingSystem(self.config.get('reporting', {}))
        
        # Create an output directory for saving results
        self.output_dir = Path(self.config.get('output_dir', 'results'))
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize empty storage for historical data
        self.historical_data = {}

    def analyze_area(self, 
                   points: np.ndarray, 
                   labels: np.ndarray, 
                   area_id: Optional[str] = None) -> Dict:
        """
        Perform comprehensive analysis of an urban area.
        
        Args:
            points: (N, 3+) array of point coordinates
            labels: (N,) array of segmentation labels
            area_id: Optional identifier for the area (for tracking over time)
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        # 1. Detect objects in the point cloud
        objects = detect_objects(points, labels, self.config.get('detection', {}))
        
        # 2. Calculate all urban metrics
        metrics = self.metrics_analyzer.analyze(points, labels, objects)
        
        # 3. Generate decision support insights
        compliance = self.decision_support.evaluate_compliance(metrics)
        recommendations = self.decision_support.generate_recommendations(metrics)
        improvement_areas = self.decision_support.identify_improvement_areas(metrics)
        
        # 4. Calculate return on investment
        roi_analysis = self.roi_calculator.calculate_roi(metrics)
        
        # 5. Combine all results
        results = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "objects": objects,
            "compliance": compliance,
            "recommendations": recommendations,
            "improvement_areas": improvement_areas,
            "roi_analysis": roi_analysis
        }
        
        # 6. Store results in historical data if area_id is provided
        if area_id:
            if area_id not in self.historical_data:
                self.historical_data[area_id] = []
            self.historical_data[area_id].append(results)
        
        return results
    
    def compare_scenarios(self, 
                        current_points: np.ndarray, 
                        current_labels: np.ndarray,
                        proposed_points: np.ndarray, 
                        proposed_labels: np.ndarray,
                        project_cost: Optional[float] = None) -> Dict:
        """
        Compare current and proposed urban development scenarios.
        
        Args:
            current_points: (N, 3+) array of current point coordinates
            current_labels: (N,) array of current segmentation labels
            proposed_points: (M, 3+) array of proposed point coordinates
            proposed_labels: (M,) array of proposed segmentation labels
            project_cost: Optional project cost for ROI calculation
            
        Returns:
            Dictionary with comparison results
        """
        # 1. Analyze current scenario
        current_objects = detect_objects(current_points, current_labels, self.config.get('detection', {}))
        current_metrics = self.metrics_analyzer.analyze(current_points, current_labels, current_objects)
        
        # 2. Analyze proposed scenario
        proposed_objects = detect_objects(proposed_points, proposed_labels, self.config.get('detection', {}))
        proposed_metrics = self.metrics_analyzer.analyze(proposed_points, proposed_labels, proposed_objects)
        
        # 3. Assess impact of changes
        impact = self.decision_support.assess_impact(current_metrics, proposed_metrics)
        
        # 4. Calculate ROI for the proposed changes
        if project_cost:
            roi_comparison = self.roi_calculator.calculate_comparative_roi(
                current_metrics, proposed_metrics, project_cost
            )
        else:
            # Estimate project cost if not provided
            estimated_cost = self._estimate_project_cost(current_metrics, proposed_metrics)
            roi_comparison = self.roi_calculator.calculate_comparative_roi(
                current_metrics, proposed_metrics, estimated_cost
            )
        
        # 5. Generate resource allocation if project costs provided
        if project_cost:
            resource_allocation = self.decision_support.optimize_resources(proposed_metrics, project_cost)
        else:
            resource_allocation = None
        
        # 6. Combine all results
        comparison_results = {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": current_metrics,
            "proposed_metrics": proposed_metrics,
            "current_objects": current_objects,
            "proposed_objects": proposed_objects,
            "impact_assessment": impact,
            "roi_comparison": roi_comparison,
            "resource_allocation": resource_allocation
        }
        
        return comparison_results
    
    def analyze_over_time(self, area_id: str) -> Dict:
        """
        Analyze how an urban area has changed over time.
        
        Args:
            area_id: Identifier for the area
            
        Returns:
            Dictionary with temporal analysis results
        """
        # Check if we have historical data for this area
        if area_id not in self.historical_data or len(self.historical_data[area_id]) < 2:
            return {"error": "Insufficient historical data for temporal analysis"}
        
        # Get historical data points
        data_points = self.historical_data[area_id]
        data_points.sort(key=lambda x: x["timestamp"])
        
        # Extract key metrics for all time points
        timestamps = []
        green_coverage = []
        building_density = []
        quality_scores = []
        roi_values = []
        
        for data_point in data_points:
            timestamps.append(data_point["timestamp"])
            metrics = data_point["metrics"]
            green_coverage.append(metrics.get("green_coverage", 0))
            building_density.append(metrics.get("building_density", 0))
            quality_scores.append(metrics.get("urban_quality_score", 0))
            
            if "roi_analysis" in data_point:
                roi_values.append(data_point["roi_analysis"].get("roi_percent", 0))
            else:
                roi_values.append(0)
        
        # Calculate trends
        if len(timestamps) >= 3:
            green_trend = self._calculate_trend(green_coverage)
            density_trend = self._calculate_trend(building_density)
            quality_trend = self._calculate_trend(quality_scores)
            roi_trend = self._calculate_trend(roi_values)
        else:
            # Simple trend with just two points
            green_trend = green_coverage[-1] - green_coverage[0]
            density_trend = building_density[-1] - building_density[0]
            quality_trend = quality_scores[-1] - quality_scores[0]
            roi_trend = roi_values[-1] - roi_values[0]
        
        # Compare first and last data points
        first_data = data_points[0]
        last_data = data_points[-1]
        first_metrics = first_data["metrics"]
        last_metrics = last_data["metrics"]
        
        # Calculate impact of changes over time
        impact_over_time = self.decision_support.assess_impact(first_metrics, last_metrics)
        
        # Assemble results
        temporal_analysis = {
            "area_id": area_id,
            "time_period": {
                "start": timestamps[0],
                "end": timestamps[-1],
                "num_data_points": len(timestamps)
            },
            "metrics_over_time": {
                "timestamps": timestamps,
                "green_coverage": green_coverage,
                "building_density": building_density,
                "quality_scores": quality_scores,
                "roi_values": roi_values
            },
            "trends": {
                "green_coverage_trend": green_trend,
                "building_density_trend": density_trend,
                "quality_score_trend": quality_trend,
                "roi_trend": roi_trend
            },
            "impact_over_time": impact_over_time
        }
        
        return temporal_analysis
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate a simple linear trend from a list of values."""
        if len(values) < 2:
            return 0
        
        # Use numpy's polyfit for linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        try:
            slope, _ = np.polyfit(x, y, 1)
            return slope
        except:
            # Fallback to simple start-end difference if polyfit fails
            return values[-1] - values[0]
    
    def _estimate_project_cost(self, current_metrics: Dict, proposed_metrics: Dict) -> float:
        """
        Estimate project cost based on the scale of changes between current and proposed metrics.
        
        Args:
            current_metrics: Current urban metrics
            proposed_metrics: Proposed urban metrics
            
        Returns:
            Estimated project cost
        """
        # Get area sizes
        current_area = current_metrics.get('total_area_m2', 10000)  # Default to 1 hectare
        
        # Estimate costs based on area and changes in key metrics
        # These are very simplified estimates
        base_cost_per_m2 = 200  # $200 per square meter as base cost
        
        # Calculate changes in building area
        current_building_area = current_metrics.get('building_area_m2', 0)
        proposed_building_area = proposed_metrics.get('building_area_m2', 0)
        building_area_change = abs(proposed_building_area - current_building_area)
        
        # Calculate changes in green area
        current_green_area = current_metrics.get('vegetation_area_m2', 0)
        proposed_green_area = proposed_metrics.get('vegetation_area_m2', 0)
        green_area_change = abs(proposed_green_area - current_green_area)
        
        # Construction costs
        building_cost = building_area_change * 1500  # $1500 per m² for building construction
        green_space_cost = green_area_change * 100   # $100 per m² for green space development
        
        # Add infrastructure costs (roads, utilities)
        infrastructure_share = 0.3  # Infrastructure is typically 30% of project cost
        infrastructure_cost = (building_cost + green_space_cost) * infrastructure_share / (1 - infrastructure_share)
        
        # Total estimated cost
        total_cost = building_cost + green_space_cost + infrastructure_cost
        
        # Ensure a minimum reasonable cost
        min_cost = current_area * 50  # At least $50 per m² of the total area
        return max(total_cost, min_cost)
    
    def generate_comprehensive_report(self, 
                                    results: Dict, 
                                    report_type: str = 'standard',
                                    output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive report from analysis results.
        
        Args:
            results: Analysis results dict from analyze_area or compare_scenarios
            report_type: Type of report ('standard', 'comparison', 'compliance', 'project_impact')
            output_file: Optional path to save the report
            
        Returns:
            Path to the saved report
        """
        # Generate file path if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"{report_type}_report_{timestamp}.html"
        
        # Generate report based on type
        if report_type == 'standard':
            metrics = results.get('metrics', {})
            return self.reporting_system.create_single_page_report(
                metrics, str(output_file), "Urban Analysis Report"
            )
        
        elif report_type == 'comparison':
            current_metrics = results.get('current_metrics', {})
            proposed_metrics = results.get('proposed_metrics', {})
            return self.reporting_system.create_comparison_report(
                current_metrics, proposed_metrics, str(output_file), "Urban Development Comparison"
            )
        
        elif report_type == 'compliance':
            metrics = results.get('metrics', results.get('proposed_metrics', {}))
            regulation_type = results.get('regulation_type', 'standard')
            return self.reporting_system.create_compliance_report(
                metrics, regulation_type, str(output_file)
            )
        
        elif report_type == 'project_impact':
            current_metrics = results.get('current_metrics', {})
            proposed_metrics = results.get('proposed_metrics', {})
            
            # Get costs
            if 'roi_comparison' in results and 'project_cost' in results['roi_comparison']:
                costs = {'total_project_cost': results['roi_comparison']['project_cost']}
            else:
                costs = {'total_project_cost': 1000000}  # Default $1M if not available
            
            return self.reporting_system.create_project_impact_report(
                current_metrics, proposed_metrics, costs, str(output_file)
            )
        
        else:
            raise ValueError(f"Unknown report type: {report_type}")
    
    def save_results(self, results: Dict, filename: Optional[str] = None) -> str:
        """
        Save analysis results to file.
        
        Args:
            results: Analysis results dictionary
            filename: Optional filename
            
        Returns:
            Path to saved file
        """
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_results_{timestamp}.json"
        
        file_path = self.output_dir / filename
        
        # Convert numpy values to Python types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_for_json(results)
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        return str(file_path)
    
    def load_results(self, filepath: str) -> Dict:
        """
        Load analysis results from file.
        
        Args:
            filepath: Path to results file
            
        Returns:
            Analysis results dictionary
        """
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        return results
    
    def get_optimization_recommendations(self, metrics: Dict, budget: Optional[float] = None) -> Dict:
        """
        Get resource optimization recommendations based on urban metrics.
        
        Args:
            metrics: Urban metrics dictionary
            budget: Optional budget constraint
            
        Returns:
            Dictionary with optimization recommendations
        """
        # Get improvement areas
        improvement_areas = self.decision_support.identify_improvement_areas(metrics)
        
        # If budget is provided, optimize resource allocation
        if budget:
            resource_allocation = self.decision_support.optimize_resources(metrics, budget)
        else:
            resource_allocation = None
        
        # Generate decision report
        decision_report = self.decision_support.generate_decision_report(metrics, budget)
        
        return {
            "improvement_areas": improvement_areas,
            "resource_allocation": resource_allocation,
            "decision_report": decision_report
        }