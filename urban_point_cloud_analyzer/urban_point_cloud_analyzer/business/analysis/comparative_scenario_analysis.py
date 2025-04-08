# urban_point_cloud_analyzer/business/analysis/comparative_scenario_analysis.py
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import os
import json
from pathlib import Path
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64

from urban_point_cloud_analyzer.business.metrics.integrated_metrics import IntegratedUrbanAnalyzer
from urban_point_cloud_analyzer.business.decision_support.decision_support_system import DecisionSupportSystem
from urban_point_cloud_analyzer.business.roi.roi_calculator import ROICalculator
from urban_point_cloud_analyzer.models.detection import detect_objects

class ComparativeScenarioAnalyzer:
    """
    Specialized analyzer for comparing multiple urban development scenarios
    and assessing their relative impacts, costs, and benefits.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize comparative scenario analyzer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize subsystems
        self.metrics_analyzer = IntegratedUrbanAnalyzer(self.config.get('metrics', {}))
        self.decision_support = DecisionSupportSystem(self.config.get('decision_support', {}))
        self.roi_calculator = ROICalculator(self.config.get('roi', {}))
    
    def analyze_scenarios(self, 
                        scenarios: Dict[str, Dict],
                        base_scenario: Optional[str] = None) -> Dict:
        """
        Compare multiple urban development scenarios.
        
        Args:
            scenarios: Dictionary mapping scenario names to scenario data
                      Each scenario should have 'points' and 'labels' keys
            base_scenario: Optional name of the base scenario for comparison
            
        Returns:
            Dictionary with comparative analysis results
        """
        if not scenarios:
            return {"error": "No scenarios provided"}
        
        # Determine base scenario
        if base_scenario is None and scenarios:
            # Use first scenario as base if not specified
            base_scenario = next(iter(scenarios))
        
        # Process each scenario
        scenario_metrics = {}
        scenario_objects = {}
        
        for name, scenario_data in scenarios.items():
            # Skip scenarios without required data
            if 'points' not in scenario_data or 'labels' not in scenario_data:
                continue
            
            # Process scenario
            points = scenario_data['points']
            labels = scenario_data['labels']
            
            # Detect objects
            objects = detect_objects(points, labels, self.config.get('detection', {}))
            scenario_objects[name] = objects
            
            # Calculate metrics
            metrics = self.metrics_analyzer.analyze(points, labels, objects)
            scenario_metrics[name] = metrics
        
        # Compare scenarios against base
        comparisons = {}
        
        for name in scenario_metrics:
            if name == base_scenario:
                continue
            
            # Skip if base scenario doesn't exist
            if base_scenario not in scenario_metrics:
                continue
            
            # Assess impact
            impact = self.decision_support.assess_impact(
                scenario_metrics[base_scenario], scenario_metrics[name]
            )
            
            # Calculate ROI
            # Use cost from scenario data if available, otherwise estimate
            project_cost = scenarios[name].get('project_cost', 1000000)  # Default to $1M
            
            roi_comparison = self.roi_calculator.calculate_comparative_roi(
                scenario_metrics[base_scenario], scenario_metrics[name], project_cost
            )
            
            # Store comparison results
            comparisons[name] = {
                "impact_assessment": impact,
                "roi_comparison": roi_comparison
            }
        
        # Generate multi-scenario comparison visuals
        comparative_visuals = self._generate_comparative_visuals(scenario_metrics)
        
        # Rank scenarios
        scenario_rankings = self._rank_scenarios(scenario_metrics, comparisons)
        
        # Prepare final results
        results = {
            "base_scenario": base_scenario,
            "scenario_metrics": scenario_metrics,
            "scenario_objects": scenario_objects,
            "comparisons": comparisons,
            "rankings": scenario_rankings,
            "comparative_visuals": comparative_visuals
        }
        
        return results
    
    def _rank_scenarios(self, 
                      scenario_metrics: Dict[str, Dict], 
                      comparisons: Dict[str, Dict]) -> Dict:
        """
        Rank scenarios based on multiple criteria.
        
        Args:
            scenario_metrics: Dictionary of metrics for each scenario
            comparisons: Dictionary of comparisons with base scenario
            
        Returns:
            Dictionary with scenario rankings
        """
        if not scenario_metrics:
            return {}
        
        # Rank by urban quality score
        quality_scores = {}
        for name, metrics in scenario_metrics.items():
            quality_scores[name] = metrics.get('urban_quality_score', 0)
        
        # Sort by quality score (descending)
        quality_ranking = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Rank by green coverage
        green_coverage = {}
        for name, metrics in scenario_metrics.items():
            green_coverage[name] = metrics.get('green_coverage', 0)
        
        # Sort by green coverage (descending)
        green_ranking = sorted(green_coverage.items(), key=lambda x: x[1], reverse=True)
        
        # Rank by ROI (for scenarios with comparisons)
        roi_values = {}
        for name, comparison in comparisons.items():
            if 'roi_comparison' in comparison and 'roi_improvement_percent' in comparison['roi_comparison']:
                roi_values[name] = comparison['roi_comparison']['roi_improvement_percent']
        
        # Sort by ROI (descending)
        roi_ranking = sorted(roi_values.items(), key=lambda x: x[1], reverse=True)
        
        # Create multi-criteria score
        # Weight factors can be adjusted in config
        weights = self.config.get('ranking_weights', {
            'quality': 0.4,
            'green': 0.3,
            'roi': 0.3
        })
        
        # Normalize scores to 0-1 scale
        normalized_scores = {}
        
        for name in scenario_metrics:
            score = 0
            
            # Quality score component
            if quality_scores and max(quality_scores.values()) > 0:
                quality_norm = quality_scores.get(name, 0) / max(quality_scores.values())
                score += weights.get('quality', 0.4) * quality_norm
            
            # Green coverage component
            if green_coverage and max(green_coverage.values()) > 0:
                green_norm = green_coverage.get(name, 0) / max(green_coverage.values())
                score += weights.get('green', 0.3) * green_norm
            
            # ROI component (if available)
            if name in roi_values and roi_values and max(roi_values.values()) > 0:
                # Handle negative ROI values
                min_roi = min(0, min(roi_values.values()))
                roi_range = max(roi_values.values()) - min_roi
                
                if roi_range > 0:
                    roi_norm = (roi_values[name] - min_roi) / roi_range
                    score += weights.get('roi', 0.3) * roi_norm
            
            normalized_scores[name] = score
        
        # Sort by overall score (descending)
        overall_ranking = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "by_quality": quality_ranking,
            "by_green_coverage": green_ranking,
            "by_roi": roi_ranking,
            "overall": overall_ranking
        }
    
    def _generate_comparative_visuals(self, scenario_metrics: Dict[str, Dict]) -> Dict:
        """
        Generate comparative visualizations for scenarios.
        
        Args:
            scenario_metrics: Dictionary of metrics for each scenario
            
        Returns:
            Dictionary with base64-encoded visualizations
        """
        if not scenario_metrics:
            return {}
        
        visuals = {}
        
        # Urban quality comparison
        quality_scores = {}
        for name, metrics in scenario_metrics.items():
            quality_scores[name] = metrics.get('urban_quality_score', 0)
        
        if quality_scores:
            visuals["quality_comparison"] = self._create_bar_chart(
                quality_scores, "Urban Quality Score Comparison", "Scenario", "Quality Score (0-100)"
            )
        
        # Land use comparison
        land_use_comparison = {}
        for name, metrics in scenario_metrics.items():
            land_use_comparison[name] = {
                "Building": metrics.get('building_density', 0) * 100,
                "Green Space": metrics.get('green_coverage', 0) * 100,
                "Water": metrics.get('water_coverage', 0) * 100,
                "Other": 100 - (metrics.get('building_density', 0) + metrics.get('green_coverage', 0) + 
                               metrics.get('water_coverage', 0)) * 100
            }
        
        if land_use_comparison:
            visuals["land_use_comparison"] = self._create_stacked_bar_chart(
                land_use_comparison, "Land Use Comparison", "Scenario", "Percentage (%)"
            )
        
        # Environmental metrics radar chart
        if all('environmental' in metrics for metrics in scenario_metrics.values()):
            env_metrics = {}
            for name, metrics in scenario_metrics.items():
                if 'environmental' in metrics:
                    env_metrics[name] = {
                        "Permeable Surface": metrics['environmental'].get('permeable_surface_ratio', 0),
                        "Solar Exposure": metrics['environmental'].get('solar_exposure', 0),
                        "Green-Built Ratio": metrics['environmental'].get('green_to_built_ratio', 0)
                    }
            
            if env_metrics:
                visuals["environmental_comparison"] = self._create_radar_chart(
                    env_metrics, "Environmental Metrics Comparison"
                )
        
        # Accessibility metrics comparison
        if all('accessibility' in metrics for metrics in scenario_metrics.values()):
            access_metrics = {}
            for name, metrics in scenario_metrics.items():
                if 'accessibility' in metrics:
                    access_metrics[name] = {
                        "Building-Road": metrics['accessibility'].get('building_to_road_accessibility', 0),
                        "Green Space": metrics['accessibility'].get('green_space_accessibility', 0)
                    }
            
            if access_metrics:
                visuals["accessibility_comparison"] = self._create_grouped_bar_chart(
                    access_metrics, "Accessibility Comparison", "Metric", "Accessibility Score (0-1)"
                )
        
        return visuals
    
    def _create_bar_chart(self, 
                        data: Dict[str, float], 
                        title: str, 
                        xlabel: str, 
                        ylabel: str) -> str:
        """
        Create a bar chart visualization.
        
        Args:
            data: Dictionary mapping categories to values
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            
        Returns:
            Base64-encoded image
        """
        try:
            plt.figure(figsize=(10, 6))
            
            categories = list(data.keys())
            values = list(data.values())
            
            # Create bar chart
            bars = plt.bar(categories, values, color='#4285f4')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.1f}', ha='center', va='bottom')
            
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            return image_base64
            
        except Exception as e:
            # Return error message if visualization fails
            return f"Error generating bar chart: {str(e)}"
    
    def _create_stacked_bar_chart(self, 
                                data: Dict[str, Dict[str, float]], 
                                title: str, 
                                xlabel: str, 
                                ylabel: str) -> str:
        """
        Create a stacked bar chart visualization.
        
        Args:
            data: Dictionary mapping scenarios to dictionaries of category values
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            
        Returns:
            Base64-encoded image
        """
        try:
            plt.figure(figsize=(12, 7))
            
            scenarios = list(data.keys())
            categories = list(data[scenarios[0]].keys())
            
            # Create bottom coordinate for stacking
            bottoms = np.zeros(len(scenarios))
            
            # Define colors for categories
            colors = ['#996633', '#33CC33', '#3333CC', '#AAAAAA']
            
            # Create stacked bars for each category
            for i, category in enumerate(categories):
                values = [data[scenario][category] for scenario in scenarios]
                plt.bar(scenarios, values, bottom=bottoms, label=category, color=colors[i % len(colors)])
                bottoms += values
            
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            return image_base64
            
        except Exception as e:
            # Return error message if visualization fails
            return f"Error generating stacked bar chart: {str(e)}"
    
    def _create_radar_chart(self, 
                          data: Dict[str, Dict[str, float]], 
                          title: str) -> str:
        """
        Create a radar chart visualization.
        
        Args:
            data: Dictionary mapping scenarios to dictionaries of metric values
            title: Chart title
            
        Returns:
            Base64-encoded image
        """
        try:
            plt.figure(figsize=(10, 8))
            
            # Get scenarios and metrics
            scenarios = list(data.keys())
            metrics = list(data[scenarios[0]].keys())
            
            # Number of metrics
            N = len(metrics)
            
            # Angle of each axis in the plot (divide the plot into equal parts)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Initialize the radar chart
            ax = plt.subplot(111, polar=True)
            
            # Draw one axis per metric and add labels
            plt.xticks(angles[:-1], metrics, size=10)
            
            # Draw ylabels
            ax.set_rlabel_position(0)
            plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=8)
            plt.ylim(0, 1)
            
            # Plot each scenario
            colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853', '#673ab7', '#ff6d00']
            for i, scenario in enumerate(scenarios):
                # Get values for this scenario
                values = [data[scenario][metric] for metric in metrics]
                values += values[:1]  # Close the loop
                
                # Plot values
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=scenario, color=colors[i % len(colors)])
                ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)
            
            # Add title and legend
            plt.title(title, size=15, y=1.1)
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            # Convert plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            return image_base64
            
        except Exception as e:
            # Return error message if visualization fails
            return f"Error generating radar chart: {str(e)}"
    
    def _create_grouped_bar_chart(self, 
                                data: Dict[str, Dict[str, float]], 
                                title: str, 
                                xlabel: str, 
                                ylabel: str) -> str:
        """
        Create a grouped bar chart visualization.
        
        Args:
            data: Dictionary mapping scenarios to dictionaries of metric values
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            
        Returns:
            Base64-encoded image
        """
        try:
            plt.figure(figsize=(12, 7))
            
            # Get scenarios and metrics
            scenarios = list(data.keys())
            metrics = list(data[scenarios[0]].keys())
            
            # Number of scenarios
            n_scenarios = len(scenarios)
            
            # Width of bars
            bar_width = 0.8 / n_scenarios
            
            # Position of bars on x-axis
            r = np.arange(len(metrics))
            
            # Define colors for scenarios
            colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853', '#673ab7', '#ff6d00']
            
            # Create grouped bars for each scenario
            for i, scenario in enumerate(scenarios):
                values = [data[scenario][metric] for metric in metrics]
                position = [x + bar_width * i for x in r]
                bars = plt.bar(position, values, width=bar_width, label=scenario, color=colors[i % len(colors)])
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
            
            # Add labels and legend
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.xticks([r + bar_width * (n_scenarios-1) / 2 for r in range(len(metrics))], metrics)
            plt.legend()
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            return image_base64
            
        except Exception as e:
            # Return error message if visualization fails
            return f"Error generating grouped bar chart: {str(e)}"
    
    def get_optimal_scenario(self, analysis_results: Dict, criteria: Optional[str] = None) -> Dict:
        """
        Get the optimal scenario based on specified criteria.
        
        Args:
            analysis_results: Results from analyze_scenarios method
            criteria: Optional criteria for selection ('quality', 'green', 'roi', 'overall')
            
        Returns:
            Dictionary with optimal scenario information
        """
        if 'rankings' not in analysis_results:
            return {"error": "No rankings available in analysis results"}
        
        rankings = analysis_results.get('rankings', {})
        
        # Use specified criteria or default to overall
        if criteria is None or criteria not in rankings:
            criteria = 'overall'
        
        # Get ranking for the specified criteria
        ranking = rankings.get(criteria, [])
        
        if not ranking:
            return {"error": f"No ranking available for criteria: {criteria}"}
        
        # Get top-ranked scenario
        top_scenario = ranking[0][0]
        
        # Get metrics and comparison for the top scenario
        scenario_metrics = analysis_results.get('scenario_metrics', {})
        comparisons = analysis_results.get('comparisons', {})
        
        # Prepare results
        result = {
            "criteria": criteria,
            "optimal_scenario": top_scenario,
            "ranking": ranking,
            "metrics": scenario_metrics.get(top_scenario, {})
        }
        
        # Add comparison with base scenario if available
        base_scenario = analysis_results.get('base_scenario')
        if base_scenario and base_scenario != top_scenario and top_scenario in comparisons:
            result["comparison_with_base"] = comparisons[top_scenario]
        
        return result