# urban_point_cloud_analyzer/business/metrics/integrated_metrics.py
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
import json
from pathlib import Path

from urban_point_cloud_analyzer.business.metrics.urban_metrics import calculate_urban_metrics
from urban_point_cloud_analyzer.business.metrics.advanced_metrics import (
    calculate_road_connectivity,
    calculate_accessibility_metrics,
    calculate_urban_density_metrics,
    calculate_environmental_metrics
)

class IntegratedUrbanAnalyzer:
    """
    Comprehensive urban analysis integrating multiple metrics.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize urban analyzer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config if config is not None else {}
    
    def analyze(self, 
            points: np.ndarray, 
            labels: np.ndarray, 
            detected_objects: Optional[List[Dict]] = None) -> Dict:
        """
        Perform comprehensive urban analysis with safeguards against numerical issues.
        
        Args:
            points: (N, 3+) array of point coordinates
            labels: (N,) array of point labels
            detected_objects: Optional list of detected objects
            
        Returns:
            Dictionary of all urban metrics
        """
        # Add error handling for empty arrays and division by zero
        try:
            # Calculate basic urban metrics
            basic_metrics = calculate_urban_metrics(points, labels)
        except Exception as e:
            print(f"Warning: Error calculating basic metrics: {e}")
            basic_metrics = {}
        
        try:
            # Calculate road connectivity metrics
            road_metrics = calculate_road_connectivity(points, labels)
        except Exception as e:
            print(f"Warning: Error calculating road metrics: {e}")
            road_metrics = {}
        
        try:
            # Calculate accessibility metrics
            accessibility_metrics = calculate_accessibility_metrics(points, labels)
        except Exception as e:
            print(f"Warning: Error calculating accessibility metrics: {e}")
            accessibility_metrics = {}
        
        try:
            # Calculate urban density metrics
            density_metrics = calculate_urban_density_metrics(points, labels)
        except Exception as e:
            print(f"Warning: Error calculating density metrics: {e}")
            density_metrics = {}
        
        try:
            # Calculate environmental metrics
            environmental_metrics = calculate_environmental_metrics(points, labels)
        except Exception as e:
            print(f"Warning: Error calculating environmental metrics: {e}")
            environmental_metrics = {}
        
        # Combine all metrics
        all_metrics = {
            **basic_metrics,
            'road': road_metrics,
            'accessibility': accessibility_metrics,
            'density': density_metrics,
            'environmental': environmental_metrics
        }
        
        # Add object detection metrics if available
        if detected_objects:
            try:
                object_metrics = self._calculate_object_metrics(detected_objects)
                all_metrics['objects'] = object_metrics
            except Exception as e:
                print(f"Warning: Error calculating object metrics: {e}")
        
        # Calculate urban quality score with error handling
        try:
            all_metrics['urban_quality_score'] = self._calculate_urban_quality_score(all_metrics)
        except Exception as e:
            print(f"Warning: Error calculating urban quality score: {e}")
            all_metrics['urban_quality_score'] = 0.0
        
        return all_metrics
    
    def _calculate_object_metrics(self, objects: List[Dict]) -> Dict:
        """
        Calculate metrics based on detected objects.
        
        Args:
            objects: List of detected objects
            
        Returns:
            Dictionary of object-based metrics
        """
        # Count objects by class
        object_counts = {}
        for obj in objects:
            class_name = obj['class_name']
            if class_name not in object_counts:
                object_counts[class_name] = 0
            object_counts[class_name] += 1
        
        # Calculate vehicle metrics
        car_count = object_counts.get('Car', 0)
        truck_count = object_counts.get('Truck', 0)
        total_vehicles = car_count + truck_count
        
        # Calculate infrastructure metrics
        powerline_count = object_counts.get('Powerline', 0)
        fence_count = object_counts.get('Fence', 0)
        
        # Summarize object metrics
        object_metrics = {
            'counts': object_counts,
            'total_vehicles': total_vehicles,
            'car_to_truck_ratio': car_count / (truck_count + 1e-10),
            'infrastructure_count': powerline_count + fence_count
        }
        
        return object_metrics
    
    def _calculate_urban_quality_score(self, metrics: Dict) -> float:
        """
        Calculate overall urban quality score.
        
        Args:
            metrics: Dictionary of urban metrics
            
        Returns:
            Urban quality score (0-100)
        """
        # Define weights for different metrics
        weights = {
            'green_coverage': 20,
            'accessibility': {
                'building_to_road_accessibility': 15,
                'green_space_accessibility': 15
            },
            'road': {
                'connectivity_score': 10
            },
            'density': {
                'urban_compactness': 10
            },
            'environmental': {
                'permeable_surface_ratio': 15,
                'solar_exposure': 15
            }
        }
        
        # Calculate weighted score
        score = 0.0
        
        # Basic metrics
        score += metrics.get('green_coverage', 0) * weights['green_coverage'] * 100
        
        # Accessibility metrics
        if 'accessibility' in metrics:
            access = metrics['accessibility']
            score += access.get('building_to_road_accessibility', 0) * weights['accessibility']['building_to_road_accessibility']
            score += access.get('green_space_accessibility', 0) * weights['accessibility']['green_space_accessibility']
        
        # Road metrics
        if 'road' in metrics:
            road = metrics['road']
            score += road.get('connectivity_score', 0) * weights['road']['connectivity_score']
        
        # Density metrics
        if 'density' in metrics:
            density = metrics['density']
            # Urban compactness (normalized to 0-1)
            compactness = min(1.0, density.get('urban_compactness', 0) / 0.5)
            score += compactness * weights['density']['urban_compactness']
        
        # Environmental metrics
        if 'environmental' in metrics:
            env = metrics['environmental']
            score += env.get('permeable_surface_ratio', 0) * weights['environmental']['permeable_surface_ratio'] * 100
            score += env.get('solar_exposure', 0) * weights['environmental']['solar_exposure']
        
        # Cap score at 100
        score = min(100.0, score)
        
        return score
    
    def generate_report(self, metrics: Dict, output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive urban analysis report.
        
        Args:
            metrics: Dictionary of urban metrics
            output_file: Optional path to save report
            
        Returns:
            Report as string
        """
        # Start report
        report = "=======================================\n"
        report += "COMPREHENSIVE URBAN ANALYSIS REPORT\n"
        report += "=======================================\n\n"
        
        # Overall quality score
        quality_score = metrics.get('urban_quality_score', 0.0)
        report += f"URBAN QUALITY SCORE: {quality_score:.1f}/100\n\n"
        
        # Basic metrics
        report += "BASIC URBAN METRICS\n"
        report += "-----------------\n"
        report += f"Total Area: {metrics.get('total_area_m2', 0):.2f} m²\n"
        report += f"Building Area: {metrics.get('building_area_m2', 0):.2f} m² ({metrics.get('building_density', 0)*100:.2f}%)\n"
        report += f"Vegetation Area: {metrics.get('vegetation_area_m2', 0):.2f} m² ({metrics.get('green_coverage', 0)*100:.2f}%)\n"
        report += f"Water Area: {metrics.get('water_area_m2', 0):.2f} m² ({metrics.get('water_coverage', 0)*100:.2f}%)\n\n"
        
        # Road network metrics
        if 'road' in metrics:
            road = metrics['road']
            report += "ROAD NETWORK METRICS\n"
            report += "-------------------\n"
            report += f"Connectivity Score: {road.get('connectivity_score', 0):.2f}\n"
            report += f"Average Road Width: {road.get('avg_road_width', 0):.2f} m\n"
            report += f"Road Density: {road.get('road_density', 0)*100:.2f}%\n"
            report += f"Intersection Count: {road.get('intersection_count', 0)}\n\n"
        
        # Accessibility metrics
        if 'accessibility' in metrics:
            access = metrics['accessibility']
            report += "ACCESSIBILITY METRICS\n"
            report += "--------------------\n"
            report += f"Building to Road Accessibility: {access.get('building_to_road_accessibility', 0):.2f}\n"
            report += f"Green Space Accessibility: {access.get('green_space_accessibility', 0):.2f}\n"
            report += f"Average Distance to Road: {access.get('avg_distance_to_road', 0):.2f} m\n"
            report += f"Average Distance to Green Space: {access.get('avg_distance_to_green', 0):.2f} m\n\n"
        
        # Urban density metrics
        if 'density' in metrics:
            density = metrics['density']
            report += "URBAN DENSITY METRICS\n"
            report += "--------------------\n"
            report += f"Floor Area Ratio (FAR): {density.get('floor_area_ratio', 0):.2f}\n"
            report += f"Building Coverage Ratio: {density.get('building_coverage_ratio', 0):.2f}\n"
            report += f"Building Height Variation: {density.get('building_height_variation', 0):.2f}\n"
            report += f"Urban Compactness: {density.get('urban_compactness', 0):.2f}\n\n"
        
        # Environmental metrics
        if 'environmental' in metrics:
            env = metrics['environmental']
            report += "ENVIRONMENTAL METRICS\n"
            report += "--------------------\n"
            report += f"Permeable Surface Ratio: {env.get('permeable_surface_ratio', 0):.2f}\n"
            report += f"Green Space Fragmentation: {env.get('green_space_fragmentation', 0):.2f}\n"
            report += f"Solar Exposure: {env.get('solar_exposure', 0):.2f}\n"
            report += f"Green to Built Ratio: {env.get('green_to_built_ratio', 0):.2f}\n\n"
        
        # Object metrics
        if 'objects' in metrics:
            obj = metrics['objects']
            report += "OBJECT METRICS\n"
            report += "-------------\n"
            report += f"Total Vehicles: {obj.get('total_vehicles', 0)}\n"
            report += f"Car to Truck Ratio: {obj.get('car_to_truck_ratio', 0):.2f}\n"
            report += f"Infrastructure Count: {obj.get('infrastructure_count', 0)}\n"
            
            # Object counts
            if 'counts' in obj:
                report += "\nObject Counts:\n"
                for class_name, count in obj['counts'].items():
                    report += f"  - {class_name}: {count}\n"
            report += "\n"
        
        # Urban planning recommendations
        report += "URBAN PLANNING RECOMMENDATIONS\n"
        report += "------------------------------\n"
        
        # Green space recommendations
        if metrics.get('green_coverage', 0) < 0.15:
            report += "⚠️  Green space coverage is below recommended levels (15%).\n"
            report += "    → Increase vegetation in the area.\n"
            report += "    → Consider adding parks or green roofs.\n"
        else:
            report += "✓  Green space coverage meets recommended levels.\n"
        
        # Building density recommendations
        if metrics.get('building_density', 0) > 0.7:
            report += "⚠️  Building density is very high.\n"
            report += "    → Consider adding more open spaces.\n"
            report += "    → Improve building spacing for better ventilation and lighting.\n"
        
        # Road connectivity recommendations
        if 'road' in metrics and metrics['road'].get('connectivity_score', 0) < 0.4:
            report += "⚠️  Road connectivity is suboptimal.\n"
            report += "    → Improve street network connectivity.\n"
            report += "    → Consider adding alternative routes to reduce congestion.\n"
        
        # Accessibility recommendations
        if 'accessibility' in metrics and metrics['accessibility'].get('green_space_accessibility', 0) < 0.4:
            report += "⚠️  Green space accessibility is low.\n"
            report += "    → Distribute green spaces more evenly throughout the area.\n"
            report += "    → Ensure all residential buildings have access to green space within 300m.\n"
        
        # Permeable surface recommendations
        if 'environmental' in metrics and metrics['environmental'].get('permeable_surface_ratio', 0) < 0.3:
            report += "⚠️  Permeable surface ratio is low.\n"
            report += "    → Increase permeable surfaces to improve stormwater management.\n"
            report += "    → Consider permeable pavements and rain gardens.\n"
        
        # Save report if output file is provided
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report