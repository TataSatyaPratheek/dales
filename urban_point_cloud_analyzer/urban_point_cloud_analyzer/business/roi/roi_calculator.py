# urban_point_cloud_analyzer/business/roi/roi_calculator.py
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

class ROICalculator:
    """
    Calculate Return on Investment for urban planning projects.
    This provides practical financial metrics for decision-makers based on LiDAR analysis.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ROI calculator with optional configuration.
        
        Args:
            config: Configuration dictionary with financial parameters
        """
        self.config = config or {}
        
        # Default financial parameters
        self.discount_rate = self.config.get('discount_rate', 0.05)  # 5% discount rate
        self.analysis_period = self.config.get('analysis_period', 20)  # 20 year analysis
        self.maintenance_cost_per_m2 = self.config.get('maintenance_cost_per_m2', {
            'building': 25.0,  # $25/m² annual maintenance
            'roads': 15.0,     # $15/m² annual road maintenance
            'vegetation': 5.0,  # $5/m² annual vegetation maintenance
            'water': 2.0       # $2/m² annual water feature maintenance
        })
        
        # Default valuation parameters
        self.property_value_per_m2 = self.config.get('property_value_per_m2', 2000)  # $2000/m²
        self.green_space_value_multiplier = self.config.get('green_space_value_multiplier', 1.15)  # 15% value increase
        self.water_feature_value_multiplier = self.config.get('water_feature_value_multiplier', 1.2)  # 20% value increase
    
    def calculate_maintenance_costs(self, metrics: Dict) -> Dict:
        """
        Calculate annual and total maintenance costs based on area metrics.
        
        Args:
            metrics: Dictionary containing urban area metrics
            
        Returns:
            Dictionary with maintenance cost calculations
        """
        # Extract area metrics
        building_area = metrics.get('building_area_m2', 0)
        vegetation_area = metrics.get('vegetation_area_m2', 0)
        water_area = metrics.get('water_area_m2', 0)
        total_area = metrics.get('total_area_m2', 0)
        
        # Estimate road area if not directly provided
        if 'road_area_m2' in metrics:
            road_area = metrics['road_area_m2']
        else:
            # Estimate road area as a percentage of total area minus building, vegetation, and water
            remaining_area = total_area - building_area - vegetation_area - water_area
            road_area = remaining_area * 0.6  # Assume 60% of remaining area is roads
        
        # Calculate annual maintenance costs
        annual_costs = {
            'building': building_area * self.maintenance_cost_per_m2['building'],
            'roads': road_area * self.maintenance_cost_per_m2['roads'],
            'vegetation': vegetation_area * self.maintenance_cost_per_m2['vegetation'],
            'water': water_area * self.maintenance_cost_per_m2['water']
        }
        
        # Calculate total annual cost
        total_annual_cost = sum(annual_costs.values())
        
        # Calculate present value of maintenance over analysis period
        present_value = 0
        for year in range(1, self.analysis_period + 1):
            present_value += total_annual_cost / ((1 + self.discount_rate) ** year)
        
        return {
            'annual_maintenance_costs': annual_costs,
            'total_annual_maintenance_cost': total_annual_cost,
            'present_value_of_maintenance': present_value
        }
    
    def calculate_property_value(self, metrics: Dict) -> Dict:
        """
        Calculate property value based on area metrics.
        
        Args:
            metrics: Dictionary containing urban area metrics
            
        Returns:
            Dictionary with property value calculations
        """
        # Extract area metrics
        building_area = metrics.get('building_area_m2', 0)
        total_area = metrics.get('total_area_m2', 0)
        
        # Calculate base property value
        base_property_value = building_area * self.property_value_per_m2
        
        # Apply multipliers based on green space and water features
        green_coverage = metrics.get('green_coverage', 0)
        water_coverage = metrics.get('water_coverage', 0)
        
        # Calculate green space quality impact
        if 'environmental' in metrics and 'green_space_fragmentation' in metrics['environmental']:
            # Adjust multiplier based on fragmentation (lower is better)
            fragmentation = metrics['environmental']['green_space_fragmentation']
            adjusted_green_multiplier = self.green_space_value_multiplier
            if fragmentation > 1.0:
                adjusted_green_multiplier = max(1.05, self.green_space_value_multiplier - (fragmentation - 1.0) * 0.05)
        else:
            adjusted_green_multiplier = self.green_space_value_multiplier
        
        # Calculate value multipliers
        green_space_multiplier = 1.0 + (adjusted_green_multiplier - 1.0) * green_coverage
        water_feature_multiplier = 1.0 + (self.water_feature_value_multiplier - 1.0) * water_coverage
        
        # Apply multipliers
        adjusted_property_value = base_property_value * green_space_multiplier * water_feature_multiplier
        
        # Calculate value per m²
        value_per_m2 = adjusted_property_value / building_area if building_area > 0 else 0
        
        return {
            'base_property_value': base_property_value,
            'green_space_multiplier': green_space_multiplier,
            'water_feature_multiplier': water_feature_multiplier,
            'adjusted_property_value': adjusted_property_value,
            'value_per_m2': value_per_m2
        }
    
    def calculate_infrastructure_costs(self, metrics: Dict) -> Dict:
        """
        Calculate infrastructure costs and benefits.
        
        Args:
            metrics: Dictionary containing urban area metrics
            
        Returns:
            Dictionary with infrastructure cost calculations
        """
        # Extract metrics
        total_area = metrics.get('total_area_m2', 0)
        building_density = metrics.get('building_density', 0)
        
        # Extract road metrics if available
        road_metrics = metrics.get('road', {})
        road_connectivity = road_metrics.get('connectivity_score', 0.5)
        avg_road_width = road_metrics.get('avg_road_width', 7.0)  # Default 7m if not provided
        
        # Calculate infrastructure costs
        # More connected road networks are typically more cost-effective
        road_cost_multiplier = 1.0 + (0.5 - road_connectivity) * 2  # Lower connectivity = higher costs
        
        # Typical road construction costs per m²
        base_road_cost_per_m2 = 300  # $300/m²
        road_cost_per_m2 = base_road_cost_per_m2 * road_cost_multiplier
        
        # Estimate road area
        if 'road_area_m2' in metrics:
            road_area = metrics['road_area_m2']
        else:
            # Estimate from total area and building density
            remaining_area = total_area * (1 - building_density)
            road_area = remaining_area * 0.5  # Assume half of remaining space is roads
        
        # Calculate total road infrastructure cost
        road_infrastructure_cost = road_area * road_cost_per_m2
        
        # Estimate utility costs (water, sewer, electricity)
        # These are typically proportional to road length
        utility_cost_per_m2 = 150  # $150/m²
        utility_infrastructure_cost = road_area * utility_cost_per_m2
        
        # Calculate total infrastructure cost
        total_infrastructure_cost = road_infrastructure_cost + utility_infrastructure_cost
        
        # Calculate infrastructure cost per building area
        building_area = metrics.get('building_area_m2', 0)
        if building_area > 0:
            infrastructure_cost_per_building_m2 = total_infrastructure_cost / building_area
        else:
            infrastructure_cost_per_building_m2 = 0
        
        return {
            'road_infrastructure_cost': road_infrastructure_cost,
            'utility_infrastructure_cost': utility_infrastructure_cost,
            'total_infrastructure_cost': total_infrastructure_cost,
            'infrastructure_cost_per_building_m2': infrastructure_cost_per_building_m2
        }
    
    def calculate_social_benefits(self, metrics: Dict) -> Dict:
        """
        Calculate social benefits from urban design.
        
        Args:
            metrics: Dictionary containing urban area metrics
            
        Returns:
            Dictionary with social benefit calculations
        """
        # Extract metrics
        green_coverage = metrics.get('green_coverage', 0)
        building_density = metrics.get('building_density', 0)
        
        # Get accessibility metrics if available
        accessibility = metrics.get('accessibility', {})
        green_space_accessibility = accessibility.get('green_space_accessibility', 0)
        
        # Estimate population based on building area
        building_area = metrics.get('building_area_m2', 0)
        estimated_population = building_area / 30  # Rough estimate: 30m² per person
        
        # Value of health benefits from green space per person per year
        health_benefit_per_person = 250  # $250 per person per year
        
        # Adjust based on green space accessibility
        adjusted_health_benefit = health_benefit_per_person * green_space_accessibility
        
        # Calculate annual health benefits
        annual_health_benefits = adjusted_health_benefit * estimated_population
        
        # Calculate present value of health benefits over analysis period
        present_value_health_benefits = 0
        for year in range(1, self.analysis_period + 1):
            present_value_health_benefits += annual_health_benefits / ((1 + self.discount_rate) ** year)
        
        # Calculate productivity benefits from well-designed urban areas
        # Better connected, accessible areas with good green space promote productivity
        if 'road' in metrics and 'connectivity_score' in metrics['road']:
            connectivity_score = metrics['road']['connectivity_score']
        else:
            connectivity_score = 0.5  # Default
        
        productivity_multiplier = (green_coverage * 0.3) + (connectivity_score * 0.3) + (green_space_accessibility * 0.4)
        annual_productivity_benefit_per_person = 500 * productivity_multiplier  # Up to $500 per person
        annual_productivity_benefits = annual_productivity_benefit_per_person * estimated_population
        
        # Calculate present value of productivity benefits
        present_value_productivity_benefits = 0
        for year in range(1, self.analysis_period + 1):
            present_value_productivity_benefits += annual_productivity_benefits / ((1 + self.discount_rate) ** year)
        
        # Combined social benefits
        total_annual_social_benefits = annual_health_benefits + annual_productivity_benefits
        total_present_value_social_benefits = present_value_health_benefits + present_value_productivity_benefits
        
        return {
            'estimated_population': estimated_population,
            'annual_health_benefits': annual_health_benefits,
            'annual_productivity_benefits': annual_productivity_benefits,
            'total_annual_social_benefits': total_annual_social_benefits,
            'present_value_health_benefits': present_value_health_benefits,
            'present_value_productivity_benefits': present_value_productivity_benefits,
            'total_present_value_social_benefits': total_present_value_social_benefits
        }
    
    def calculate_roi(self, metrics: Dict, project_cost: Optional[float] = None) -> Dict:
        """
        Calculate Return on Investment for an urban development project.
        
        Args:
            metrics: Dictionary containing urban area metrics
            project_cost: Optional project cost (if not provided, will be estimated)
            
        Returns:
            Dictionary with ROI calculations
        """
        # Calculate various components
        maintenance_costs = self.calculate_maintenance_costs(metrics)
        property_values = self.calculate_property_value(metrics)
        infrastructure_costs = self.calculate_infrastructure_costs(metrics)
        social_benefits = self.calculate_social_benefits(metrics)
        
        # Estimate project cost if not provided
        if project_cost is None:
            # Base cost on infrastructure and typical development costs
            building_area = metrics.get('building_area_m2', 0)
            building_cost_per_m2 = 1500  # $1500/m² construction cost
            building_cost = building_area * building_cost_per_m2
            
            # Total project cost is building cost plus infrastructure
            project_cost = building_cost + infrastructure_costs['total_infrastructure_cost']
        
        # Calculate total benefits
        total_benefits = property_values['adjusted_property_value'] + social_benefits['total_present_value_social_benefits']
        
        # Calculate total costs
        total_costs = project_cost + maintenance_costs['present_value_of_maintenance']
        
        # Calculate ROI metrics
        net_present_value = total_benefits - total_costs
        
        # Benefit-cost ratio
        benefit_cost_ratio = total_benefits / total_costs if total_costs > 0 else 0
        
        # Simple ROI (%)
        simple_roi = (net_present_value / total_costs) * 100 if total_costs > 0 else 0
        
        # Calculate payback period (years)
        annual_benefits = property_values['adjusted_property_value'] / self.analysis_period + social_benefits['total_annual_social_benefits']
        annual_costs = maintenance_costs['total_annual_maintenance_cost']
        annual_net_benefit = annual_benefits - annual_costs
        
        if annual_net_benefit > 0:
            payback_period = project_cost / annual_net_benefit
        else:
            payback_period = float('inf')  # No payback
        
        # Assemble results
        roi_results = {
            'project_cost': project_cost,
            'total_benefits': total_benefits,
            'total_costs': total_costs,
            'net_present_value': net_present_value,
            'benefit_cost_ratio': benefit_cost_ratio,
            'roi_percent': simple_roi,
            'payback_period_years': payback_period,
            'is_profitable': net_present_value > 0,
            'property_value': property_values['adjusted_property_value'],
            'infrastructure_costs': infrastructure_costs['total_infrastructure_cost'],
            'maintenance_costs': maintenance_costs['present_value_of_maintenance'],
            'social_benefits': social_benefits['total_present_value_social_benefits']
        }
        
        return roi_results
    
    def calculate_comparative_roi(self, current_metrics: Dict, proposed_metrics: Dict, 
                                project_cost: Optional[float] = None) -> Dict:
        """
        Calculate comparative ROI between current and proposed urban designs.
        
        Args:
            current_metrics: Dictionary containing current urban area metrics
            proposed_metrics: Dictionary containing proposed urban area metrics
            project_cost: Optional project cost (if not provided, will be estimated)
            
        Returns:
            Dictionary with comparative ROI calculations
        """
        # Calculate ROI for current and proposed states
        current_roi = self.calculate_roi(current_metrics)
        proposed_roi = self.calculate_roi(proposed_metrics, project_cost)
        
        # Calculate differences
        roi_difference = proposed_roi['roi_percent'] - current_roi['roi_percent']
        npv_difference = proposed_roi['net_present_value'] - current_roi['net_present_value']
        benefit_cost_ratio_difference = proposed_roi['benefit_cost_ratio'] - current_roi['benefit_cost_ratio']
        
        # Calculate improvement percentages
        if current_roi['roi_percent'] != 0:
            roi_improvement_percent = (roi_difference / abs(current_roi['roi_percent'])) * 100
        else:
            roi_improvement_percent = float('inf') if roi_difference > 0 else 0
            
        if current_roi['net_present_value'] != 0:
            npv_improvement_percent = (npv_difference / abs(current_roi['net_present_value'])) * 100
        else:
            npv_improvement_percent = float('inf') if npv_difference > 0 else 0
        
        # Determine if the project is worth doing
        is_worthwhile = npv_difference > 0
        
        return {
            'current_roi': current_roi,
            'proposed_roi': proposed_roi,
            'roi_difference': roi_difference,
            'npv_difference': npv_difference,
            'benefit_cost_ratio_difference': benefit_cost_ratio_difference,
            'roi_improvement_percent': roi_improvement_percent,
            'npv_improvement_percent': npv_improvement_percent,
            'is_worthwhile': is_worthwhile,
            'recommendation': "Proceed with project" if is_worthwhile else "Project not financially justified"
        }
    
    def generate_roi_report(self, metrics: Dict, project_cost: Optional[float] = None, 
                          project_name: str = "Urban Development Project") -> str:
        """
        Generate a human-readable ROI report.
        
        Args:
            metrics: Dictionary containing urban area metrics
            project_cost: Optional project cost
            project_name: Name of the project
            
        Returns:
            Formatted ROI report
        """
        # Calculate ROI
        roi_results = self.calculate_roi(metrics, project_cost)
        
        # Format currency values
        def format_currency(value):
            return f"${value:,.2f}"
        
        # Generate report
        report = [
            f"ROI ANALYSIS: {project_name}",
            f"Date: {datetime.now().strftime('%Y-%m-%d')}",
            f"Analysis Period: {self.analysis_period} years",
            f"Discount Rate: {self.discount_rate:.1%}",
            "",
            "SUMMARY:",
            f"Project Cost: {format_currency(roi_results['project_cost'])}",
            f"Total Benefits (NPV): {format_currency(roi_results['total_benefits'])}",
            f"Total Costs (NPV): {format_currency(roi_results['total_costs'])}",
            f"Net Present Value: {format_currency(roi_results['net_present_value'])}",
            f"Benefit-Cost Ratio: {roi_results['benefit_cost_ratio']:.2f}",
            f"ROI: {roi_results['roi_percent']:.1f}%",
            f"Payback Period: {roi_results['payback_period_years']:.1f} years",
            "",
            "FINANCIAL RECOMMENDATION:",
            "✅ Project is financially viable." if roi_results['is_profitable'] else "❌ Project is not financially viable.",
            "",
            "DETAILED BREAKDOWN:",
            "Benefits:",
            f"  Property Value: {format_currency(roi_results['property_value'])}",
            f"  Social Benefits: {format_currency(roi_results['social_benefits'])}",
            "Costs:",
            f"  Infrastructure Costs: {format_currency(roi_results['infrastructure_costs'])}",
            f"  Maintenance Costs (NPV): {format_currency(roi_results['maintenance_costs'])}",
            "",
            "NOTE: This analysis is based on LiDAR-derived metrics and standard cost assumptions. Actual results may vary."
        ]
        
        return "\n".join(report)