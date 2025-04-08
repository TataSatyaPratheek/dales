# urban_point_cloud_analyzer/business/decision_support/decision_support_system.py
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from pathlib import Path
import json

class DecisionSupportSystem:
    """
    Decision Support System for urban planning that provides data-driven recommendations.
    This system integrates LiDAR analysis with business intelligence to support urban planning decisions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the decision support system.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Load urban planning guidelines if provided
        guidelines_path = self.config.get('guidelines_path')
        if guidelines_path:
            with open(guidelines_path, 'r') as f:
                self.guidelines = json.load(f)
        else:
            # Default urban planning guidelines
            self.guidelines = {
                'green_coverage': {
                    'min': 0.15,  # Minimum 15% green coverage
                    'ideal': 0.30,  # Ideal 30% green coverage
                    'weight': 4    # Importance weight (1-5)
                },
                'building_density': {
                    'max': 0.7,    # Maximum 70% building density
                    'ideal': 0.5,  # Ideal 50% building density 
                    'weight': 3    # Importance weight (1-5)
                },
                'road_connectivity': {
                    'min': 0.4,    # Minimum connectivity score
                    'ideal': 0.8,  # Ideal connectivity score
                    'weight': 3    # Importance weight (1-5)
                },
                'green_space_accessibility': {
                    'min': 0.4,    # Minimum accessibility score
                    'ideal': 0.9,  # Ideal accessibility score
                    'weight': 4    # Importance weight (1-5)
                },
                'permeable_surface_ratio': {
                    'min': 0.3,    # Minimum permeable surface ratio
                    'ideal': 0.5,  # Ideal permeable surface ratio
                    'weight': 3    # Importance weight (1-5)
                }
            }
    
    def evaluate_compliance(self, metrics: Dict) -> Dict:
        """
        Evaluate compliance with urban planning guidelines.
        
        Args:
            metrics: Dictionary of urban metrics
            
        Returns:
            Dictionary with compliance scores and recommendations
        """
        compliance_results = {}
        
        # Check green coverage
        if 'green_coverage' in metrics:
            green_coverage = metrics['green_coverage']
            guideline = self.guidelines['green_coverage']
            
            if green_coverage < guideline['min']:
                compliance = 0.0  # Non-compliant
                recommendation = f"Increase green coverage from {green_coverage:.1%} to at least {guideline['min']:.1%}."
            else:
                # Calculate compliance score (0-1) based on how close to ideal
                compliance = min(1.0, green_coverage / guideline['ideal'])
                if compliance < 0.8:
                    recommendation = f"Consider increasing green coverage from {green_coverage:.1%} closer to the ideal of {guideline['ideal']:.1%}."
                else:
                    recommendation = f"Green coverage of {green_coverage:.1%} meets guidelines."
            
            compliance_results['green_coverage'] = {
                'value': green_coverage,
                'min_requirement': guideline['min'],
                'ideal': guideline['ideal'],
                'compliance_score': compliance,
                'recommendation': recommendation,
                'weight': guideline['weight']
            }
        
        # Check building density
        if 'building_density' in metrics:
            building_density = metrics['building_density']
            guideline = self.guidelines['building_density']
            
            if building_density > guideline['max']:
                compliance = 0.0  # Non-compliant
                recommendation = f"Reduce building density from {building_density:.1%} to below {guideline['max']:.1%}."
            else:
                # Calculate compliance score (0-1) based on proximity to ideal
                # For building density, being under ideal is fine, but being over is problematic
                if building_density <= guideline['ideal']:
                    compliance = 1.0
                else:
                    # Linear scale from ideal to max
                    compliance = 1.0 - (building_density - guideline['ideal']) / (guideline['max'] - guideline['ideal'])
                
                if compliance < 0.8:
                    recommendation = f"Consider reducing building density from {building_density:.1%} closer to the ideal of {guideline['ideal']:.1%}."
                else:
                    recommendation = f"Building density of {building_density:.1%} meets guidelines."
            
            compliance_results['building_density'] = {
                'value': building_density,
                'max_requirement': guideline['max'],
                'ideal': guideline['ideal'],
                'compliance_score': compliance,
                'recommendation': recommendation,
                'weight': guideline['weight']
            }
        
        # Check road connectivity
        if 'road' in metrics and 'connectivity_score' in metrics['road']:
            connectivity = metrics['road']['connectivity_score']
            guideline = self.guidelines['road_connectivity']
            
            if connectivity < guideline['min']:
                compliance = 0.0  # Non-compliant
                recommendation = f"Improve road connectivity from {connectivity:.2f} to at least {guideline['min']:.2f}."
            else:
                # Calculate compliance score (0-1) based on how close to ideal
                compliance = min(1.0, connectivity / guideline['ideal'])
                if compliance < 0.8:
                    recommendation = f"Consider improving road connectivity from {connectivity:.2f} closer to the ideal of {guideline['ideal']:.2f}."
                else:
                    recommendation = f"Road connectivity of {connectivity:.2f} meets guidelines."
            
            compliance_results['road_connectivity'] = {
                'value': connectivity,
                'min_requirement': guideline['min'],
                'ideal': guideline['ideal'],
                'compliance_score': compliance,
                'recommendation': recommendation,
                'weight': guideline['weight']
            }
        
        # Check green space accessibility
        if 'accessibility' in metrics and 'green_space_accessibility' in metrics['accessibility']:
            accessibility = metrics['accessibility']['green_space_accessibility']
            guideline = self.guidelines['green_space_accessibility']
            
            if accessibility < guideline['min']:
                compliance = 0.0  # Non-compliant
                recommendation = f"Improve green space accessibility from {accessibility:.2f} to at least {guideline['min']:.2f}."
            else:
                # Calculate compliance score (0-1) based on how close to ideal
                compliance = min(1.0, accessibility / guideline['ideal'])
                if compliance < 0.8:
                    recommendation = f"Consider improving green space accessibility from {accessibility:.2f} closer to the ideal of {guideline['ideal']:.2f}."
                else:
                    recommendation = f"Green space accessibility of {accessibility:.2f} meets guidelines."
            
            compliance_results['green_space_accessibility'] = {
                'value': accessibility,
                'min_requirement': guideline['min'],
                'ideal': guideline['ideal'],
                'compliance_score': compliance,
                'recommendation': recommendation,
                'weight': guideline['weight']
            }
        
        # Check permeable surface ratio
        if 'environmental' in metrics and 'permeable_surface_ratio' in metrics['environmental']:
            ratio = metrics['environmental']['permeable_surface_ratio']
            guideline = self.guidelines['permeable_surface_ratio']
            
            if ratio < guideline['min']:
                compliance = 0.0  # Non-compliant
                recommendation = f"Increase permeable surface ratio from {ratio:.2f} to at least {guideline['min']:.2f}."
            else:
                # Calculate compliance score (0-1) based on how close to ideal
                compliance = min(1.0, ratio / guideline['ideal'])
                if compliance < 0.8:
                    recommendation = f"Consider increasing permeable surface ratio from {ratio:.2f} closer to the ideal of {guideline['ideal']:.2f}."
                else:
                    recommendation = f"Permeable surface ratio of {ratio:.2f} meets guidelines."
            
            compliance_results['permeable_surface_ratio'] = {
                'value': ratio,
                'min_requirement': guideline['min'],
                'ideal': guideline['ideal'],
                'compliance_score': compliance,
                'recommendation': recommendation,
                'weight': guideline['weight']
            }
        
        # Calculate overall compliance score
        if compliance_results:
            total_weight = sum(item['weight'] for item in compliance_results.values())
            weighted_sum = sum(item['compliance_score'] * item['weight'] for item in compliance_results.values())
            
            overall_compliance = weighted_sum / total_weight if total_weight > 0 else 0
            
            # Determine compliance category
            if overall_compliance >= 0.9:
                compliance_category = "Excellent"
            elif overall_compliance >= 0.8:
                compliance_category = "Good"
            elif overall_compliance >= 0.6:
                compliance_category = "Acceptable"
            elif overall_compliance >= 0.3:
                compliance_category = "Needs Improvement"
            else:
                compliance_category = "Non-Compliant"
            
            compliance_results['overall'] = {
                'compliance_score': overall_compliance,
                'compliance_category': compliance_category
            }
        
        return compliance_results
    
    def identify_improvement_areas(self, metrics: Dict) -> List[Dict]:
        """
        Identify key areas for improvement based on metrics.
        
        Args:
            metrics: Dictionary of urban metrics
            
        Returns:
            List of improvement opportunities with impact scores
        """
        # Get compliance evaluation
        compliance = self.evaluate_compliance(metrics)
        
        # Identify areas with low compliance scores
        improvement_areas = []
        
        for key, value in compliance.items():
            if key == 'overall':
                continue
                
            # Skip areas that are already compliant
            if value['compliance_score'] >= 0.8:
                continue
            
            # Calculate impact potential (higher weight and lower compliance means higher impact)
            impact_potential = value['weight'] * (1 - value['compliance_score'])
            
            improvement_areas.append({
                'area': key,
                'current_value': value['value'],
                'target_value': value.get('ideal', value.get('min_requirement', value.get('max_requirement'))),
                'compliance_score': value['compliance_score'],
                'recommendation': value['recommendation'],
                'impact_potential': impact_potential,
                'weight': value['weight']
            })
        
        # Sort by impact potential (highest first)
        improvement_areas.sort(key=lambda x: x['impact_potential'], reverse=True)
        
        return improvement_areas
    
    def assess_impact(self, current_metrics: Dict, proposed_metrics: Dict) -> Dict:
        """
        Assess the impact of proposed changes.
        
        Args:
            current_metrics: Dictionary of current urban metrics
            proposed_metrics: Dictionary of proposed urban metrics
            
        Returns:
            Dictionary with impact assessment
        """
        # Evaluate compliance for both current and proposed
        current_compliance = self.evaluate_compliance(current_metrics)
        proposed_compliance = self.evaluate_compliance(proposed_metrics)
        
        # Calculate improvements
        improvements = {}
        
        for key in set(current_compliance.keys()) & set(proposed_compliance.keys()):
            if key == 'overall':
                continue
                
            current = current_compliance[key]
            proposed = proposed_compliance[key]
            
            # Calculate absolute and relative improvements
            absolute_improvement = proposed['compliance_score'] - current['compliance_score']
            
            if current['compliance_score'] > 0:
                relative_improvement = (absolute_improvement / current['compliance_score']) * 100
            else:
                relative_improvement = float('inf') if absolute_improvement > 0 else 0
            
            # Determine impact category
            if absolute_improvement <= 0:
                impact_category = "Negative"
            elif absolute_improvement < 0.1:
                impact_category = "Minor"
            elif absolute_improvement < 0.3:
                impact_category = "Moderate"
            else:
                impact_category = "Significant"
            
            improvements[key] = {
                'current_value': current.get('value'),
                'proposed_value': proposed.get('value'),
                'current_compliance': current['compliance_score'],
                'proposed_compliance': proposed['compliance_score'],
                'absolute_improvement': absolute_improvement,
                'relative_improvement': relative_improvement,
                'impact_category': impact_category,
                'weight': current['weight']
            }
        
        # Calculate overall impact
        if 'overall' in current_compliance and 'overall' in proposed_compliance:
            current_overall = current_compliance['overall']['compliance_score']
            proposed_overall = proposed_compliance['overall']['compliance_score']
            
            absolute_improvement = proposed_overall - current_overall
            
            if current_overall > 0:
                relative_improvement = (absolute_improvement / current_overall) * 100
            else:
                relative_improvement = float('inf') if absolute_improvement > 0 else 0
            
            # Determine overall impact category
            if absolute_improvement <= 0:
                impact_category = "Negative"
            elif absolute_improvement < 0.1:
                impact_category = "Minor"
            elif absolute_improvement < 0.2:
                impact_category = "Moderate"
            else:
                impact_category = "Significant"
            
            improvements['overall'] = {
                'current_compliance': current_overall,
                'proposed_compliance': proposed_overall,
                'absolute_improvement': absolute_improvement,
                'relative_improvement': relative_improvement,
                'impact_category': impact_category,
                'current_category': current_compliance['overall']['compliance_category'],
                'proposed_category': proposed_compliance['overall']['compliance_category']
            }
        
        return improvements
    
    def generate_recommendations(self, metrics: Dict, focus: Optional[List[str]] = None) -> List[Dict]:
        """
        Generate specific, actionable recommendations.
        
        Args:
            metrics: Dictionary of urban metrics
            focus: Optional list of focus areas
            
        Returns:
            List of recommendation objects
        """
        # Get improvement areas
        improvement_areas = self.identify_improvement_areas(metrics)
        
        # Filter by focus areas if provided
        if focus:
            improvement_areas = [area for area in improvement_areas if area['area'] in focus]
        
        # Generate specific recommendations
        recommendations = []
        
        for area in improvement_areas:
            recommendation = {
                'area': area['area'],
                'priority': 'High' if area['impact_potential'] > 3 else 'Medium' if area['impact_potential'] > 1.5 else 'Low',
                'current_status': area['current_value'],
                'target': area['target_value'],
                'summary': area['recommendation'],
                'actions': []
            }
            
            # Add specific actions based on area
            if area['area'] == 'green_coverage':
                green_coverage = metrics.get('green_coverage', 0)
                target = area['target_value']
                
                # Calculate additional green area needed
                total_area = metrics.get('total_area_m2', 0)
                current_green_area = total_area * green_coverage
                target_green_area = total_area * target
                additional_area_needed = target_green_area - current_green_area
                
                recommendation['actions'] = [
                    f"Add {additional_area_needed:.1f} m² of green space to reach target coverage of {target:.1%}.",
                    "Identify underutilized spaces for new vegetation or green roofs.",
                    "Replace impervious surfaces with permeable green spaces where possible.",
                    "Incorporate green infrastructure into streetscapes."
                ]
                
            elif area['area'] == 'building_density':
                building_density = metrics.get('building_density', 0)
                target = area['target_value']
                
                # Calculate reduction needed
                total_area = metrics.get('total_area_m2', 0)
                current_building_area = total_area * building_density
                target_building_area = total_area * target
                reduction_needed = current_building_area - target_building_area
                
                recommendation['actions'] = [
                    f"Reduce building footprint by {reduction_needed:.1f} m² to reach target density of {target:.1%}.",
                    "Increase building height while reducing footprint for the same floor area.",
                    "Create more open spaces between buildings.",
                    "Convert some built areas to green or public spaces."
                ]
                
            elif area['area'] == 'road_connectivity':
                connectivity = metrics.get('road', {}).get('connectivity_score', 0)
                
                recommendation['actions'] = [
                    "Add connecting roads or pathways to improve network connectivity.",
                    "Reduce dead-end streets and cul-de-sacs.",
                    "Create more direct paths between major destinations.",
                    "Implement a more grid-like street pattern where possible."
                ]
                
            elif area['area'] == 'green_space_accessibility':
                recommendation['actions'] = [
                    "Add smaller green spaces distributed throughout the area.",
                    "Ensure all residential buildings have access to green space within 300m.",
                    "Create pedestrian paths connecting residential areas to green spaces.",
                    "Remove barriers to green space access."
                ]
                
            elif area['area'] == 'permeable_surface_ratio':
                recommendation['actions'] = [
                    "Replace impervious surfaces with permeable materials.",
                    "Install permeable pavement for parking areas and low-traffic roads.",
                    "Create rain gardens and bioswales for stormwater management.",
                    "Add green roofs to buildings to increase permeable area."
                ]
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def optimize_resources(self, metrics: Dict, budget: float) -> Dict:
        """
        Optimize resource allocation for maximum impact.
        
        Args:
            metrics: Dictionary of urban metrics
            budget: Available budget for improvements
            
        Returns:
            Dictionary with optimized resource allocation
        """
        # Get improvement areas
        improvement_areas = self.identify_improvement_areas(metrics)
        
        # Define typical costs for different improvements
        improvement_costs = {
            'green_coverage': 100,  # $100/m² to add green space
            'building_density': 200,  # $200/m² to redevelop building area
            'road_connectivity': 300,  # $300/m² for new road connections
            'green_space_accessibility': 150,  # $150/m² for accessibility improvements
            'permeable_surface_ratio': 120  # $120/m² for permeable surface conversion
        }
        
        # Define maximum possible improvements per dollar
        max_improvement_per_dollar = {}
        for area in improvement_areas:
            current = area['current_value']
            target = area['target_value']
            
            if area['area'] == 'building_density':
                # For building density, we want to decrease
                delta = max(0, current - target)
            else:
                # For everything else, we want to increase
                delta = max(0, target - current)
            
            # Estimate total area needed for improvement
            total_area = metrics.get('total_area_m2', 0)
            area_to_improve = total_area * delta
            
            # Calculate total cost
            total_cost = area_to_improve * improvement_costs[area['area']]
            
            # Calculate impact per dollar
            if total_cost > 0:
                impact_per_dollar = area['impact_potential'] / total_cost
            else:
                impact_per_dollar = 0
            
            max_improvement_per_dollar[area['area']] = {
                'impact_per_dollar': impact_per_dollar,
                'total_cost': total_cost,
                'impact_potential': area['impact_potential'],
                'area_to_improve': area_to_improve
            }
        
        # Sort by impact per dollar
        sorted_improvements = sorted(
            max_improvement_per_dollar.items(),
            key=lambda x: x[1]['impact_per_dollar'],
            reverse=True
        )
        
        # Allocate budget for maximum impact
        remaining_budget = budget
        allocation = {}
        
        for area_name, data in sorted_improvements:
            # Skip if no improvement needed
            if data['area_to_improve'] <= 0:
                continue
                
            # Calculate how much we can allocate to this area
            allocated = min(remaining_budget, data['total_cost'])
            remaining_budget -= allocated
            
            # Calculate percentage of needed improvement we can fund
            if data['total_cost'] > 0:
                improvement_percentage = allocated / data['total_cost'] * 100
            else:
                improvement_percentage = 100
            
            # Calculate area we can improve with this budget
            improved_area = (allocated / improvement_costs[area_name])
            
            allocation[area_name] = {
                'allocated_budget': allocated,
                'percentage_of_need': improvement_percentage,
                'area_improved': improved_area,
                'impact_score': data['impact_potential'] * (improvement_percentage / 100)
            }
            
            # Stop if we've used all the budget
            if remaining_budget <= 0:
                break
        
        # Calculate total impact
        total_impact = sum(item['impact_score'] for item in allocation.values())
        
        return {
            'total_budget': budget,
            'allocated_budget': budget - remaining_budget,
            'remaining_budget': remaining_budget,
            'allocation': allocation,
            'total_impact': total_impact
        }
    
    def generate_decision_report(self, metrics: Dict, budget: Optional[float] = None) -> str:
        """
        Generate a comprehensive decision support report.
        
        Args:
            metrics: Dictionary of urban metrics
            budget: Optional budget for improvements
            
        Returns:
            Formatted decision report
        """
        # Evaluate compliance
        compliance = self.evaluate_compliance(metrics)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(metrics)
        
        # Optimize resources if budget provided
        if budget:
            resource_allocation = self.optimize_resources(metrics, budget)
        else:
            resource_allocation = None
        
        # Generate report
        report = [
            "URBAN PLANNING DECISION SUPPORT REPORT",
            "=========================================",
            "",
            "COMPLIANCE ASSESSMENT:",
        ]
        
        if 'overall' in compliance:
            overall = compliance['overall']
            report.append(f"Overall Compliance: {overall['compliance_score']:.2f} ({overall['compliance_category']})")
        
        report.append("")
        report.append("Detailed Compliance Scores:")
        
        for key, value in compliance.items():
            if key != 'overall':
                report.append(f"- {key.replace('_', ' ').title()}: {value['compliance_score']:.2f}")
                if 'recommendation' in value:
                    report.append(f"  Recommendation: {value['recommendation']}")
        
        report.append("")
        report.append("PRIORITY RECOMMENDATIONS:")
        
        for i, rec in enumerate(recommendations[:5], 1):  # Top 5 recommendations
            report.append(f"{i}. {rec['area'].replace('_', ' ').title()} ({rec['priority']} Priority)")
            report.append(f"   Current: {rec['current_status']:.2f} | Target: {rec['target']:.2f}")
            report.append(f"   Summary: {rec['summary']}")
            report.append("   Actions:")
            for action in rec['actions']:
                report.append(f"   - {action}")
            report.append("")
        
        if resource_allocation:
            report.append("RESOURCE ALLOCATION:")
            report.append(f"Total Budget: ${resource_allocation['total_budget']:,.2f}")
            report.append(f"Allocated: ${resource_allocation['allocated_budget']:,.2f}")
            report.append(f"Remaining: ${resource_allocation['remaining_budget']:,.2f}")
            report.append("")
            report.append("Allocation by Area:")
            
            for area, data in resource_allocation['allocation'].items():
                report.append(f"- {area.replace('_', ' ').title()}: ${data['allocated_budget']:,.2f} ({data['percentage_of_need']:.1f}% of needed improvement)")
                report.append(f"  Area Improved: {data['area_improved']:.1f} m²")
                report.append(f"  Impact Score: {data['impact_score']:.2f}")
            
            report.append("")
            report.append(f"Total Expected Impact: {resource_allocation['total_impact']:.2f}")
        
        report.append("")
        report.append("NOTE: This analysis is based on LiDAR-derived metrics and urban planning guidelines.")
        report.append("Recommendations should be reviewed by qualified urban planners before implementation.")
        
        return "\n".join(report)
    
    def check_regulation_compliance(self, metrics: Dict, regulation_type: str = 'standard') -> Dict:
        """
        Check compliance with specific regulations.
        
        Args:
            metrics: Dictionary of urban metrics
            regulation_type: Type of regulations to check ('standard', 'green', 'accessibility')
            
        Returns:
            Dictionary with regulation compliance results
        """
        # Define regulations based on type
        if regulation_type == 'green':
            regulations = {
                'min_green_coverage': 0.20,  # Minimum 20% green coverage for green regulations
                'min_permeable_surface': 0.35,  # Minimum 35% permeable surface
                'max_building_density': 0.60  # Maximum 60% building density
            }
        elif regulation_type == 'accessibility':
            regulations = {
                'min_green_space_accessibility': 0.6,  # Higher accessibility requirements
                'min_road_connectivity': 0.5  # Higher connectivity requirements
            }
        else:  # standard
            regulations = {
                'min_green_coverage': 0.15,  # Minimum 15% green coverage
                'min_permeable_surface': 0.30,  # Minimum 30% permeable surface
                'max_building_density': 0.70,  # Maximum 70% building density
                'min_green_space_accessibility': 0.4,  # Minimum accessibility score
                'min_road_connectivity': 0.4  # Minimum connectivity score
            }
        
        # Check compliance with each regulation
        compliance_results = {
            'compliant': True,
            'violations': [],
            'details': {}
        }
        
        # Check green coverage
        if 'min_green_coverage' in regulations and 'green_coverage' in metrics:
            green_coverage = metrics['green_coverage']
            min_required = regulations['min_green_coverage']
            compliant = green_coverage >= min_required
            
            compliance_results['details']['green_coverage'] = {
                'required': min_required,
                'actual': green_coverage,
                'compliant': compliant
            }
            
            if not compliant:
                compliance_results['compliant'] = False
                compliance_results['violations'].append(
                    f"Green coverage ({green_coverage:.1%}) is below minimum requirement ({min_required:.1%})"
                )
        
        # Check building density
        if 'max_building_density' in regulations and 'building_density' in metrics:
            building_density = metrics['building_density']
            max_allowed = regulations['max_building_density']
            compliant = building_density <= max_allowed
            
            compliance_results['details']['building_density'] = {
                'required': f"≤ {max_allowed}",
                'actual': building_density,
                'compliant': compliant
            }
            
            if not compliant:
                compliance_results['compliant'] = False
                compliance_results['violations'].append(
                    f"Building density ({building_density:.1%}) exceeds maximum allowed ({max_allowed:.1%})"
                )
        
        # Check permeable surface
        if 'min_permeable_surface' in regulations and 'environmental' in metrics and 'permeable_surface_ratio' in metrics['environmental']:
            ratio = metrics['environmental']['permeable_surface_ratio']
            min_required = regulations['min_permeable_surface']
            compliant = ratio >= min_required
            
            compliance_results['details']['permeable_surface'] = {
                'required': min_required,
                'actual': ratio,
                'compliant': compliant
            }
            
            if not compliant:
                compliance_results['compliant'] = False
                compliance_results['violations'].append(
                    f"Permeable surface ratio ({ratio:.2f}) is below minimum requirement ({min_required:.2f})"
                )
        
        # Check green space accessibility
        if 'min_green_space_accessibility' in regulations and 'accessibility' in metrics and 'green_space_accessibility' in metrics['accessibility']:
            accessibility = metrics['accessibility']['green_space_accessibility']
            min_required = regulations['min_green_space_accessibility']
            compliant = accessibility >= min_required
            
            compliance_results['details']['green_space_accessibility'] = {
                'required': min_required,
                'actual': accessibility,
                'compliant': compliant
            }
            
            if not compliant:
                compliance_results['compliant'] = False
                compliance_results['violations'].append(
                    f"Green space accessibility ({accessibility:.2f}) is below minimum requirement ({min_required:.2f})"
                )
        
        # Check road connectivity
        if 'min_road_connectivity' in regulations and 'road' in metrics and 'connectivity_score' in metrics['road']:
            connectivity = metrics['road']['connectivity_score']
            min_required = regulations['min_road_connectivity']
            compliant = connectivity >= min_required
            
            compliance_results['details']['road_connectivity'] = {
                'required': min_required,
                'actual': connectivity,
                'compliant': compliant
            }
            
            if not compliant:
                compliance_results['compliant'] = False
                compliance_results['violations'].append(
                    f"Road connectivity ({connectivity:.2f}) is below minimum requirement ({min_required:.2f})"
                )
        
        return compliance_results