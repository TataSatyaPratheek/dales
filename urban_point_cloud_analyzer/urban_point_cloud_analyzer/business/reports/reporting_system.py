# urban_point_cloud_analyzer/business/reports/reporting_system.py
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape
import base64
from io import BytesIO
import re

class ComprehensiveReportingSystem:
    """
    Comprehensive reporting system that generates various types of reports
    from urban metrics data.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize reporting system.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Set up templates directory
        template_dir = self.config.get('template_dir')
        if template_dir:
            self.template_dir = Path(template_dir)
        else:
            # Default to module directory/templates
            current_dir = Path(__file__).parent
            self.template_dir = current_dir / 'templates'
        
        # Ensure template directory exists
        os.makedirs(self.template_dir, exist_ok=True)
        
        # Default class color map for consistent visualization
        self.class_colors = {
            0: '#AAAAAA',  # Ground - Gray
            1: '#33CC33',  # Vegetation - Green
            2: '#996633',  # Buildings - Brown
            3: '#3333CC',  # Water - Blue
            4: '#CC3333',  # Car - Red
            5: '#FF9933',  # Truck - Orange
            6: '#FFFF33',  # Powerline - Yellow
            7: '#993399'   # Fence - Purple
        }
        
        # Override with config if provided
        if 'class_colors' in self.config:
            self.class_colors.update(self.config['class_colors'])
        
        # Set up Jinja environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
        # Create default templates if they don't exist
        self._create_default_templates()
    
    def _create_default_templates(self):
        """Create default report templates if they don't exist."""
        # HTML template
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report_title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        h1, h2, h3, h4 {
            color: #2c3e50;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }
        .summary-box {
            background-color: #f8f9fa;
            border-left: 4px solid #4285f4;
            padding: 15px;
            margin-bottom: 20px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            background-color: #fff;
        }
        .metric-title {
            font-weight: bold;
            margin-bottom: 10px;
            color: #4285f4;
        }
        .metric-value {
            font-size: 24px;
            margin-bottom: 10px;
        }
        .chart-container {
            margin-bottom: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        th, td {
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f8f9fa;
        }
        .recommendation {
            background-color: #e8f5e9;
            border-left: 4px solid #34a853;
            padding: 15px;
            margin-bottom: 15px;
        }
        .warning {
            background-color: #fef6e0;
            border-left: 4px solid #fbbc04;
            padding: 15px;
            margin-bottom: 15px;
        }
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            text-align: center;
            font-size: 0.9em;
            color: #666;
        }
        .comparison-table td.better {
            background-color: rgba(52, 168, 83, 0.1);
        }
        .comparison-table td.worse {
            background-color: rgba(234, 67, 53, 0.1);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ report_title }}</h1>
        <p>Generated on {{ generation_date }}</p>
    </div>
    
    <div class="summary-box">
        <h2>Executive Summary</h2>
        {{ executive_summary | safe }}
    </div>
    
    <h2>Urban Metrics Analysis</h2>
    <div class="metrics-grid">
        {% for metric in main_metrics %}
        <div class="metric-card">
            <div class="metric-title">{{ metric.title }}</div>
            <div class="metric-value">{{ metric.value }}</div>
            <div class="metric-description">{{ metric.description }}</div>
        </div>
        {% endfor %}
    </div>
    
    <h2>Detailed Analysis</h2>
    
    {% for section in detail_sections %}
    <h3>{{ section.title }}</h3>
    <p>{{ section.description }}</p>
    
    {% if section.image %}
    <div class="chart-container">
        <img src="data:image/png;base64,{{ section.image }}" alt="{{ section.title }}" style="max-width:100%;">
    </div>
    {% endif %}
    
    {% if section.table %}
    <table>
        <thead>
            <tr>
                {% for header in section.table.headers %}
                <th>{{ header }}</th>
                {% endfor %}
            </tr>
        </thead>
        <tbody>
            {% for row in section.table.rows %}
            <tr>
                {% for cell in row %}
                <td>{{ cell }}</td>
                {% endfor %}
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% endif %}
    {% endfor %}
    
    <h2>Recommendations</h2>
    {% for rec in recommendations %}
    <div class="recommendation">
        <h3>{{ rec.title }}</h3>
        <p>{{ rec.description }}</p>
        {% if rec.action_items %}
        <ul>
            {% for item in rec.action_items %}
            <li>{{ item }}</li>
            {% endfor %}
        </ul>
        {% endif %}
    </div>
    {% endfor %}
    
    {% if warnings %}
    <h2>Warnings and Compliance Issues</h2>
    {% for warning in warnings %}
    <div class="warning">
        <h3>{{ warning.title }}</h3>
        <p>{{ warning.description }}</p>
    </div>
    {% endfor %}
    {% endif %}
    
    <div class="footer">
        <p>Report generated using Urban Point Cloud Analyzer</p>
        <p>{{ footer_text }}</p>
    </div>
</body>
</html>
"""
        html_template_path = self.template_dir / 'html_report.html'
        if not html_template_path.exists():
            with open(html_template_path, 'w') as f:
                f.write(html_template)
        
        # Comparison template
        comparison_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report_title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        h1, h2, h3, h4 {
            color: #2c3e50;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }
        .summary-box {
            background-color: #f8f9fa;
            border-left: 4px solid #4285f4;
            padding: 15px;
            margin-bottom: 20px;
        }
        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .comparison-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            background-color: #fff;
        }
        .comparison-title {
            font-weight: bold;
            margin-bottom: 10px;
            color: #4285f4;
        }
        .comparison-values {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 10px;
        }
        .current-value, .proposed-value {
            padding: 10px;
            text-align: center;
        }
        .current-value {
            background-color: #f8f9fa;
        }
        .proposed-value {
            background-color: #e8f5e9;
        }
        .improvement {
            font-weight: bold;
            color: #34a853;
        }
        .decline {
            font-weight: bold;
            color: #ea4335;
        }
        .chart-container {
            margin-bottom: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        th, td {
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f8f9fa;
        }
        .comparison-table td.better {
            background-color: rgba(52, 168, 83, 0.1);
        }
        .comparison-table td.worse {
            background-color: rgba(234, 67, 53, 0.1);
        }
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            text-align: center;
            font-size: 0.9em;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ report_title }}</h1>
        <p>Generated on {{ generation_date }}</p>
    </div>
    
    <div class="summary-box">
        <h2>Executive Summary</h2>
        {{ executive_summary | safe }}
    </div>
    
    <h2>Comparative Analysis</h2>
    <div class="comparison-grid">
        {% for metric in comparison_metrics %}
        <div class="comparison-card">
            <div class="comparison-title">{{ metric.title }}</div>
            <div class="comparison-values">
                <div class="current-value">
                    <div>Current</div>
                    <div>{{ metric.current_value }}</div>
                </div>
                <div class="proposed-value">
                    <div>Proposed</div>
                    <div>{{ metric.proposed_value }}</div>
                </div>
            </div>
            <div class="
                {% if metric.is_improvement %}improvement{% else %}decline{% endif %}
            ">
                {{ metric.change_description }}
            </div>
        </div>
        {% endfor %}
    </div>
    
    <h2>Impact Analysis</h2>
    
    {% for section in impact_sections %}
    <h3>{{ section.title }}</h3>
    <p>{{ section.description }}</p>
    
    {% if section.image %}
    <div class="chart-container">
        <img src="data:image/png;base64,{{ section.image }}" alt="{{ section.title }}" style="max-width:100%;">
    </div>
    {% endif %}
    
    {% if section.table %}
    <table class="comparison-table">
        <thead>
            <tr>
                {% for header in section.table.headers %}
                <th>{{ header }}</th>
                {% endfor %}
            </tr>
        </thead>
        <tbody>
            {% for row in section.table.rows %}
            <tr>
                {% for cell in row.cells %}
                <td class="{{ cell.class }}">{{ cell.value }}</td>
                {% endfor %}
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% endif %}
    {% endfor %}
    
    <h2>Recommendations Based on Comparison</h2>
    {% for rec in recommendations %}
    <div class="recommendation">
        <h3>{{ rec.title }}</h3>
        <p>{{ rec.description }}</p>
        {% if rec.action_items %}
        <ul>
            {% for item in rec.action_items %}
            <li>{{ item }}</li>
            {% endfor %}
        </ul>
        {% endif %}
    </div>
    {% endfor %}
    
    <div class="footer">
        <p>Report generated using Urban Point Cloud Analyzer</p>
        <p>{{ footer_text }}</p>
    </div>
</body>
</html>
"""
        comparison_template_path = self.template_dir / 'comparison_report.html'
        if not comparison_template_path.exists():
            with open(comparison_template_path, 'w') as f:
                f.write(comparison_template)
    
    def _fig_to_base64(self, fig):
        """Convert a matplotlib figure to base64 encoded string."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        return img_data
    
    def _plotly_to_base64(self, fig):
        """Convert a plotly figure to base64 encoded string."""
        img_bytes = fig.to_image(format="png", width=1000, height=600)
        img_data = base64.b64encode(img_bytes).decode('utf-8')
        return img_data
    
    def create_single_page_report(self, metrics: Dict, output_file: str, 
                                report_title: str = "Urban Metrics Report") -> str:
        """
        Create a single-page HTML report.
        
        Args:
            metrics: Dictionary of urban metrics
            output_file: Path to save the report
            report_title: Title of the report
            
        Returns:
            Path to the saved report
        """
        # Create executive summary
        if 'urban_quality_score' in metrics:
            quality_score = metrics['urban_quality_score']
            if quality_score >= 80:
                quality_category = "excellent"
            elif quality_score >= 60:
                quality_category = "good"
            elif quality_score >= 40:
                quality_category = "average"
            else:
                quality_category = "below average"
                
            executive_summary = f"""
            <p>This urban area has an overall quality score of <strong>{quality_score:.1f}/100</strong>,
            indicating <strong>{quality_category}</strong> urban design quality. The area has 
            {metrics.get('building_density', 0)*100:.1f}% building coverage,
            {metrics.get('green_coverage', 0)*100:.1f}% green space coverage, and
            {metrics.get('water_coverage', 0)*100:.1f}% water coverage.</p>
            """
        else:
            executive_summary = f"""
            <p>This urban area has {metrics.get('building_density', 0)*100:.1f}% building coverage,
            {metrics.get('green_coverage', 0)*100:.1f}% green space coverage, and
            {metrics.get('water_coverage', 0)*100:.1f}% water coverage. The area shows
            a mix of built and natural elements that define its urban character.</p>
            """
        
        # Create main metrics
        main_metrics = []
        
        # Add building density
        if 'building_density' in metrics:
            main_metrics.append({
                'title': 'Building Density',
                'value': f"{metrics['building_density']*100:.1f}%",
                'description': "Percentage of total area covered by buildings."
            })
        
        # Add green coverage
        if 'green_coverage' in metrics:
            main_metrics.append({
                'title': 'Green Coverage',
                'value': f"{metrics['green_coverage']*100:.1f}%",
                'description': "Percentage of total area covered by vegetation."
            })
        
        # Add water coverage
        if 'water_coverage' in metrics:
            main_metrics.append({
                'title': 'Water Coverage',
                'value': f"{metrics['water_coverage']*100:.1f}%",
                'description': "Percentage of total area covered by water bodies."
            })
        
        # Add building height
        if 'mean_building_height' in metrics:
            main_metrics.append({
                'title': 'Avg. Building Height',
                'value': f"{metrics['mean_building_height']:.1f} m",
                'description': "Average height of buildings in the area."
            })
        
        # Add vehicle count
        if 'vehicle_count' in metrics:
            main_metrics.append({
                'title': 'Vehicle Count',
                'value': f"{metrics['vehicle_count']}",
                'description': "Number of vehicles detected in the area."
            })
        
        # Urban quality score
        if 'urban_quality_score' in metrics:
            main_metrics.append({
                'title': 'Urban Quality Score',
                'value': f"{metrics['urban_quality_score']:.1f}/100",
                'description': "Overall urban quality score based on multiple factors."
            })
        
        # Create detail sections
        detail_sections = []
        
        # Land use distribution
        land_use_data = {
            'Building': metrics.get('building_density', 0) * 100,
            'Vegetation': metrics.get('green_coverage', 0) * 100,
            'Water': metrics.get('water_coverage', 0) * 100,
            'Other': 100 - (metrics.get('building_density', 0) + metrics.get('green_coverage', 0) + metrics.get('water_coverage', 0)) * 100
        }
        
        # Create pie chart
        plt.figure(figsize=(8, 6))
        plt.pie(land_use_data.values(), labels=land_use_data.keys(), autopct='%1.1f%%', 
                colors=['#996633', '#33CC33', '#3333CC', '#AAAAAA'])
        plt.title('Land Use Distribution')
        land_use_img = self._fig_to_base64(plt.gcf())
        
        detail_sections.append({
            'title': 'Land Use Distribution',
            'description': 'The distribution of land use in the urban area.',
            'image': land_use_img
        })
        
        # Road connectivity analysis
        if 'road' in metrics:
            road_metrics = metrics['road']
            road_data = {
                'Metric': ['Connectivity Score', 'Avg. Road Width', 'Road Density', 'Intersection Count'],
                'Value': [
                    f"{road_metrics.get('connectivity_score', 0):.2f}",
                    f"{road_metrics.get('avg_road_width', 0):.1f} m",
                    f"{road_metrics.get('road_density', 0)*100:.1f}%",
                    f"{road_metrics.get('intersection_count', 0)}"
                ]
            }
            
            detail_sections.append({
                'title': 'Road Network Analysis',
                'description': 'Analysis of the road network connectivity and characteristics.',
                'table': {
                    'headers': road_data['Metric'],
                    'rows': [road_data['Value']]
                }
            })
        
        # Accessibility analysis
        if 'accessibility' in metrics:
            access_metrics = metrics['accessibility']
            
            # Create bar chart
            access_data = {
                'Metric': ['Building to Road', 'Green Space', 'Avg. Distance to Road', 'Avg. Distance to Green'],
                'Value': [
                    access_metrics.get('building_to_road_accessibility', 0),
                    access_metrics.get('green_space_accessibility', 0),
                    access_metrics.get('avg_distance_to_road', 0),
                    access_metrics.get('avg_distance_to_green', 0)
                ]
            }
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(['Building-Road', 'Green Space'], 
                  [access_metrics.get('building_to_road_accessibility', 0), 
                   access_metrics.get('green_space_accessibility', 0)],
                  color=['#4285f4', '#34a853'])
            ax.set_ylim(0, 1)
            ax.set_ylabel('Accessibility Score (0-1)')
            ax.set_title('Accessibility Scores')
            access_img = self._fig_to_base64(fig)
            
            detail_sections.append({
                'title': 'Accessibility Analysis',
                'description': 'Analysis of how accessible different features are within the urban area.',
                'image': access_img,
                'table': {
                    'headers': ['Metric', 'Value'],
                    'rows': [
                        ['Building to Road Accessibility', f"{access_metrics.get('building_to_road_accessibility', 0):.2f}"],
                        ['Green Space Accessibility', f"{access_metrics.get('green_space_accessibility', 0):.2f}"],
                        ['Avg. Distance to Road', f"{access_metrics.get('avg_distance_to_road', 0):.1f} m"],
                        ['Avg. Distance to Green', f"{access_metrics.get('avg_distance_to_green', 0):.1f} m"]
                    ]
                }
            })
        
        # Urban density metrics
        if 'density' in metrics:
            density_metrics = metrics['density']
            
            # Create bar chart for density metrics
            density_data = {
                'Metric': ['Floor Area Ratio', 'Building Coverage', 'Height Variation', 'Urban Compactness'],
                'Value': [
                    density_metrics.get('floor_area_ratio', 0),
                    density_metrics.get('building_coverage_ratio', 0),
                    density_metrics.get('building_height_variation', 0),
                    density_metrics.get('urban_compactness', 0)
                ]
            }
            
            detail_sections.append({
                'title': 'Urban Density Metrics',
                'description': 'Analysis of urban density characteristics.',
                'table': {
                    'headers': ['Metric', 'Value'],
                    'rows': [
                        ['Floor Area Ratio', f"{density_metrics.get('floor_area_ratio', 0):.2f}"],
                        ['Building Coverage Ratio', f"{density_metrics.get('building_coverage_ratio', 0):.2f}"],
                        ['Building Height Variation', f"{density_metrics.get('building_height_variation', 0):.2f}"],
                        ['Urban Compactness', f"{density_metrics.get('urban_compactness', 0):.2f}"]
                    ]
                }
            })
        
        # Environmental metrics
        if 'environmental' in metrics:
            env_metrics = metrics['environmental']
            
            # Create bar chart for environmental metrics
            env_data = {
                'Metric': ['Permeable Surface', 'Green Space Frag.', 'Solar Exposure', 'Green-Built Ratio'],
                'Value': [
                    env_metrics.get('permeable_surface_ratio', 0),
                    env_metrics.get('green_space_fragmentation', 0),
                    env_metrics.get('solar_exposure', 0),
                    env_metrics.get('green_to_built_ratio', 0)
                ]
            }
            
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(env_data['Metric'], env_data['Value'], color=['#34a853', '#fbbc04', '#ea4335', '#4285f4'])
            ax.set_ylabel('Score')
            ax.set_title('Environmental Metrics')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            env_img = self._fig_to_base64(fig)
            
            detail_sections.append({
                'title': 'Environmental Metrics',
                'description': 'Analysis of environmental characteristics and sustainability.',
                'image': env_img
            })
        
        # Create recommendations
        recommendations = []
        
        # Green space recommendation
        if 'green_coverage' in metrics:
            green_coverage = metrics['green_coverage']
            if green_coverage < 0.15:
                recommendations.append({
                    'title': 'Increase Green Space Coverage',
                    'description': f"Current green coverage ({green_coverage:.1%}) is below the recommended minimum (15%).",
                    'action_items': [
                        "Add parks or green areas in underserved neighborhoods.",
                        "Implement green roofs on suitable buildings.",
                        "Convert underutilized spaces to pocket parks.",
                        "Plant street trees along major roads."
                    ]
                })
        
        # Building density recommendation
        if 'building_density' in metrics:
            building_density = metrics['building_density']
            if building_density > 0.7:
                recommendations.append({
                    'title': 'Reduce Building Density',
                    'description': f"Current building density ({building_density:.1%}) is above recommended levels.",
                    'action_items': [
                        "Create more open spaces between buildings.",
                        "Consider vertical development to reduce building footprint.",
                        "Implement maximum lot coverage regulations.",
                        "Establish minimum open space requirements for new developments."
                    ]
                })
        
        # Road connectivity recommendation
        if 'road' in metrics and 'connectivity_score' in metrics['road']:
            connectivity = metrics['road']['connectivity_score']
            if connectivity < 0.4:
                recommendations.append({
                    'title': 'Improve Road Connectivity',
                    'description': f"Current road connectivity score ({connectivity:.2f}) is below optimal levels.",
                    'action_items': [
                        "Add connecting streets to reduce dead-ends.",
                        "Implement a more grid-like street pattern where possible.",
                        "Add pedestrian and bicycle connections between disconnected areas.",
                        "Reduce block sizes in future developments."
                    ]
                })
        
        # Warnings section
        warnings = []
        
        # Check for potential issues
        # Green space accessibility warning
        if 'accessibility' in metrics and 'green_space_accessibility' in metrics['accessibility']:
            green_access = metrics['accessibility']['green_space_accessibility']
            if green_access < 0.4:
                warnings.append({
                    'title': 'Limited Green Space Accessibility',
                    'description': f"Green space accessibility score is {green_access:.2f}, indicating poor access to green spaces for many residents."
                })
        
        # Permeable surface warning
        if 'environmental' in metrics and 'permeable_surface_ratio' in metrics['environmental']:
            permeable = metrics['environmental']['permeable_surface_ratio']
            if permeable < 0.3:
                warnings.append({
                    'title': 'Low Permeable Surface Ratio',
                    'description': f"Permeable surface ratio is {permeable:.2f}, increasing flood risk and urban heat island effect."
                })
        
        # Render template
        template = self.jinja_env.get_template('html_report.html')
        rendered_html = template.render(
            report_title=report_title,
            generation_date=datetime.now().strftime('%Y-%m-%d %H:%M'),
            executive_summary=executive_summary,
            main_metrics=main_metrics,
            detail_sections=detail_sections,
            recommendations=recommendations,
            warnings=warnings,
            footer_text="Analysis based on LiDAR point cloud data"
        )
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(rendered_html)
        
        return output_file
    
    def create_comparison_report(self, current_metrics: Dict, proposed_metrics: Dict, 
                              output_file: str, report_title: str = "Urban Development Comparison") -> str:
        """
        Create a comparison report between current and proposed urban metrics.
        
        Args:
            current_metrics: Dictionary of current urban metrics
            proposed_metrics: Dictionary of proposed urban metrics
            output_file: Path to save the report
            report_title: Title of the report
            
        Returns:
            Path to the saved report
        """
        # Create executive summary
        if 'urban_quality_score' in current_metrics and 'urban_quality_score' in proposed_metrics:
            current_score = current_metrics['urban_quality_score']
            proposed_score = proposed_metrics['urban_quality_score']
            score_change = proposed_score - current_score
            
            if score_change > 0:
                change_description = f"an improvement of {score_change:.1f} points"
            else:
                change_description = f"a decrease of {abs(score_change):.1f} points"
                
            executive_summary = f"""
            <p>The proposed urban development would change the urban quality score from 
            <strong>{current_score:.1f}/100</strong> to <strong>{proposed_score:.1f}/100</strong>,
            representing {change_description}. Key changes include modifications to building density
            ({current_metrics.get('building_density', 0)*100:.1f}% to {proposed_metrics.get('building_density', 0)*100:.1f}%),
            green coverage ({current_metrics.get('green_coverage', 0)*100:.1f}% to {proposed_metrics.get('green_coverage', 0)*100:.1f}%),
            and water coverage ({current_metrics.get('water_coverage', 0)*100:.1f}% to {proposed_metrics.get('water_coverage', 0)*100:.1f}%).</p>
            """
        else:
            executive_summary = f"""
            <p>The proposed urban development would modify key urban metrics including 
            building density ({current_metrics.get('building_density', 0)*100:.1f}% to {proposed_metrics.get('building_density', 0)*100:.1f}%),
            green coverage ({current_metrics.get('green_coverage', 0)*100:.1f}% to {proposed_metrics.get('green_coverage', 0)*100:.1f}%),
            and water coverage ({current_metrics.get('water_coverage', 0)*100:.1f}% to {proposed_metrics.get('water_coverage', 0)*100:.1f}%).
            These changes would substantially alter the urban character of the area.</p>
            """
        
        # Create comparison metrics
        comparison_metrics = []
        
        # Add building density
        if 'building_density' in current_metrics and 'building_density' in proposed_metrics:
            current_val = current_metrics['building_density']
            proposed_val = proposed_metrics['building_density']
            change = proposed_val - current_val
            
            # For building density, lower is generally better (up to a point)
            is_improvement = change < 0 if current_val > 0.5 else change > 0
            
            if change > 0:
                change_description = f"Increases by {change*100:.1f}%"
            else:
                change_description = f"Decreases by {abs(change)*100:.1f}%"
            
            comparison_metrics.append({
                'title': 'Building Density',
                'current_value': f"{current_val*100:.1f}%",
                'proposed_value': f"{proposed_val*100:.1f}%",
                'change_description': change_description,
                'is_improvement': is_improvement
            })
        
        # Add green coverage
        if 'green_coverage' in current_metrics and 'green_coverage' in proposed_metrics:
            current_val = current_metrics['green_coverage']
            proposed_val = proposed_metrics['green_coverage']
            change = proposed_val - current_val
            
            # For green coverage, higher is better
            is_improvement = change > 0
            
            if change > 0:
                change_description = f"Increases by {change*100:.1f}%"
            else:
                change_description = f"Decreases by {abs(change)*100:.1f}%"
            
            comparison_metrics.append({
                'title': 'Green Coverage',
                'current_value': f"{current_val*100:.1f}%",
                'proposed_value': f"{proposed_val*100:.1f}%",
                'change_description': change_description,
                'is_improvement': is_improvement
            })
        
        # Add water coverage
        if 'water_coverage' in current_metrics and 'water_coverage' in proposed_metrics:
            current_val = current_metrics['water_coverage']
            proposed_val = proposed_metrics['water_coverage']
            change = proposed_val - current_val
            
            # For water coverage, higher is generally better
            is_improvement = change > 0
            
            if change > 0:
                change_description = f"Increases by {change*100:.1f}%"
            else:
                change_description = f"Decreases by {abs(change)*100:.1f}%"
            
            comparison_metrics.append({
                'title': 'Water Coverage',
                'current_value': f"{current_val*100:.1f}%",
                'proposed_value': f"{proposed_val*100:.1f}%",
                'change_description': change_description,
                'is_improvement': is_improvement
            })
        
        # Add mean building height
        if 'mean_building_height' in current_metrics and 'mean_building_height' in proposed_metrics:
            current_val = current_metrics['mean_building_height']
            proposed_val = proposed_metrics['mean_building_height']
            change = proposed_val - current_val
            
            # For building height, context-dependent - we'll say neutral
            is_improvement = True
            
            if change > 0:
                change_description = f"Increases by {change:.1f} m"
            else:
                change_description = f"Decreases by {abs(change):.1f} m"
            
            comparison_metrics.append({
                'title': 'Avg. Building Height',
                'current_value': f"{current_val:.1f} m",
                'proposed_value': f"{proposed_val:.1f} m",
                'change_description': change_description,
                'is_improvement': is_improvement
            })
        
        # Add urban quality score
        if 'urban_quality_score' in current_metrics and 'urban_quality_score' in proposed_metrics:
            current_val = current_metrics['urban_quality_score']
            proposed_val = proposed_metrics['urban_quality_score']
            change = proposed_val - current_val
            
            # For quality score, higher is better
            is_improvement = change > 0
            
            if change > 0:
                change_description = f"Improves by {change:.1f} points"
            else:
                change_description = f"Decreases by {abs(change):.1f} points"
            
            comparison_metrics.append({
                'title': 'Urban Quality Score',
                'current_value': f"{current_val:.1f}/100",
                'proposed_value': f"{proposed_val:.1f}/100",
                'change_description': change_description,
                'is_improvement': is_improvement
            })
        
        # Create impact sections
        impact_sections = []
        
        # Land use change
        current_land_use = {
            'Building': current_metrics.get('building_density', 0) * 100,
            'Vegetation': current_metrics.get('green_coverage', 0) * 100,
            'Water': current_metrics.get('water_coverage', 0) * 100,
            'Other': 100 - (current_metrics.get('building_density', 0) + 
                          current_metrics.get('green_coverage', 0) + 
                          current_metrics.get('water_coverage', 0)) * 100
        }
        
        proposed_land_use = {
            'Building': proposed_metrics.get('building_density', 0) * 100,
            'Vegetation': proposed_metrics.get('green_coverage', 0) * 100,
            'Water': proposed_metrics.get('water_coverage', 0) * 100,
            'Other': 100 - (proposed_metrics.get('building_density', 0) + 
                          proposed_metrics.get('green_coverage', 0) + 
                          proposed_metrics.get('water_coverage', 0)) * 100
        }
        
        # Create side-by-side pie charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.pie(current_land_use.values(), labels=current_land_use.keys(), autopct='%1.1f%%',
               colors=['#996633', '#33CC33', '#3333CC', '#AAAAAA'])
        ax1.set_title('Current Land Use')
        
        ax2.pie(proposed_land_use.values(), labels=proposed_land_use.keys(), autopct='%1.1f%%',
               colors=['#996633', '#33CC33', '#3333CC', '#AAAAAA'])
        ax2.set_title('Proposed Land Use')
        
        plt.tight_layout()
        land_use_img = self._fig_to_base64(fig)
        
        impact_sections.append({
            'title': 'Land Use Changes',
            'description': 'Comparison of land use distribution between current and proposed development.',
            'image': land_use_img
        })
        
        # Accessibility impact
        if ('accessibility' in current_metrics and 'accessibility' in proposed_metrics and
            'building_to_road_accessibility' in current_metrics['accessibility'] and
            'building_to_road_accessibility' in proposed_metrics['accessibility']):
            
            current_access = current_metrics['accessibility']
            proposed_access = proposed_metrics['accessibility']
            
            # Create side-by-side bar chart
            labels = ['Building-Road', 'Green Space']
            current_values = [current_access.get('building_to_road_accessibility', 0),
                            current_access.get('green_space_accessibility', 0)]
            proposed_values = [proposed_access.get('building_to_road_accessibility', 0),
                             proposed_access.get('green_space_accessibility', 0)]
            
            x = np.arange(len(labels))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(x - width/2, current_values, width, label='Current', color='#4285f4')
            ax.bar(x + width/2, proposed_values, width, label='Proposed', color='#34a853')
            
            ax.set_ylim(0, 1)
            ax.set_ylabel('Accessibility Score (0-1)')
            ax.set_title('Accessibility Impact')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            
            plt.tight_layout()
            access_img = self._fig_to_base64(fig)
            
            impact_sections.append({
                'title': 'Accessibility Impact',
                'description': 'Changes in accessibility metrics between current and proposed development.',
                'image': access_img,
                'table': {
                    'headers': ['Metric', 'Current', 'Proposed', 'Change'],
                    'rows': [
                        {
                            'cells': [
                                {'value': 'Building to Road', 'class': ''},
                                {'value': f"{current_access.get('building_to_road_accessibility', 0):.2f}", 'class': ''},
                                {'value': f"{proposed_access.get('building_to_road_accessibility', 0):.2f}", 'class': ''},
                                {'value': f"{proposed_access.get('building_to_road_accessibility', 0) - current_access.get('building_to_road_accessibility', 0):.2f}", 
                                'class': 'better' if proposed_access.get('building_to_road_accessibility', 0) > current_access.get('building_to_road_accessibility', 0) else 'worse'}
                            ]
                        },
                        {
                            'cells': [
                                {'value': 'Green Space', 'class': ''},
                                {'value': f"{current_access.get('green_space_accessibility', 0):.2f}", 'class': ''},
                                {'value': f"{proposed_access.get('green_space_accessibility', 0):.2f}", 'class': ''},
                                {'value': f"{proposed_access.get('green_space_accessibility', 0) - current_access.get('green_space_accessibility', 0):.2f}", 
                                'class': 'better' if proposed_access.get('green_space_accessibility', 0) > current_access.get('green_space_accessibility', 0) else 'worse'}
                            ]
                        }
                    ]
                }
            })
        
        # Environmental impact
        if ('environmental' in current_metrics and 'environmental' in proposed_metrics):
            current_env = current_metrics['environmental']
            proposed_env = proposed_metrics['environmental']
            
            # Create side-by-side bar chart for environmental metrics
            labels = ['Permeable Surface', 'Solar Exposure', 'Green-Built Ratio']
            current_values = [
                current_env.get('permeable_surface_ratio', 0),
                current_env.get('solar_exposure', 0),
                current_env.get('green_to_built_ratio', 0)
            ]
            proposed_values = [
                proposed_env.get('permeable_surface_ratio', 0),
                proposed_env.get('solar_exposure', 0),
                proposed_env.get('green_to_built_ratio', 0)
            ]
            
            x = np.arange(len(labels))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x - width/2, current_values, width, label='Current', color='#4285f4')
            ax.bar(x + width/2, proposed_values, width, label='Proposed', color='#34a853')
            
            ax.set_ylabel('Score')
            ax.set_title('Environmental Impact')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            
            plt.tight_layout()
            env_img = self._fig_to_base64(fig)
            
            # Create table rows
            env_rows = []
            env_metrics = [
                ('Permeable Surface', 'permeable_surface_ratio'),
                ('Solar Exposure', 'solar_exposure'),
                ('Green-Built Ratio', 'green_to_built_ratio')
            ]
            
            for label, key in env_metrics:
                current_val = current_env.get(key, 0)
                proposed_val = proposed_env.get(key, 0)
                change = proposed_val - current_val
                
                env_rows.append({
                    'cells': [
                        {'value': label, 'class': ''},
                        {'value': f"{current_val:.2f}", 'class': ''},
                        {'value': f"{proposed_val:.2f}", 'class': ''},
                        {'value': f"{change:.2f}", 'class': 'better' if change > 0 else 'worse'}
                    ]
                })
            
            impact_sections.append({
                'title': 'Environmental Impact',
                'description': 'Changes in environmental metrics between current and proposed development.',
                'image': env_img,
                'table': {
                    'headers': ['Metric', 'Current', 'Proposed', 'Change'],
                    'rows': env_rows
                }
            })
        
        # Create recommendations
        recommendations = []
        
        # Analyze the changes to make recommendations
        # Building density recommendation
        if 'building_density' in current_metrics and 'building_density' in proposed_metrics:
            current_val = current_metrics['building_density']
            proposed_val = proposed_metrics['building_density']
            change = proposed_val - current_val
            
            if proposed_val > 0.7 and change > 0:
                recommendations.append({
                    'title': 'Reduce Proposed Building Density',
                    'description': f"The proposed building density ({proposed_val:.1%}) exceeds recommended levels.",
                    'action_items': [
                        "Reduce building footprint while maintaining floor area.",
                        "Increase green space to balance building density.",
                        "Consider more efficient building layouts to reduce density."
                    ]
                })
        
        # Green space recommendation
        if 'green_coverage' in current_metrics and 'green_coverage' in proposed_metrics:
            current_val = current_metrics['green_coverage']
            proposed_val = proposed_metrics['green_coverage']
            change = proposed_val - current_val
            
            if proposed_val < 0.15:
                recommendations.append({
                    'title': 'Increase Green Space in Proposal',
                    'description': f"The proposed green coverage ({proposed_val:.1%}) is below recommended minimum (15%).",
                    'action_items': [
                        "Add additional green spaces throughout the development.",
                        "Incorporate green roofs and vertical gardens.",
                        "Preserve existing vegetation where possible."
                    ]
                })
            elif change < -0.05:  # Significant decrease in green coverage
                recommendations.append({
                    'title': 'Preserve More Green Space',
                    'description': f"The proposal would reduce green coverage by {abs(change)*100:.1f}%, which may impact quality of life.",
                    'action_items': [
                        "Revise plans to preserve more existing vegetation.",
                        "Add compensatory green spaces to offset losses.",
                        "Consider the ecosystem services lost with reduced green coverage."
                    ]
                })
        
        # Accessibility recommendation
        if ('accessibility' in current_metrics and 'accessibility' in proposed_metrics and
            'green_space_accessibility' in current_metrics['accessibility'] and
            'green_space_accessibility' in proposed_metrics['accessibility']):
            
            current_val = current_metrics['accessibility']['green_space_accessibility']
            proposed_val = proposed_metrics['accessibility']['green_space_accessibility']
            change = proposed_val - current_val
            
            if proposed_val < 0.4:
                recommendations.append({
                    'title': 'Improve Green Space Accessibility',
                    'description': f"The proposed green space accessibility ({proposed_val:.2f}) is below recommended levels.",
                    'action_items': [
                        "Distribute green spaces more evenly throughout the development.",
                        "Ensure pedestrian paths connect residential areas to green spaces.",
                        "Aim for all residences to have green space access within 300m."
                    ]
                })
        
        # Render template
        template = self.jinja_env.get_template('comparison_report.html')
        rendered_html = template.render(
            report_title=report_title,
            generation_date=datetime.now().strftime('%Y-%m-%d %H:%M'),
            executive_summary=executive_summary,
            comparison_metrics=comparison_metrics,
            impact_sections=impact_sections,
            recommendations=recommendations,
            footer_text="Analysis based on LiDAR point cloud data"
        )
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(rendered_html)
        
        return output_file
    
    def create_compliance_report(self, metrics: Dict, regulation_type: str = 'standard',
                              output_file: str = None) -> Union[str, Dict]:
        """
        Create a compliance report to check against regulations.
        
        Args:
            metrics: Dictionary of urban metrics
            regulation_type: Type of regulations to check ('standard', 'green', 'accessibility')
            output_file: Optional path to save the report
            
        Returns:
            Path to saved report or compliance results dictionary
        """
        # Import decision support system to check compliance
        from urban_point_cloud_analyzer.business.decision_support.decision_support_system import DecisionSupportSystem
        
        # Create decision support system
        dss = DecisionSupportSystem()
        
        # Check compliance
        compliance_results = dss.check_regulation_compliance(metrics, regulation_type)
        
        # If no output file, return the compliance results
        if not output_file:
            return compliance_results
        
        # Create report content
        report_title = f"Urban Planning Compliance Report: {regulation_type.title()} Regulations"
        
        # Create executive summary
        if compliance_results['compliant']:
            executive_summary = f"""
            <p>The urban area <strong>complies with all {regulation_type} regulations</strong>. 
            All key metrics meet or exceed the minimum requirements for this regulatory framework.</p>
            """
        else:
            num_violations = len(compliance_results['violations'])
            executive_summary = f"""
            <p>The urban area has <strong>{num_violations} compliance violations</strong> with {regulation_type} regulations.
            These violations will need to be addressed to achieve regulatory compliance.</p>
            """
        
        # Create main metrics table
        metrics_table = {
            'headers': ['Metric', 'Requirement', 'Actual Value', 'Status'],
            'rows': []
        }
        
        for metric, details in compliance_results['details'].items():
            metrics_table['rows'].append([
                metric.replace('_', ' ').title(),
                str(details['required']),
                f"{details['actual']:.2f}" if isinstance(details['actual'], float) else str(details['actual']),
                '✅ Compliant' if details['compliant'] else '❌ Non-compliant'
            ])
        
        # Create warnings section
        warnings = []
        for violation in compliance_results['violations']:
            warnings.append({
                'title': 'Compliance Violation',
                'description': violation
            })
        
        # Create recommendations
        recommendations = []
        
        # Building density recommendation
        if ('building_density' in compliance_results['details'] and 
            not compliance_results['details']['building_density']['compliant']):
            
            actual = compliance_results['details']['building_density']['actual']
            required = compliance_results['details']['building_density']['required']
            
            required_val = float(required.replace('≤ ', ''))
            reduction_needed = actual - required_val
            
            recommendations.append({
                'title': 'Reduce Building Density',
                'description': f"Building density ({actual:.1%}) exceeds maximum allowed ({required}).",
                'action_items': [
                    f"Reduce building coverage by at least {reduction_needed*100:.1f}%.",
                    "Increase open space between buildings.",
                    "Consider vertical development to reduce building footprint."
                ]
            })
        
        # Green coverage recommendation
        if ('green_coverage' in compliance_results['details'] and 
            not compliance_results['details']['green_coverage']['compliant']):
            
            actual = compliance_results['details']['green_coverage']['actual']
            required = compliance_results['details']['green_coverage']['required']
            
            increase_needed = required - actual
            
            recommendations.append({
                'title': 'Increase Green Coverage',
                'description': f"Green coverage ({actual:.1%}) is below required minimum ({required}).",
                'action_items': [
                    f"Add at least {increase_needed*100:.1f}% additional green space.",
                    "Convert hardscape areas to permeable green space.",
                    "Add green roofs to suitable buildings.",
                    "Plant more street trees and vegetation."
                ]
            })
        
        # Permeable surface recommendation
        if ('permeable_surface' in compliance_results['details'] and 
            not compliance_results['details']['permeable_surface']['compliant']):
            
            actual = compliance_results['details']['permeable_surface']['actual']
            required = compliance_results['details']['permeable_surface']['required']
            
            increase_needed = required - actual
            
            recommendations.append({
                'title': 'Increase Permeable Surface',
                'description': f"Permeable surface ratio ({actual:.2f}) is below required minimum ({required}).",
                'action_items': [
                    "Replace impervious surfaces with permeable materials.",
                    "Install permeable pavement for parking areas.",
                    "Create more rain gardens and bioswales.",
                    "Reduce total paved area."
                ]
            })
        
        # Create report content
        detail_sections = [{
            'title': 'Compliance Details',
            'description': 'Detailed comparison of urban metrics against regulatory requirements.',
            'table': metrics_table
        }]
        
        # Render a simple HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .summary {{ background-color: {'#d4edda' if compliance_results['compliant'] else '#f8d7da'}; 
                            padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .compliant {{ color: green; }}
                .non-compliant {{ color: red; }}
                .warning {{ background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin-bottom: 10px; }}
                .recommendation {{ background-color: #d1ecf1; padding: 10px; border-left: 4px solid #17a2b8; margin-bottom: 10px; }}
            </style>
        </head>
        <body>
            <h1>{report_title}</h1>
            <div class="summary">
                {executive_summary}
            </div>
            
            <h2>Compliance Details</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Requirement</th>
                    <th>Actual Value</th>
                    <th>Status</th>
                </tr>
                {"".join([f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td><td class=\"{'compliant' if '✅' in row[3] else 'non-compliant'}\">{row[3]}</td></tr>" for row in metrics_table['rows']])}
            </table>
            
            {"<h2>Compliance Violations</h2>" if warnings else ""}
            {"".join([f"<div class=\"warning\"><h3>{w['title']}</h3><p>{w['description']}</p></div>" for w in warnings])}
            
            {"<h2>Recommendations to Achieve Compliance</h2>" if recommendations else ""}
            {"".join([f"<div class=\"recommendation\"><h3>{r['title']}</h3><p>{r['description']}</p><ul>{''.join([f'<li>{item}</li>' for item in r['action_items']])}</ul></div>" for r in recommendations])}
            
            <div style="margin-top: 30px; font-size: 0.8em; color: #666;">
                <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                <p>Urban Point Cloud Analyzer - Compliance Report</p>
            </div>
        </body>
        </html>
        """
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_file
    
    def create_project_impact_report(self, current_metrics: Dict, proposed_metrics: Dict, 
                                  costs: Dict, output_file: str) -> str:
        """
        Create a comprehensive project impact report with ROI analysis.
        
        Args:
            current_metrics: Dictionary of current urban metrics
            proposed_metrics: Dictionary of proposed urban metrics
            costs: Dictionary of project costs
            output_file: Path to save the report
            
        Returns:
            Path to the saved report
        """
        # Import ROI calculator
        from urban_point_cloud_analyzer.business.roi.roi_calculator import ROICalculator
        
        # Create ROI calculator
        roi_calculator = ROICalculator()
        
        # Calculate ROI
        project_cost = costs.get('total_project_cost', 1000000)  # Default $1M if not provided
        maintenance_cost = costs.get('annual_maintenance_cost', 50000)  # Default $50K/year if not provided
        
        # Calculate comparative ROI
        roi_results = roi_calculator.calculate_comparative_roi(current_metrics, proposed_metrics, project_cost)
        
        # Create report title
        report_title = "Project Impact and ROI Analysis"
        
        # Create executive summary
        if roi_results['is_worthwhile']:
            executive_summary = f"""
            <p>The proposed project has a <strong>positive return on investment</strong>, with an NPV difference of 
            <strong>${roi_results['npv_difference']:,.2f}</strong> and an ROI improvement of 
            <strong>{roi_results['roi_improvement_percent']:.1f}%</strong> compared to the current state.
            The project is economically justified with a benefit-cost ratio of 
            <strong>{roi_results['proposed_roi']['benefit_cost_ratio']:.2f}</strong>.</p>
            """
        else:
            executive_summary = f"""
            <p>The proposed project has a <strong>negative return on investment</strong>, with an NPV difference of 
            <strong>${roi_results['npv_difference']:,.2f}</strong>. With a benefit-cost ratio of 
            <strong>{roi_results['proposed_roi']['benefit_cost_ratio']:.2f}</strong>, the project is not 
            economically justified based on financial metrics alone.</p>
            """
        
        # Create financial summary
        financial_summary = {
            'headers': ['Metric', 'Current', 'Proposed', 'Difference'],
            'rows': []
        }
        
        # Add NPV
        financial_summary['rows'].append({
            'cells': [
                {'value': 'Net Present Value', 'class': ''},
                {'value': f"${roi_results['current_roi']['net_present_value']:,.2f}", 'class': ''},
                {'value': f"${roi_results['proposed_roi']['net_present_value']:,.2f}", 'class': ''},
                {'value': f"${roi_results['npv_difference']:,.2f}", 
                 'class': 'better' if roi_results['npv_difference'] > 0 else 'worse'}
            ]
        })
        
        # Add benefit-cost ratio
        financial_summary['rows'].append({
            'cells': [
                {'value': 'Benefit-Cost Ratio', 'class': ''},
                {'value': f"{roi_results['current_roi']['benefit_cost_ratio']:.2f}", 'class': ''},
                {'value': f"{roi_results['proposed_roi']['benefit_cost_ratio']:.2f}", 'class': ''},
                {'value': f"{roi_results['benefit_cost_ratio_difference']:.2f}", 
                 'class': 'better' if roi_results['benefit_cost_ratio_difference'] > 0 else 'worse'}
            ]
        })
        
        # Add ROI
        financial_summary['rows'].append({
            'cells': [
                {'value': 'ROI (%)', 'class': ''},
                {'value': f"{roi_results['current_roi']['roi_percent']:.1f}%", 'class': ''},
                {'value': f"{roi_results['proposed_roi']['roi_percent']:.1f}%", 'class': ''},
                {'value': f"{roi_results['roi_difference']:.1f}%", 
                 'class': 'better' if roi_results['roi_difference'] > 0 else 'worse'}
            ]
        })
        
        # Add payback period
        financial_summary['rows'].append({
            'cells': [
                {'value': 'Payback Period (years)', 'class': ''},
                {'value': f"{roi_results['current_roi']['payback_period_years']:.1f}", 'class': ''},
                {'value': f"{roi_results['proposed_roi']['payback_period_years']:.1f}", 'class': ''},
                {'value': f"{roi_results['current_roi']['payback_period_years'] - roi_results['proposed_roi']['payback_period_years']:.1f}", 
                 'class': 'better' if roi_results['proposed_roi']['payback_period_years'] < roi_results['current_roi']['payback_period_years'] else 'worse'}
            ]
        })
        
        # Create cost breakdown
        cost_breakdown = {
            'headers': ['Cost Category', 'Amount'],
            'rows': []
        }
        
        # Add project cost
        cost_breakdown['rows'].append(['Project Cost', f"${project_cost:,.2f}"])
        
        # Add maintenance costs
        maintenance_npv = roi_results['proposed_roi']['maintenance_costs']
        cost_breakdown['rows'].append(['Maintenance (NPV)', f"${maintenance_npv:,.2f}"])
        
        # Add infrastructure costs
        infrastructure_costs = roi_results['proposed_roi']['infrastructure_costs']
        cost_breakdown['rows'].append(['Infrastructure', f"${infrastructure_costs:,.2f}"])
        
        # Create benefits breakdown
        benefits_breakdown = {
            'headers': ['Benefit Category', 'Current Value', 'Proposed Value', 'Change'],
            'rows': []
        }
        
        # Add property value
        benefits_breakdown['rows'].append([
            'Property Value', 
            f"${roi_results['current_roi']['property_value']:,.2f}",
            f"${roi_results['proposed_roi']['property_value']:,.2f}",
            f"${roi_results['proposed_roi']['property_value'] - roi_results['current_roi']['property_value']:,.2f}"
        ])
        
        # Add social benefits
        benefits_breakdown['rows'].append([
            'Social Benefits', 
            f"${roi_results['current_roi']['social_benefits']:,.2f}",
            f"${roi_results['proposed_roi']['social_benefits']:,.2f}",
            f"${roi_results['proposed_roi']['social_benefits'] - roi_results['current_roi']['social_benefits']:,.2f}"
        ])
        
        # Create urban metric impacts
        urban_impacts = []
        
        # Green coverage impact
        if 'green_coverage' in current_metrics and 'green_coverage' in proposed_metrics:
            current_val = current_metrics['green_coverage']
            proposed_val = proposed_metrics['green_coverage']
            change = proposed_val - current_val
            
            impact_category = "Significant Improvement" if change > 0.1 else \
                            "Moderate Improvement" if change > 0 else \
                            "No Change" if change == 0 else \
                            "Moderate Decline" if change > -0.1 else \
                            "Significant Decline"
            
            urban_impacts.append({
                'metric': 'Green Coverage',
                'current': f"{current_val*100:.1f}%",
                'proposed': f"{proposed_val*100:.1f}%",
                'change': f"{change*100:+.1f}%",
                'impact': impact_category
            })
        
        # Building density impact
        if 'building_density' in current_metrics and 'building_density' in proposed_metrics:
            current_val = current_metrics['building_density']
            proposed_val = proposed_metrics['building_density']
            change = proposed_val - current_val
            
            if current_val > 0.7:  # Already too dense
                impact_category = "Improvement" if change < 0 else "Further Decline"
            elif proposed_val > 0.7:  # Becoming too dense
                impact_category = "Significant Decline"
            else:
                impact_category = "Acceptable Change"
            
            urban_impacts.append({
                'metric': 'Building Density',
                'current': f"{current_val*100:.1f}%",
                'proposed': f"{proposed_val*100:.1f}%",
                'change': f"{change*100:+.1f}%",
                'impact': impact_category
            })
        
        # Accessibility impact
        if ('accessibility' in current_metrics and 'accessibility' in proposed_metrics and
            'green_space_accessibility' in current_metrics['accessibility'] and
            'green_space_accessibility' in proposed_metrics['accessibility']):
            
            current_val = current_metrics['accessibility']['green_space_accessibility']
            proposed_val = proposed_metrics['accessibility']['green_space_accessibility']
            change = proposed_val - current_val
            
            impact_category = "Significant Improvement" if change > 0.2 else \
                            "Moderate Improvement" if change > 0 else \
                            "No Change" if change == 0 else \
                            "Moderate Decline" if change > -0.2 else \
                            "Significant Decline"
            
            urban_impacts.append({
                'metric': 'Green Space Accessibility',
                'current': f"{current_val:.2f}",
                'proposed': f"{proposed_val:.2f}",
                'change': f"{change:+.2f}",
                'impact': impact_category
            })
        
        # Create comparative visualizations
        # Financial comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create grouped bar chart for financial metrics
        labels = ['Net Present Value (÷10,000)', 'Benefit-Cost Ratio (×10)', 'ROI (%)']
        current_values = [
            roi_results['current_roi']['net_present_value'] / 10000,
            roi_results['current_roi']['benefit_cost_ratio'] * 10,
            roi_results['current_roi']['roi_percent']
        ]
        proposed_values = [
            roi_results['proposed_roi']['net_present_value'] / 10000,
            roi_results['proposed_roi']['benefit_cost_ratio'] * 10,
            roi_results['proposed_roi']['roi_percent']
        ]
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x - width/2, current_values, width, label='Current', color='#4285f4')
        ax.bar(x + width/2, proposed_values, width, label='Proposed', color='#34a853')
        
        ax.set_ylabel('Value (scaled)')
        ax.set_title('Financial Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        plt.tight_layout()
        financial_img = self._fig_to_base64(fig)
        
        # Create urban metrics comparison
        if urban_impacts:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            labels = [impact['metric'] for impact in urban_impacts]
            current_values = []
            proposed_values = []
            
            for impact in urban_impacts:
                # Convert string percentages to numbers
                current_str = impact['current']
                proposed_str = impact['proposed']
                
                if '%' in current_str:
                    current_val = float(current_str.replace('%', ''))
                else:
                    current_val = float(current_str)
                    
                if '%' in proposed_str:
                    proposed_val = float(proposed_str.replace('%', ''))
                else:
                    proposed_val = float(proposed_str)
                
                current_values.append(current_val)
                proposed_values.append(proposed_val)
            
            x = np.arange(len(labels))
            width = 0.35
            
            ax.bar(x - width/2, current_values, width, label='Current', color='#4285f4')
            ax.bar(x + width/2, proposed_values, width, label='Proposed', color='#34a853')
            
            ax.set_ylabel('Value')
            ax.set_title('Urban Metrics Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            
            plt.tight_layout()
            urban_img = self._fig_to_base64(fig)
        else:
            urban_img = None
        
        # Create impact sections
        impact_sections = [
            {
                'title': 'Financial Summary',
                'description': 'Comparison of financial metrics between current and proposed development.',
                'table': financial_summary
            },
            {
                'title': 'Cost Breakdown',
                'description': 'Breakdown of project costs by category.',
                'table': {
                    'headers': cost_breakdown['headers'],
                    'rows': [{'cells': [{'value': row[0], 'class': ''}, {'value': row[1], 'class': ''}]} for row in cost_breakdown['rows']]
                }
            },
            {
                'title': 'Benefits Breakdown',
                'description': 'Breakdown of project benefits by category.',
                'table': {
                    'headers': benefits_breakdown['headers'],
                    'rows': [{'cells': [{'value': row[0], 'class': ''}, 
                                      {'value': row[1], 'class': ''}, 
                                      {'value': row[2], 'class': ''}, 
                                      {'value': row[3], 'class': ''}]} for row in benefits_breakdown['rows']]
                }
            },
            {
                'title': 'Financial Comparison',
                'description': 'Visual comparison of key financial metrics.',
                'image': financial_img
            }
        ]
        
        if urban_img:
            impact_sections.append({
                'title': 'Urban Metrics Impact',
                'description': 'Analysis of how the project affects key urban metrics.',
                'image': urban_img,
                'table': {
                    'headers': ['Metric', 'Current', 'Proposed', 'Change', 'Impact'],
                    'rows': [{'cells': [
                        {'value': impact['metric'], 'class': ''},
                        {'value': impact['current'], 'class': ''},
                        {'value': impact['proposed'], 'class': ''},
                        {'value': impact['change'], 'class': 'better' if '+' in impact['change'] else 'worse'},
                        {'value': impact['impact'], 'class': ''}
                    ]} for impact in urban_impacts]
                }
            })
        
        # Create recommendations
        recommendations = []
        
        # Financial recommendations
        if roi_results['is_worthwhile']:
            recommendations.append({
                'title': 'Proceed with Project',
                'description': 'The project shows positive financial returns and should proceed.',
                'action_items': [
                    "Secure funding based on positive ROI projections.",
                    "Implement the project as proposed.",
                    "Consider phasing to optimize cash flow."
                ]
            })
        else:
            recommendations.append({
                'title': 'Revise Project Economics',
                'description': 'The project does not meet financial viability thresholds.',
                'action_items': [
                    "Reduce project costs while maintaining key benefits.",
                    "Seek additional funding sources or subsidies.",
                    "Consider a phased approach with lower initial investment.",
                    "Revise project scope to improve ROI."
                ]
            })
        
        # Urban impact recommendations
        for impact in urban_impacts:
            if 'Decline' in impact['impact'] and impact['metric'] == 'Green Coverage':
                recommendations.append({
                    'title': 'Improve Green Space Provision',
                    'description': f"The project reduces green coverage ({impact['change']}), which has negative impacts on urban quality and property values.",
                    'action_items': [
                        "Increase green space allocation in the project.",
                        "Add green roofs and vertical gardens to mitigate green space loss.",
                        "Ensure high-quality green spaces to compensate for reduced quantity."
                    ]
                })
            elif 'Decline' in impact['impact'] and impact['metric'] == 'Green Space Accessibility':
                recommendations.append({
                    'title': 'Improve Accessibility to Green Spaces',
                    'description': f"The project reduces accessibility to green spaces ({impact['change']}), which negatively impacts quality of life.",
                    'action_items': [
                        "Add pedestrian paths to improve green space access.",
                        "Distribute smaller green spaces throughout the development.",
                        "Remove barriers between residential areas and green spaces."
                    ]
                })
        
        # Render template
        template = self.jinja_env.get_template('comparison_report.html')
        rendered_html = template.render(
            report_title=report_title,
            generation_date=datetime.now().strftime('%Y-%m-%d %H:%M'),
            executive_summary=executive_summary,
            comparison_metrics=[
                {
                    'title': 'Net Present Value',
                    'current_value': f"${roi_results['current_roi']['net_present_value']:,.2f}",
                    'proposed_value': f"${roi_results['proposed_roi']['net_present_value']:,.2f}",
                    'change_description': f"${roi_results['npv_difference']:+,.2f}",
                    'is_improvement': roi_results['npv_difference'] > 0
                },
                {
                    'title': 'Benefit-Cost Ratio',
                    'current_value': f"{roi_results['current_roi']['benefit_cost_ratio']:.2f}",
                    'proposed_value': f"{roi_results['proposed_roi']['benefit_cost_ratio']:.2f}",
                    'change_description': f"{roi_results['benefit_cost_ratio_difference']:+.2f}",
                    'is_improvement': roi_results['benefit_cost_ratio_difference'] > 0
                },
                {
                    'title': 'Return on Investment',
                    'current_value': f"{roi_results['current_roi']['roi_percent']:.1f}%",
                    'proposed_value': f"{roi_results['proposed_roi']['roi_percent']:.1f}%",
                    'change_description': f"{roi_results['roi_difference']:+.1f}%",
                    'is_improvement': roi_results['roi_difference'] > 0
                },
                {
                    'title': 'Payback Period',
                    'current_value': f"{roi_results['current_roi']['payback_period_years']:.1f} years",
                    'proposed_value': f"{roi_results['proposed_roi']['payback_period_years']:.1f} years",
                    'change_description': f"{roi_results['current_roi']['payback_period_years'] - roi_results['proposed_roi']['payback_period_years']:+.1f} years",
                    'is_improvement': roi_results['proposed_roi']['payback_period_years'] < roi_results['current_roi']['payback_period_years']
                }
            ],
            impact_sections=impact_sections,
            recommendations=recommendations,
            footer_text="Analysis based on LiDAR point cloud data and financial modeling"
        )
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(rendered_html)
        
        return output_file