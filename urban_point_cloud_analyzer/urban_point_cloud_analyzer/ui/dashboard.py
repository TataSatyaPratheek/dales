# urban_point_cloud_analyzer/ui/dashboard.py
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
from pathlib import Path
import laspy
from typing import Dict, List, Optional, Union
import os
import sys

# Adjust path to import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from urban_point_cloud_analyzer.business.metrics.urban_metrics import calculate_urban_metrics

class UrbanDashboard:
    """
    Interactive dashboard for Urban Point Cloud Analyzer.
    """
    
    def __init__(self, results_dir: Union[str, Path]):
        """
        Initialize dashboard.
        
        Args:
            results_dir: Directory containing results
        """
        self.results_dir = Path(results_dir)
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Load available results
        self.load_results()
        
        # Create layout
        self.create_layout()
        
        # Set up callbacks
        self.setup_callbacks()
    
    def load_results(self):
        """Load available results from results directory."""
        self.point_clouds = list(self.results_dir.glob('*.la[sz]'))
        self.segmentation_files = list(self.results_dir.glob('*_segmented.la[sz]'))
        self.metrics_files = list(self.results_dir.glob('*_metrics.json'))
        
        # Load metrics
        self.metrics_data = {}
        for metrics_file in self.metrics_files:
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                self.metrics_data[metrics_file.stem] = metrics
            except Exception as e:
                print(f"Error loading metrics from {metrics_file}: {e}")
    
    def create_layout(self):
        """Create dashboard layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Urban Point Cloud Analyzer Dashboard"),
                    html.Hr(),
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Select Dataset"),
                    dcc.Dropdown(
                        id='dataset-dropdown',
                        options=[
                            {'label': f.stem, 'value': str(f)} 
                            for f in self.point_clouds
                        ],
                        value=str(self.point_clouds[0]) if self.point_clouds else None
                    ),
                    
                    html.Div(id='dataset-info', className='mt-3'),
                ], width=3),
                
                dbc.Col([
                    html.H4("3D Visualization"),
                    dcc.Graph(
                        id='point-cloud-3d',
                        figure=go.Figure(),
                        style={'height': '60vh'}
                    ),
                ], width=9)
            ]),
            
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Urban Metrics"),
                    dcc.Graph(
                        id='urban-metrics-chart',
                        figure=go.Figure(),
                        style={'height': '40vh'}
                    ),
                ], width=6),
                
                dbc.Col([
                    html.H4("Class Distribution"),
                    dcc.Graph(
                        id='class-distribution-chart',
                        figure=go.Figure(),
                        style={'height': '40vh'}
                    ),
                ], width=6)
            ]),
            
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Urban Analysis"),
                    html.Div(id='urban-analysis', className='p-3 border rounded')
                ], width=12)
            ]),
            
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Compare Datasets"),
                    dcc.Dropdown(
                        id='compare-dropdown',
                        options=[
                            {'label': f.stem, 'value': str(f)} 
                            for f in self.point_clouds
                        ],
                        value=[],
                        multi=True
                    ),
                    
                    dcc.Graph(
                        id='comparison-chart',
                        figure=go.Figure(),
                        style={'height': '50vh'}
                    ),
                ], width=12)
            ])
        ], fluid=True)
    
    def setup_callbacks(self):
        """Set up dashboard callbacks."""
        # Dataset selection callback
        @self.app.callback(
            [
                Output('dataset-info', 'children'),
                Output('point-cloud-3d', 'figure'),
                Output('urban-metrics-chart', 'figure'),
                Output('class-distribution-chart', 'figure'),
                Output('urban-analysis', 'children')
            ],
            [Input('dataset-dropdown', 'value')]
        )
        def update_dataset_info(selected_dataset):
            if not selected_dataset:
                return [
                    "No dataset selected",
                    go.Figure(),
                    go.Figure(),
                    go.Figure(),
                    "No dataset selected"
                ]
            
            try:
                # Load point cloud
                file_path = Path(selected_dataset)
                las = laspy.read(file_path)
                
                # Extract points
                points = np.vstack([las.x, las.y, las.z]).T
                
                # Extract labels if available
                labels = None
                if hasattr(las, 'classification'):
                    labels = np.array(las.classification)
                
                # Dataset info
                num_points = len(points)
                bounds = [
                    [np.min(points[:, 0]), np.max(points[:, 0])],
                    [np.min(points[:, 1]), np.max(points[:, 1])],
                    [np.min(points[:, 2]), np.max(points[:, 2])]
                ]
                
                dataset_info = html.Div([
                    html.P(f"File: {file_path.name}"),
                    html.P(f"Points: {num_points:,}"),
                    html.P(f"X Range: {bounds[0][0]:.2f} to {bounds[0][1]:.2f} m"),
                    html.P(f"Y Range: {bounds[1][0]:.2f} to {bounds[1][1]:.2f} m"),
                    html.P(f"Z Range: {bounds[2][0]:.2f} to {bounds[2][1]:.2f} m"),
                ])
                
                # 3D visualization
                colors = None
                if labels is not None:
                    # Color map
                    color_map = {
                        0: [0.7, 0.7, 0.7],  # Ground - Gray
                        1: [0.0, 0.8, 0.0],  # Vegetation - Green
                        2: [0.7, 0.4, 0.1],  # Buildings - Brown
                        3: [0.0, 0.0, 0.8],  # Water - Blue
                        4: [0.8, 0.0, 0.0],  # Car - Red
                        5: [1.0, 0.5, 0.0],  # Truck - Orange
                        6: [1.0, 1.0, 0.0],  # Powerline - Yellow
                        7: [0.7, 0.0, 0.7]   # Fence - Purple
                    }
                    
                    colors = []
                    for label in labels:
                        if label in color_map:
                            colors.append(f'rgb({int(color_map[label][0]*255)}, {int(color_map[label][1]*255)}, {int(color_map[label][2]*255)})')
                        else:
                            colors.append('rgb(128, 128, 128)')
                
                # Downsample for visualization
                max_points = 10000  # Limit for browser performance
                if len(points) > max_points:
                    indices = np.random.choice(len(points), max_points, replace=False)
                    points = points[indices]
                    colors = [colors[i] for i in indices] if colors else None
                
                fig_3d = go.Figure(data=[
                    go.Scatter3d(
                        x=points[:, 0],
                        y=points[:, 1],
                        z=points[:, 2],
                        mode='markers',
                        marker=dict(
                            size=2,
                            color=colors,
                            opacity=0.8
                        )
                    )
                ])
                
                fig_3d.update_layout(
                    scene=dict(
                        aspectmode='data'
                    ),
                    margin=dict(l=0, r=0, b=0, t=0)
                )
                
                # Urban metrics chart
                metrics_stem = file_path.stem + "_metrics"
                
                if metrics_stem in self.metrics_data:
                    metrics = self.metrics_data[metrics_stem]
                    
                    # Create metrics chart
                    metrics_chart_data = [
                        {'metric': 'Building Density', 'value': metrics.get('building_density', 0) * 100},
                        {'metric': 'Green Coverage', 'value': metrics.get('green_coverage', 0) * 100},
                        {'metric': 'Water Coverage', 'value': metrics.get('water_coverage', 0) * 100}
                    ]
                    
                    df_metrics = pd.DataFrame(metrics_chart_data)
                    
                    fig_metrics = px.bar(
                        df_metrics, 
                        x='metric', 
                        y='value',
                        title='Urban Metrics',
                        labels={'value': 'Percentage (%)', 'metric': 'Metric'},
                        color='metric',
                        color_discrete_map={
                            'Building Density': 'brown',
                            'Green Coverage': 'green',
                            'Water Coverage': 'blue'
                        }
                    )
                    
                    # Class distribution chart
                    if labels is not None:
                        class_names = {
                            0: 'Ground',
                            1: 'Vegetation',
                            2: 'Buildings',
                            3: 'Water',
                            4: 'Car',
                            5: 'Truck',
                            6: 'Powerline',
                            7: 'Fence'
                        }
                        
                        class_counts = {}
                        for i in range(8):
                            if i in labels:
                                class_counts[class_names.get(i, f'Class {i}')] = np.sum(labels == i)
                            else:
                                class_counts[class_names.get(i, f'Class {i}')] = 0
                        
                        df_classes = pd.DataFrame([
                            {'class': class_name, 'count': count}
                            for class_name, count in class_counts.items()
                        ])
                        
                        fig_classes = px.pie(
                            df_classes,
                            values='count',
                            names='class',
                            title='Class Distribution',
                            hole=0.3
                        )
                    else:
                        fig_classes = go.Figure()
                        fig_classes.update_layout(
                            title='No classification data available'
                        )
                    
                    # Urban analysis
                    urban_analysis = html.Div([
                        html.H5("Metrics Summary"),
                        html.Ul([
                            html.Li(f"Total Area: {metrics.get('total_area_m2', 0):.2f} m²"),
                            html.Li(f"Building Area: {metrics.get('building_area_m2', 0):.2f} m² ({metrics.get('building_density', 0)*100:.2f}%)"),
                            html.Li(f"Vegetation Area: {metrics.get('vegetation_area_m2', 0):.2f} m² ({metrics.get('green_coverage', 0)*100:.2f}%)"),
                            html.Li(f"Water Area: {metrics.get('water_area_m2', 0):.2f} m² ({metrics.get('water_coverage', 0)*100:.2f}%)")
                        ]),
                        
                        html.H5("Building Statistics"),
                        html.Ul([
                            html.Li(f"Mean Height: {metrics.get('mean_building_height', 0):.2f} m"),
                            html.Li(f"Max Height: {metrics.get('max_building_height', 0):.2f} m"),
                            html.Li(f"Min Height: {metrics.get('min_building_height', 0):.2f} m")
                        ]),
                        
                        html.H5("Vegetation Statistics"),
                        html.Ul([
                            html.Li(f"Mean Height: {metrics.get('mean_vegetation_height', 0):.2f} m"),
                            html.Li(f"Max Height: {metrics.get('max_vegetation_height', 0):.2f} m")
                        ]),
                        
                        html.H5("Vehicle Statistics"),
                        html.Ul([
                            html.Li(f"Vehicle Count: {metrics.get('vehicle_count', 0)}"),
                            html.Li(f"Vehicle Density: {metrics.get('vehicle_density', 0):.6f} vehicles/m²")
                        ]),
                        
                        html.H5("Urban Planning Analysis"),
                        html.Div([
                            dbc.Alert(
                                "Green space coverage is below recommended levels (15%). Consider adding more vegetation.",
                                color="warning"
                            ) if metrics.get('green_coverage', 0) < 0.15 else 
                            dbc.Alert(
                                "Green space coverage meets recommended levels.",
                                color="success"
                            ),
                            
                            dbc.Alert(
                                "Building density is very high. Consider adding more open spaces.",
                                color="warning"
                            ) if metrics.get('building_density', 0) > 0.7 else 
                            dbc.Alert(
                                "Building density is high, typical of urban centers.",
                                color="info"
                            ) if metrics.get('building_density', 0) > 0.5 else
                            dbc.Alert(
                                "Building density is moderate to low.",
                                color="success"
                            ),
                            
                            dbc.Alert(
                                "High vehicle density detected. Consider improving public transportation.",
                                color="warning"
                            ) if metrics.get('vehicle_density', 0) > 0.01 else None
                        ])
                    ])
                else:
                    fig_metrics = go.Figure()
                    fig_metrics.update_layout(
                        title='No metrics data available'
                    )
                    
                    fig_classes = go.Figure()
                    fig_classes.update_layout(
                        title='No classification data available'
                    )
                    
                    urban_analysis = "No urban metrics data available for this dataset"
                
                return [
                    dataset_info,
                    fig_3d,
                    fig_metrics,
                    fig_classes,
                    urban_analysis
                ]
                
            except Exception as e:
                print(f"Error processing dataset: {e}")
                return [
                    html.Div([
                        html.P(f"Error: {str(e)}"),
                        html.P(f"File: {selected_dataset}")
                    ]),
                    go.Figure(),
                    go.Figure(),
                    go.Figure(),
                    f"Error processing dataset: {str(e)}"
                ]
        
        # Comparison callback
        @self.app.callback(
            Output('comparison-chart', 'figure'),
            [Input('compare-dropdown', 'value')]
        )
        def update_comparison(selected_datasets):
            if not selected_datasets or len(selected_datasets) < 2:
                fig = go.Figure()
                fig.update_layout(
                    title='Select at least two datasets to compare'
                )
                return fig
            
            try:
                comparison_data = []
                
                for dataset_path in selected_datasets:
                    file_path = Path(dataset_path)
                    metrics_stem = file_path.stem + "_metrics"
                    
                    if metrics_stem in self.metrics_data:
                        metrics = self.metrics_data[metrics_stem]
                        
                        comparison_data.append({
                            'dataset': file_path.stem,
                            'building_density': metrics.get('building_density', 0) * 100,
                            'green_coverage': metrics.get('green_coverage', 0) * 100,
                            'water_coverage': metrics.get('water_coverage', 0) * 100,
                            'mean_building_height': metrics.get('mean_building_height', 0),
                            'vehicle_density': metrics.get('vehicle_density', 0) * 10000  # Scale for visibility
                        })
                
                if not comparison_data:
                    fig = go.Figure()
                    fig.update_layout(
                        title='No metrics data available for selected datasets'
                    )
                    return fig
                
                df_comparison = pd.DataFrame(comparison_data)
                
                fig = go.Figure()
                
                # Add traces for each metric
                metrics_to_compare = [
                    {'name': 'Building Density (%)', 'column': 'building_density', 'color': 'brown'},
                    {'name': 'Green Coverage (%)', 'column': 'green_coverage', 'color': 'green'},
                    {'name': 'Water Coverage (%)', 'column': 'water_coverage', 'color': 'blue'},
                    {'name': 'Mean Building Height (m)', 'column': 'mean_building_height', 'color': 'orange'},
                    {'name': 'Vehicle Density (per 10,000 m²)', 'column': 'vehicle_density', 'color': 'red'}
                ]
                
                for metric in metrics_to_compare:
                    fig.add_trace(
                        go.Bar(
                            x=df_comparison['dataset'],
                            y=df_comparison[metric['column']],
                            name=metric['name'],
                            marker_color=metric['color']
                        )
                    )
                
                fig.update_layout(
                    title='Urban Metrics Comparison',
                    barmode='group',
                    xaxis_title='Dataset',
                    yaxis_title='Value',
                    legend_title='Metric'
                )
                
                return fig
                
            except Exception as e:
                print(f"Error in comparison: {e}")
                fig = go.Figure()
                fig.update_layout(
                    title=f'Error in comparison: {str(e)}'
                )
                return fig
    
    def run_server(self, debug=False, port=8050):
        """Run the dashboard server."""
        self.app.run_server(debug=debug, port=port)