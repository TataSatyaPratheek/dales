# urban_point_cloud_analyzer/business/metrics/urban_metrics.py
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial import ConvexHull, Delaunay
import open3d as o3d
from shapely.geometry import Polygon


def calculate_convex_hull_area(points_2d: np.ndarray) -> float:
    """
    Calculate the area of the convex hull of 2D points.
    
    Args:
        points_2d: (N, 2) array of 2D points
        
    Returns:
        Area of the convex hull
    """
    if len(points_2d) < 3:
        return 0.0
    
    hull = ConvexHull(points_2d)
    return hull.volume


def calculate_alpha_shape_area(points_2d: np.ndarray, alpha: float = 0.5) -> float:
    """
    Calculate the area of the alpha shape of 2D points.
    
    Args:
        points_2d: (N, 2) array of 2D points
        alpha: Alpha parameter for the alpha shape
        
    Returns:
        Area of the alpha shape
    """
    if len(points_2d) < 3:
        return 0.0
    
    # Compute Delaunay triangulation
    tri = Delaunay(points_2d)
    
    # Get triangles
    triangles = points_2d[tri.simplices]
    
    # Compute circumradius of each triangle
    def circumradius(triangle):
        a, b, c = triangle
        ab = np.linalg.norm(a - b)
        bc = np.linalg.norm(b - c)
        ca = np.linalg.norm(c - a)
        s = (ab + bc + ca) / 2.0
        area = np.sqrt(s * (s - ab) * (s - bc) * (s - ca))
        if area < 1e-10:
            return float('inf')
        return (ab * bc * ca) / (4.0 * area)
    
    # Calculate area of alpha shape
    area = 0.0
    for triangle in triangles:
        if circumradius(triangle) < 1.0 / alpha:
            # Add triangle area
            a, b, c = triangle
            area += 0.5 * np.abs(np.cross(b - a, c - a))
    
    return area


def calculate_urban_metrics(point_cloud: np.ndarray, segmentation_labels: np.ndarray) -> Dict:
    """
    Calculate urban metrics from point cloud and segmentation.
    
    Args:
        point_cloud: (N, 3+) array of point coordinates and features
        segmentation_labels: (N,) array of class labels
        
    Returns:
        Dict of urban metrics
    """
    metrics = {}
    
    # Class definitions (from DALES dataset)
    # 0: Ground, 1: Vegetation, 2: Buildings, 3: Water, 4: Car, 5: Truck, 6: Powerline, 7: Fence
    
    # Get total area using convex hull
    total_area = calculate_convex_hull_area(point_cloud[:, 0:2])
    metrics['total_area_m2'] = total_area
    
    # Building density
    building_mask = segmentation_labels == 2  # Building class
    if np.sum(building_mask) > 0:
        building_points = point_cloud[building_mask]
        building_area = calculate_alpha_shape_area(building_points[:, 0:2])
        metrics['building_density'] = building_area / total_area
        metrics['building_area_m2'] = building_area
    else:
        metrics['building_density'] = 0.0
        metrics['building_area_m2'] = 0.0
    
    # Green coverage
    vegetation_mask = segmentation_labels == 1  # Vegetation class
    if np.sum(vegetation_mask) > 0:
        vegetation_points = point_cloud[vegetation_mask]
        vegetation_area = calculate_alpha_shape_area(vegetation_points[:, 0:2])
        metrics['green_coverage'] = vegetation_area / total_area
        metrics['vegetation_area_m2'] = vegetation_area
    else:
        metrics['green_coverage'] = 0.0
        metrics['vegetation_area_m2'] = 0.0
    
    # Water coverage
    water_mask = segmentation_labels == 3  # Water class
    if np.sum(water_mask) > 0:
        water_points = point_cloud[water_mask]
        water_area = calculate_alpha_shape_area(water_points[:, 0:2])
        metrics['water_coverage'] = water_area / total_area
        metrics['water_area_m2'] = water_area
    else:
        metrics['water_coverage'] = 0.0
        metrics['water_area_m2'] = 0.0
    
    # Vehicle density
    car_mask = segmentation_labels == 4  # Car class
    truck_mask = segmentation_labels == 5  # Truck class
    
    # Count vehicles by clustering points
    vehicle_count = 0
    if np.sum(car_mask) > 0 or np.sum(truck_mask) > 0:
        vehicle_mask = np.logical_or(car_mask, truck_mask)
        vehicle_points = point_cloud[vehicle_mask]
        
        # Cluster vehicle points using DBSCAN
        vehicle_pcd = o3d.geometry.PointCloud()
        vehicle_pcd.points = o3d.utility.Vector3dVector(vehicle_points[:, :3])
        
        # DBSCAN clustering
        eps = 0.5  # 0.5 meters
        min_points = 10  # minimum points to form a cluster
        labels = np.array(vehicle_pcd.cluster_dbscan(eps=eps, min_points=min_points))
        
        # Count unique clusters (excluding noise with label -1)
        if len(labels) > 0:
            vehicle_count = len(np.unique(labels[labels >= 0]))
        
        metrics['vehicle_count'] = vehicle_count
        metrics['vehicle_density'] = vehicle_count / total_area
    else:
        metrics['vehicle_count'] = 0
        metrics['vehicle_density'] = 0.0
    
    # Building height statistics
    if np.sum(building_mask) > 0:
        building_heights = point_cloud[building_mask, 2]  # Z-coordinates
        metrics['mean_building_height'] = np.mean(building_heights)
        metrics['max_building_height'] = np.max(building_heights)
        metrics['min_building_height'] = np.min(building_heights)
    else:
        metrics['mean_building_height'] = 0.0
        metrics['max_building_height'] = 0.0
        metrics['min_building_height'] = 0.0
    
    # Vegetation height statistics
    if np.sum(vegetation_mask) > 0:
        vegetation_heights = point_cloud[vegetation_mask, 2]  # Z-coordinates
        metrics['mean_vegetation_height'] = np.mean(vegetation_heights)
        metrics['max_vegetation_height'] = np.max(vegetation_heights)
    else:
        metrics['mean_vegetation_height'] = 0.0
        metrics['max_vegetation_height'] = 0.0
    
    # Road network analysis (assuming ground is road)
    ground_mask = segmentation_labels == 0  # Ground class
    
    # Additional metrics could be calculated here
    
    return metrics


def generate_urban_analysis_report(metrics: Dict) -> str:
    """
    Generate an urban analysis report from metrics.
    
    Args:
        metrics: Dictionary of urban metrics
        
    Returns:
        String report with insights
    """
    report = "Urban Analysis Report\n"
    report += "=====================\n\n"
    
    # Area statistics
    report += "Area Statistics:\n"
    report += f"  Total Area: {metrics['total_area_m2']:.2f} m²\n"
    report += f"  Building Area: {metrics['building_area_m2']:.2f} m² ({metrics['building_density']*100:.2f}%)\n"
    report += f"  Vegetation Area: {metrics['vegetation_area_m2']:.2f} m² ({metrics['green_coverage']*100:.2f}%)\n"
    report += f"  Water Area: {metrics['water_area_m2']:.2f} m² ({metrics['water_coverage']*100:.2f}%)\n\n"
    
    # Building statistics
    report += "Building Statistics:\n"
    report += f"  Mean Height: {metrics['mean_building_height']:.2f} m\n"
    report += f"  Maximum Height: {metrics['max_building_height']:.2f} m\n"
    report += f"  Minimum Height: {metrics['min_building_height']:.2f} m\n\n"
    
    # Vegetation statistics
    report += "Vegetation Statistics:\n"
    report += f"  Mean Height: {metrics['mean_vegetation_height']:.2f} m\n"
    report += f"  Maximum Height: {metrics['max_vegetation_height']:.2f} m\n\n"
    
    # Vehicle statistics
    report += "Vehicle Statistics:\n"
    report += f"  Total Count: {metrics['vehicle_count']}\n"
    report += f"  Density: {metrics['vehicle_density']:.6f} vehicles/m²\n\n"
    
    # Urban planning insights
    report += "Urban Planning Insights:\n"
    
    # Green space ratio analysis
    if metrics['green_coverage'] < 0.15:
        report += "  ⚠️ Green space coverage is below recommended levels (15%). Consider adding more vegetation.\n"
    else:
        report += "  ✓ Green space coverage meets recommended levels.\n"
    
    # Building density analysis
    if metrics['building_density'] > 0.7:
        report += "  ⚠️ Building density is very high. Consider adding more open spaces.\n"
    elif metrics['building_density'] > 0.5:
        report += "  ℹ️ Building density is high, typical of urban centers.\n"
    else:
        report += "  ✓ Building density is moderate to low.\n"
    
    # Vehicle density analysis
    if metrics['vehicle_density'] > 0.01:
        report += "  ⚠️ High vehicle density detected. Consider improving public transportation.\n"
    
    return report