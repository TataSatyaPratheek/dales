# urban_point_cloud_analyzer/business/metrics/advanced_metrics.py
import numpy as np
import networkx as nx
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from scipy.ndimage import binary_erosion, binary_dilation
from typing import Dict, List, Optional, Tuple, Union
import open3d as o3d
from shapely.geometry import Polygon, LineString, Point
import math

def calculate_road_connectivity(points: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Calculate road network connectivity metrics.
    
    Args:
        points: (N, 3+) array of point coordinates
        labels: (N,) array of point labels
        
    Returns:
        Dictionary of road connectivity metrics
    """
    # Ground class is typically roads and pavements
    road_mask = labels == 0
    road_points = points[road_mask]
    
    # Default return value if not enough points
    default_return = {
        'connectivity_score': 0.0,
        'avg_road_width': 0.0,
        'road_density': 0.0,
        'intersection_count': 0
    }
    
    if len(road_points) < 100:
        return default_return
    
    # Project to 2D for road analysis
    road_points_2d = road_points[:, :2]
    
    # Calculate road area
    road_hull = ConvexHull(road_points_2d)
    road_area = road_hull.volume  # In 2D, volume is area
    
    # Create a 2D grid representation of the roads
    min_x, min_y = np.min(road_points_2d, axis=0)
    max_x, max_y = np.max(road_points_2d, axis=0)
    
    # Define grid resolution (1m)
    resolution = 1.0
    grid_size_x = int((max_x - min_x) / resolution) + 1
    grid_size_y = int((max_y - min_y) / resolution) + 1
    
    # Create empty grid
    grid = np.zeros((grid_size_x, grid_size_y), dtype=bool)
    
    # Fill grid with road points
    for point in road_points_2d:
        grid_x = int((point[0] - min_x) / resolution)
        grid_y = int((point[1] - min_y) / resolution)
        if 0 <= grid_x < grid_size_x and 0 <= grid_y < grid_size_y:
            grid[grid_x, grid_y] = True
    
    # Apply morphological operations to clean the grid
    grid = binary_dilation(grid, iterations=1)
    grid = binary_erosion(grid, iterations=2)
    grid = binary_dilation(grid, iterations=1)
    
    # Extract road network graph
    # Create a graph from the grid
    G = nx.Graph()
    
    # Add nodes for road cells
    for i in range(grid_size_x):
        for j in range(grid_size_y):
            if grid[i, j]:
                G.add_node((i, j))
    
    # Add edges between adjacent road cells
    for i in range(grid_size_x):
        for j in range(grid_size_y):
            if grid[i, j]:
                for di, dj in [(0, 1), (1, 0), (1, 1), (-1, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid_size_x and 0 <= nj < grid_size_y and grid[ni, nj]:
                        G.add_edge((i, j), (ni, nj))
    
    # Calculate connectivity metrics
    connectivity_metrics = {}
    
    # Count intersections (nodes with >2 edges)
    intersection_nodes = [node for node, degree in G.degree() if degree > 2]
    intersection_count = len(intersection_nodes)
    
    # Calculate average road width
    # Use distance transform to estimate width
    from scipy.ndimage import distance_transform_edt
    distance_map = distance_transform_edt(grid)
    
    # When calculating means, check if array is empty first
    if len(distance_map[grid]) > 0:
        road_width = np.mean(distance_map[grid]) * 2 * resolution
    else:
        road_width = 0.0
    
    # Calculate road density
    total_area = (max_x - min_x) * (max_y - min_y)
    road_density = road_area / total_area
    
    # Calculate connectivity score
    # Based on alpha index from transport geography
    v = len(G)
    e = len(G.edges())
    if v > 2:
        alpha = (e - v + 1) / (2*v - 5)  # Alpha index
        alpha = max(0, min(1, alpha))  # Normalize to [0,1]
    else:
        alpha = 0
    
    connectivity_metrics = {
        'connectivity_score': alpha,
        'avg_road_width': road_width,
        'road_density': road_density,
        'intersection_count': intersection_count
    }
    
    return connectivity_metrics

def calculate_accessibility_metrics(points: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Calculate accessibility metrics for urban planning.
    
    Args:
        points: (N, 3+) array of point coordinates
        labels: (N,) array of point labels
        
    Returns:
        Dictionary of accessibility metrics
    """
    # Define class masks
    building_mask = labels == 2  # Buildings
    road_mask = labels == 0  # Ground/roads
    veg_mask = labels == 1  # Vegetation
    
    # Extract points
    building_points = points[building_mask]
    road_points = points[road_mask]
    veg_points = points[veg_mask]
    
    if len(building_points) < 10 or len(road_points) < 10:
        return {
            'building_to_road_accessibility': 0.0,
            'green_space_accessibility': 0.0,
            'avg_distance_to_road': 0.0,
            'avg_distance_to_green': 0.0
        }
    
    # Project to 2D for accessibility analysis
    building_points_2d = building_points[:, :2]
    road_points_2d = road_points[:, :2]
    
    # Create KD-trees for efficient distance calculations
    road_kdtree = cKDTree(road_points_2d)
    
    # Calculate distances from buildings to roads
    distances_to_road = []
    for point in building_points_2d:
        dist, _ = road_kdtree.query(point)
        distances_to_road.append(dist)
    
    # Calculate average distance to road
    avg_distance_to_road = np.mean(distances_to_road) if distances_to_road else 0.0
    
    # Calculate building to road accessibility score
    # Scale distances to [0, 1] range, where 1 is good (close)
    building_to_road_accessibility = np.exp(-avg_distance_to_road / 50.0)
    
    # Calculate green space accessibility if vegetation exists
    if len(veg_points) > 10:
        veg_points_2d = veg_points[:, :2]
        veg_kdtree = cKDTree(veg_points_2d)
        
        # Calculate distances from buildings to green spaces
        distances_to_green = []
        for point in building_points_2d:
            dist, _ = veg_kdtree.query(point)
            distances_to_green.append(dist)
        
        # Calculate average distance to green space
        if len(distances_to_green) > 0:
            avg_distance_to_green = np.mean(distances_to_green)
        else:
            avg_distance_to_green = float('inf')
        
        # Calculate green space accessibility score
        green_space_accessibility = np.exp(-avg_distance_to_green / 100.0)
    else:
        avg_distance_to_green = float('inf')
        green_space_accessibility = 0.0
    
    accessibility_metrics = {
        'building_to_road_accessibility': building_to_road_accessibility,
        'green_space_accessibility': green_space_accessibility,
        'avg_distance_to_road': avg_distance_to_road,
        'avg_distance_to_green': avg_distance_to_green
    }
    
    return accessibility_metrics

def calculate_urban_density_metrics(points: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Calculate urban density metrics.
    
    Args:
        points: (N, 3+) array of point coordinates
        labels: (N,) array of point labels
        
    Returns:
        Dictionary of urban density metrics
    """
    # Define class masks
    building_mask = labels == 2  # Buildings
    ground_mask = labels == 0  # Ground
    
    # Calculate total area
    all_points_2d = points[:, :2]
    hull = ConvexHull(all_points_2d)
    total_area = hull.volume  # In 2D, volume is actually area
    
    # Extract building points
    building_points = points[building_mask]
    
    if len(building_points) < 10:
        return {
            'floor_area_ratio': 0.0,
            'building_coverage_ratio': 0.0,
            'building_height_variation': 0.0,
            'urban_compactness': 0.0
        }
    
    # Calculate building coverage area
    building_points_2d = building_points[:, :2]
    try:
        building_hull = ConvexHull(building_points_2d)
        building_area = building_hull.volume
    except Exception:
        # If buildings are too sparse for a convex hull
        building_area = 0.0
    
    # Calculate building coverage ratio
    building_coverage_ratio = building_area / total_area
    
    # Calculate building heights
    building_heights = building_points[:, 2]
    mean_height = np.mean(building_heights)
    
    # Estimate total floor area (assuming 3m per floor)
    floor_height = 3.0  # meters
    estimated_floors = np.ceil(building_heights / floor_height)
    total_floor_area = building_area * np.mean(estimated_floors)
    
    # Calculate Floor Area Ratio (FAR)
    floor_area_ratio = total_floor_area / total_area
    
    # Calculate height variation (coefficient of variation)
    height_std = np.std(building_heights)
    if mean_height > 0:
        height_variation = height_std / mean_height
    else:
        height_variation = 0.0
    
    # Calculate urban compactness
    # Ratio of building volume to surface area
    building_volume = building_area * mean_height
    building_perimeter = building_hull.simplices.shape[0] * np.mean(
        np.sqrt(np.sum(np.diff(building_hull.points[building_hull.simplices], axis=1)**2, axis=2))
    )
    if building_perimeter > 0:
        urban_compactness = building_volume / (building_perimeter * mean_height)
    else:
        urban_compactness = 0.0
    
    density_metrics = {
        'floor_area_ratio': floor_area_ratio,
        'building_coverage_ratio': building_coverage_ratio,
        'building_height_variation': height_variation,
        'urban_compactness': urban_compactness
    }
    
    return density_metrics

def calculate_environmental_metrics(points: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Calculate environmental metrics.
    
    Args:
        points: (N, 3+) array of point coordinates
        labels: (N,) array of point labels
        
    Returns:
        Dictionary of environmental metrics
    """
    # Define class masks
    building_mask = labels == 2  # Buildings
    veg_mask = labels == 1  # Vegetation
    water_mask = labels == 3  # Water
    
    # Calculate total area
    all_points_2d = points[:, :2]
    hull = ConvexHull(all_points_2d)
    total_area = hull.volume  # In 2D, volume is actually area
    
    # Extract points by class
    building_points = points[building_mask]
    veg_points = points[veg_mask]
    water_points = points[water_mask]
    
    # Calculate green coverage
    if len(veg_points) > 10:
        veg_points_2d = veg_points[:, :2]
        try:
            veg_hull = ConvexHull(veg_points_2d)
            veg_area = veg_hull.volume
        except Exception:
            # If vegetation is too sparse for a convex hull
            veg_area = 0.0
    else:
        veg_area = 0.0
    
    # Calculate water coverage
    if len(water_points) > 10:
        water_points_2d = water_points[:, :2]
        try:
            water_hull = ConvexHull(water_points_2d)
            water_area = water_hull.volume
        except Exception:
            # If water is too sparse for a convex hull
            water_area = 0.0
    else:
        water_area = 0.0
    
    # Calculate permeable surface ratio
    permeable_area = veg_area + water_area
    permeable_ratio = permeable_area / total_area
    
    # Calculate green space fragmentation
    if len(veg_points) > 10:
        # Cluster vegetation into patches
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(veg_points[:, :3])
        
        # Use DBSCAN to find vegetation patches
        eps = 5.0  # 5 meters
        min_points = 10
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
        
        # Count number of patches
        if len(labels) > 0:
            unique_patches = np.unique(labels[labels >= 0])
            num_patches = len(unique_patches)
        else:
            num_patches = 0
        
        # Calculate fragmentation index
        # Higher value means more fragmentation
        if veg_area > 0:
            fragmentation_index = num_patches / (veg_area / 10000)  # patches per hectare
        else:
            fragmentation_index = 0.0
    else:
        fragmentation_index = 0.0
    
    # Calculate solar exposure (simplified)
    if len(building_points) > 10:
        # Get average building height
        avg_building_height = np.mean(building_points[:, 2])
        
        # Calculate building density
        building_points_2d = building_points[:, :2]
        try:
            building_hull = ConvexHull(building_points_2d)
            building_area = building_hull.volume
        except Exception:
            building_area = 0.0
        
        building_density = building_area / total_area
        
        # Simplified solar exposure index
        # Lower density and height means more solar exposure
        solar_exposure = 1.0 - (building_density * avg_building_height / 100.0)
        solar_exposure = max(0.0, min(1.0, solar_exposure))
    else:
        solar_exposure = 1.0
    
    environmental_metrics = {
        'permeable_surface_ratio': permeable_ratio,
        'green_space_fragmentation': fragmentation_index,
        'solar_exposure': solar_exposure,
        'green_to_built_ratio': veg_area / (building_area + 1e-10)
    }
    
    return environmental_metrics