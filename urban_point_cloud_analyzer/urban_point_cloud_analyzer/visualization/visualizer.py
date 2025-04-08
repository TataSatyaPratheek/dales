# urban_point_cloud_analyzer/visualization/visualizer.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
import open3d as o3d
from pathlib import Path

class PointCloudVisualizer:
    """
    Visualize point clouds with segmentation.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config if config is not None else {}
        
        # Default class colors
        self.class_colors = {
            0: [0.7, 0.7, 0.7],  # Ground - Gray
            1: [0.0, 0.8, 0.0],  # Vegetation - Green
            2: [0.7, 0.4, 0.1],  # Buildings - Brown
            3: [0.0, 0.0, 0.8],  # Water - Blue
            4: [0.8, 0.0, 0.0],  # Car - Red
            5: [1.0, 0.5, 0.0],  # Truck - Orange
            6: [1.0, 1.0, 0.0],  # Powerline - Yellow
            7: [0.7, 0.0, 0.7]   # Fence - Purple
        }
        
        # Override with config if provided
        if config is not None and 'visualization' in config:
            if 'class_colors' in config['visualization']:
                for class_id, color_name in config['visualization']['class_colors'].items():
                    if isinstance(color_name, list) and len(color_name) == 3:
                        self.class_colors[int(class_id)] = color_name
    
    def visualize_point_cloud(self, 
                             points: np.ndarray, 
                             labels: Optional[np.ndarray] = None,
                             predictions: Optional[np.ndarray] = None,
                             view_mode: str = 'predictions',
                             filename: Optional[str] = None) -> None:
        """
        Visualize point cloud with segmentation.
        
        Args:
            points: (N, 3+) array of point coordinates and features
            labels: Optional (N,) array of ground truth labels
            predictions: Optional (N,) array of predicted labels
            view_mode: 'predictions', 'labels', 'error', or 'side_by_side'
            filename: Optional filename to save visualization
        """
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # Determine colors based on view mode
        if view_mode == 'predictions' and predictions is not None:
            # Color by predictions
            colors = np.zeros((len(points), 3))
            for i in range(len(points)):
                label = predictions[i]
                if label in self.class_colors:
                    colors[i] = self.class_colors[label]
            
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        elif view_mode == 'labels' and labels is not None:
            # Color by ground truth
            colors = np.zeros((len(points), 3))
            for i in range(len(points)):
                label = labels[i]
                if label in self.class_colors:
                    colors[i] = self.class_colors[label]
            
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        elif view_mode == 'error' and labels is not None and predictions is not None:
            # Color by error (red = error, green = correct)
            colors = np.zeros((len(points), 3))
            for i in range(len(points)):
                if labels[i] == predictions[i]:
                    colors[i] = [0.0, 1.0, 0.0]  # Green for correct
                else:
                    colors[i] = [1.0, 0.0, 0.0]  # Red for error
            
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        elif view_mode == 'side_by_side' and labels is not None and predictions is not None:
            # Create two point clouds for side-by-side view
            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(points[:, :3])
            
            pcd_pred = o3d.geometry.PointCloud()
            pcd_pred.points = o3d.utility.Vector3dVector(points[:, :3])
            
            # Color ground truth
            colors_gt = np.zeros((len(points), 3))
            for i in range(len(points)):
                label = labels[i]
                if label in self.class_colors:
                    colors_gt[i] = self.class_colors[label]
            
            # Color predictions
            colors_pred = np.zeros((len(points), 3))
            for i in range(len(points)):
                label = predictions[i]
                if label in self.class_colors:
                    colors_pred[i] = self.class_colors[label]
            
            pcd_gt.colors = o3d.utility.Vector3dVector(colors_gt)
            pcd_pred.colors = o3d.utility.Vector3dVector(colors_pred)
            
            # Move ground truth to the left and predictions to the right
            max_extent = np.max(points[:, 0])
            min_extent = np.min(points[:, 0])
            offset = max_extent - min_extent + 5  # 5 meter gap
            
            pcd_gt.translate([-offset/2, 0, 0])
            pcd_pred.translate([offset/2, 0, 0])
            
            # Visualize both point clouds
            o3d.visualization.draw_geometries([pcd_gt, pcd_pred], 
                                           window_name="Ground Truth vs Predictions",
                                           width=1600, height=900)
            
            # Save if filename is provided
            if filename:
                img = o3d.visualization.capture_screen_float_buffer(True)
                plt.imshow(np.asarray(img))
                plt.axis('off')
                plt.savefig(filename, bbox_inches='tight')
            
            return
        
        # Regular visualization
        o3d.visualization.draw_geometries([pcd],
                                       window_name="Point Cloud Visualization",
                                       width=1200, height=900)
        
        # Save if filename is provided
        if filename:
            img = o3d.visualization.capture_screen_float_buffer(True)
            plt.imshow(np.asarray(img))
            plt.axis('off')
            plt.savefig(filename, bbox_inches='tight')
    
    def visualize_metrics(self, metrics: Dict, filename: Optional[str] = None) -> None:
        """
        Visualize segmentation metrics.
        
        Args:
            metrics: Dictionary of metrics
            filename: Optional filename to save visualization
        """
        # Class names
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
        
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Segmentation Metrics', fontsize=16)
        
        # IoU plot
        ax = axs[0, 0]
        if 'class_iou' in metrics:
            class_ids = sorted(metrics['class_iou'].keys())
            class_labels = [class_names.get(i, f'Class {i}') for i in class_ids]
            iou_values = [metrics['class_iou'][i] for i in class_ids]
            
            ax.bar(class_labels, iou_values, color='skyblue')
            ax.set_title('IoU by Class')
            ax.set_ylim(0, 1)
            ax.set_ylabel('IoU')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add mean IoU line
            if 'mean_iou' in metrics:
                ax.axhline(y=metrics['mean_iou'], color='r', linestyle='-', label=f'Mean IoU: {metrics["mean_iou"]:.4f}')
                ax.legend()
        
        # Precision plot
        ax = axs[0, 1]
        if 'precision' in metrics:
            class_ids = sorted(metrics['precision'].keys())
            class_labels = [class_names.get(i, f'Class {i}') for i in class_ids]
            precision_values = [metrics['precision'][i] for i in class_ids]
            
            ax.bar(class_labels, precision_values, color='lightgreen')
            ax.set_title('Precision by Class')
            ax.set_ylim(0, 1)
            ax.set_ylabel('Precision')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Recall plot
        ax = axs[1, 0]
        if 'recall' in metrics:
            class_ids = sorted(metrics['recall'].keys())
            class_labels = [class_names.get(i, f'Class {i}') for i in class_ids]
            recall_values = [metrics['recall'][i] for i in class_ids]
            
            ax.bar(class_labels, recall_values, color='coral')
            ax.set_title('Recall by Class')
            ax.set_ylim(0, 1)
            ax.set_ylabel('Recall')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # F1 plot
        ax = axs[1, 1]
        if 'f1' in metrics:
            class_ids = sorted(metrics['f1'].keys())
            class_labels = [class_names.get(i, f'Class {i}') for i in class_ids]
            f1_values = [metrics['f1'][i] for i in class_ids]
            
            ax.bar(class_labels, f1_values, color='mediumpurple')
            ax.set_title('F1 Score by Class')
            ax.set_ylim(0, 1)
            ax.set_ylabel('F1 Score')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show
        if filename:
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
        else:
            plt.show()