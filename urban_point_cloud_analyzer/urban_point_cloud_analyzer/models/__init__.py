# urban_point_cloud_analyzer/models/__init__.py
from typing import Dict, Optional

import torch
import torch.nn as nn

def get_model(config: Dict, device: Optional[torch.device] = None) -> nn.Module:
    """
    Create a model based on configuration.
    
    Args:
        config: Model configuration
        device: Device to create the model on
        
    Returns:
        Model instance
    """
    model_type = config.get('type', 'pointnet2_seg')
    
    if model_type == 'pointnet2_seg':
        from urban_point_cloud_analyzer.models.segmentation.pointnet2_segmentation import PointNet2SegmentationModel
        
        model = PointNet2SegmentationModel(
            num_classes=config.get('segmentation', {}).get('num_classes', 8),
            use_normals=config.get('backbone', {}).get('use_normals', True),
            in_channels=3,  # Base channels for coordinates
            dropout=config.get('segmentation', {}).get('dropout', 0.5),
            use_height=True,
            use_checkpoint=True  # Enable gradient checkpointing for memory efficiency
        )
        
    elif model_type == 'kpconv':
        from urban_point_cloud_analyzer.models.segmentation.kpconv_segmentation import KPConvSegmentationModel
        
        model = KPConvSegmentationModel(
            num_classes=config.get('segmentation', {}).get('num_classes', 8),
            in_channels=config.get('backbone', {}).get('in_channels', 3),
            dropout=config.get('segmentation', {}).get('dropout', 0.5),
            use_checkpoint=True  # Enable gradient checkpointing for memory efficiency
        )
        
    elif model_type == 'randlanet':
        from urban_point_cloud_analyzer.models.segmentation.randlanet_segmentation import RandLANetSegmentationModel
        
        model = RandLANetSegmentationModel(
            num_classes=config.get('segmentation', {}).get('num_classes', 8),
            in_channels=config.get('backbone', {}).get('in_channels', 3),
            dropout=config.get('segmentation', {}).get('dropout', 0.5),
            use_checkpoint=True  # Enable gradient checkpointing for memory efficiency
        )
        
    elif model_type == 'spvcnn':
        from urban_point_cloud_analyzer.models.segmentation.spvcnn_segmentation import SPVCNNSegmentationModel
        
        model = SPVCNNSegmentationModel(
            num_classes=config.get('segmentation', {}).get('num_classes', 8),
            in_channels=config.get('backbone', {}).get('in_channels', 3),
            feature_dims=config.get('backbone', {}).get('feature_dims', [32, 64, 128, 256]),
            voxel_size=config.get('data', {}).get('voxel_size', 0.05),
            dropout=config.get('segmentation', {}).get('dropout', 0.5),
            use_checkpoint=True  # Enable gradient checkpointing for memory efficiency
        )
    
    elif model_type == 'ensemble':
        from urban_point_cloud_analyzer.models.ensemble import create_ensemble
        
        # Create base models first
        base_model_configs = config.get('ensemble', {}).get('base_models', [])
        base_models = []
        
        for base_config in base_model_configs:
            base_model = get_model(base_config, device)
            base_models.append(base_model)
        
        # Create ensemble with base models
        model = create_ensemble(config.get('ensemble', {}), base_models)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move to device if specified
    if device is not None:
        model = model.to(device)
    
    return model