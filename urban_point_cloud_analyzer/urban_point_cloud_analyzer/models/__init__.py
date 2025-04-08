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
    # TODO: Add support for other model types (kpconv, randlanet, spvcnn)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move to device if specified
    if device is not None:
        model = model.to(device)
    
    return model