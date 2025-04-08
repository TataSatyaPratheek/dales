# urban_point_cloud_analyzer/training/optimizers/__init__.py
import torch
import torch.optim as optim
from typing import Dict, List, Optional, Union

def get_optimizer(model_params, config: Dict) -> torch.optim.Optimizer:
    """
    Create an optimizer based on configuration.
    
    Args:
        model_params: Model parameters to optimize
        config: Optimizer configuration
        
    Returns:
        Optimizer instance
    """
    optimizer_name = config.get('name', 'adam').lower()
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0.0001)
    
    if optimizer_name == 'adam':
        return optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config.get('momentum', 0.9)
        return optim.SGD(model_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")