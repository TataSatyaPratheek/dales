# urban_point_cloud_analyzer/training/schedulers/__init__.py
import torch
import torch.optim as optim
from typing import Dict, List, Optional, Union

def get_scheduler(optimizer: torch.optim.Optimizer, config: Dict) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create a learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer to schedule
        config: Scheduler configuration
        
    Returns:
        Scheduler instance or None if no scheduler is specified
    """
    scheduler_name = config.get('name', None)
    if scheduler_name is None:
        return None
    
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'step':
        step_size = config.get('step_size', 30)
        gamma = config.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_name == 'multistep':
        milestones = config.get('milestones', [30, 60, 90])
        gamma = config.get('gamma', 0.1)
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    elif scheduler_name == 'cosine':
        num_epochs = config.get('epochs', 100)
        min_lr = config.get('min_lr', 0.00001)
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=min_lr
        )
    
    elif scheduler_name == 'plateau':
        patience = config.get('patience', 10)
        factor = config.get('factor', 0.1)
        min_lr = config.get('min_lr', 0.00001)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr
        )
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")