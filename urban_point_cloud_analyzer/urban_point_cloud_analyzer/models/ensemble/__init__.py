# urban_point_cloud_analyzer/models/ensemble/__init__.py
from .model_ensemble import SegmentationEnsemble, MultiScaleEnsemble, SpecialistEnsemble
from typing import Dict, List, Optional, Tuple, Union
import torch.nn as nn

def create_ensemble(config: Dict, base_models: Optional[List[nn.Module]] = None) -> nn.Module:
    """
    Create an ensemble model based on configuration.
    
    Args:
        config: Ensemble configuration
        base_models: Optional list of base models
        
    Returns:
        Ensemble model
    """
    ensemble_type = config.get('type', 'basic')
    
    if ensemble_type == 'basic':
        # Simple ensemble of multiple models
        assert base_models is not None, "Base models must be provided for basic ensemble"
        weights = config.get('weights', None)
        
        return SegmentationEnsemble(
            models=base_models,
            weights=weights
        )
    
    elif ensemble_type == 'multi_scale':
        # Multi-scale ensemble
        assert len(base_models) == 1, "Exactly one base model must be provided for multi-scale ensemble"
        scales = config.get('scales', [0.5, 1.0, 2.0])
        weights = config.get('weights', None)
        
        return MultiScaleEnsemble(
            base_model=base_models[0],
            scales=scales,
            weights=weights
        )
    
    elif ensemble_type == 'specialist':
        # Specialist ensemble
        assert base_models is not None, "Base models must be provided for specialist ensemble"
        class_assignments = config.get('class_assignments', None)
        assert class_assignments is not None, "Class assignments must be provided for specialist ensemble"
        num_classes = config.get('num_classes', 8)
        
        return SpecialistEnsemble(
            models=base_models,
            class_assignments=class_assignments,
            num_classes=num_classes
        )
    
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")

__all__ = ['SegmentationEnsemble', 'MultiScaleEnsemble', 'SpecialistEnsemble', 'create_ensemble']