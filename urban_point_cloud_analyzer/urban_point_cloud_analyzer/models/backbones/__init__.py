# urban_point_cloud_analyzer/urban_point_cloud_analyzer/models/backbones/__init__.py
try:
    from .pointnet2_backbone import PointNet2Backbone, PointNet2Encoder
    CUDA_AVAILABLE = True
except ImportError:
    from .pointnet2_backbone_cpu import PointNet2BackboneCPU as PointNet2Backbone
    # Create a simple placeholder for the encoder
    import torch.nn as nn
    class PointNet2Encoder(nn.Module):
        def __init__(self, 
                     use_normals: bool = True,
                     in_channels: int = 3,
                     num_classes: int = 8,
                     use_checkpoint: bool = False,
                     use_height: bool = False):
            super(PointNet2Encoder, self).__init__()
            self.backbone = PointNet2Backbone(
                use_normals=use_normals,
                in_channels=in_channels,
                use_checkpoint=use_checkpoint,
                use_height=use_height
            )
            self.fc = nn.Conv1d(128, num_classes, 1)
            
        def forward(self, xyz, features=None):
            _, features = self.backbone(xyz, features)
            return self.fc(features)
    
    CUDA_AVAILABLE = False

__all__ = ['PointNet2Backbone', 'PointNet2Encoder']