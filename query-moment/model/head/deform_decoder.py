import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention

class DeformableDecoder3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, hs):
        pass

class DeformableDecoder1D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, hs):
        pass

class MultiScaleDeformableDecoder(nn.Module):
    def __init__(self, d_model=512, num_head=16, num_query=100, num_points=4, num_level=4) -> None:
        super().__init__()
        self.ms_defm_attn = MultiScaleDeformableAttention(d_model, num_head, num_level, num_points, )
    
    def forward(self, hs):
        