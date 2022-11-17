import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from einops import reduce, rearrange, repeat
from kn_util.nn import clones

class MultiScaleDeformableDecoderLayer(nn.Module):
    def __init__(self,
                 d_model, num_head, num_level, num_points, dropout
        ) -> None:
        super().__init__()
        self.ms_defm_attn = MultiScaleDeformableAttention(
            embed_dims=d_model,
            num_heads=num_head,
            num_levels=num_level,
            num_points=num_points,
            dropout=dropout,
            batch_first=True,
        )
    
    def forward(self, h_flatten, mask_flatten, spatial_shape, reference_points):
        pass

class MultiScaleDeformableDecoder(nn.Module):
    def __init__(
        self,
        d_model=512,
        num_head=16,
        num_query=100,
        num_points=4,
        num_level=4,
        dropout=0.1,
        num_layer=6
    ) -> None:
        super().__init__()
        layer = MultiScaleDeformableDecoderLayer(d_model, num_head, num_level, num_points, dropout)
        self.layers = clones(layer, num_layer)
    
    def get_initial_reference_points(self, ):
        pass

    def forward(self, hs, masks):
        # B, L, _ = hs[i].shape
        device = hs[0].device

        level_start_index = [0]
        h_flatten = []
        mask_flatten = []
        spatial_shape = []
        for idx, (h, mask) in enumerate(zip(hs, masks)):
            B, L, _ = h
            spatial_shape += [torch.array((1, L), device=device)]
            h_flatten += [h]
            mask_flatten += [mask]
        spatial_shape = torch.stack(spatial_shape, dim=0)
        h_flatten = torch.cat(h_flatten, dim=1)
        mask_flatten = torch.cat(mask_flatten, dim=1)
        reference_points = self.get_initial_reference_points()

        for layer in self.layers:
            h_flatten, reference_points = self.layers(h_flatten, reference_points)
