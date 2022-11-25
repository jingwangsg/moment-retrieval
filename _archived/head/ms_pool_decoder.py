import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import ROIAlign


class MultiScalePoolingDecoder(nn.Module):
    """
    multi-scale temporal pooling with given query
    """

    def __init__(self, d_model, num_head, num_query, dropout, pooling="roialign_v2") -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(num_query, d_model))
        self.query_attn = nn.MultiheadAttention(d_model, num_head, dropout)

    def forward(self, vid_feat_lvls, vid_mask_lvls):
        # vid_feat_lvls[i].shape = B, Lv, _
        # vid_mask_lvls[i].shape = B, Lv

        for vid_feat, vid_mask in zip(vid_feat, vid_mask):
            pass
        
        ret_dict=dict(intermediate_proposal=
                      )