import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, repeat, rearrange, reduce
from kn_util.general import registry

# class MultiScaleTemporalDetr(nn.Module):
#     def __init__(self, backbone, head) -> None:
#         super().__init__()
#         self.pre_clf = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU())
#         self.clf_span = nn.Linear(d_ff, )
    
#     def forward(self, vid_feat, vid_mask, txt_feat, txt_mask):
#         vid_feat_lvls, vid_mask_lvls = self.backbone(vid_feat, vid_mask, txt_feat, txt_mask)
#         # B, Lv, _ = vid_feat_lvls[i].shape[0]
#         self.head(vid_feat_lvls, vid_mask_l)
    
#     def compute_loss(self, **kwargs):
#         pass
    
#     def inference(self, ):
#         pass