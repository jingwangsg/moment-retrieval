import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalDAB(nn.Module):
    def __init__(self, backbone) -> None:
        super().__init__()
        self.backbone = backbone
    
    def forward(self, vid_feat, txt_feat, txt_mask, gt, mode="train"):
        vid_feat, txt_feat = self.backbone(vid_feat, txt_feat, txt_mask)