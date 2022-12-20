import torch
import torch.nn as nn
import torch.nn.functional as F
from ...loss import l1_loss


class MomentDETR(nn.Module):

    def __init__(self, backbone, d_model, num_query) -> None:
        super().__init__()
        self.query_embs = nn.Embedding(num_query, d_model)

    def forward(self, vid_feat, txt_feat, txt_mask, gt=None, mode="train"):
        pass