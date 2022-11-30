import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, repeat, rearrange, reduce
from kn_util.general import registry
from .segformerx import build_segformerx
from .ms_pooler import MultiScaleRoIAlign1D
from model.loss import focal_loss, l1_loss


class TemporalQueryDecoder(nn.Module):

    def __init__(self, pooler, d_model, num_query) -> None:
        super().__init__()
        self.query_embeddings = nn.Embedding(num_embeddings=num_query,
                                            embedding_dim=d_model)
        self.pooler = pooler
    
    def get_initital_reference(self, )

    def forward(self, feat_lvls, mask_lvls):
        self.pooler(feat_lvls)


class MultiScaleTemporalDetr(nn.Module):

    def __init__(self, backbone, head) -> None:
        super().__init__()
        d_model = cfg.model.d_model
        num_query = cfg.model.num_query
        self.backbone = build_segformerx(cfg)
        self.cfg = cfg

    def forward(self, vid_feat, vid_mask, txt_feat, txt_mask, mode="tensor"):
        vid_feat_lvls, vid_mask_lvls = self.backbone(vid_feat, vid_mask,
                                                     txt_feat, txt_mask)
        # B, Lv, _ = vid_feat_lvls[i].shape[0]
        ret_dict = self.head(vid_feat_lvls, vid_mask_lvls)

        return ret_dict
