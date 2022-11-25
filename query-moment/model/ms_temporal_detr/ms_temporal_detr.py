import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, repeat, rearrange, reduce
from kn_util.general import registry
from .segformerx import build_segformerx
# from model.head.ms_pool_decoder import MultiScalePoolingDecoder
from model.loss import focal_loss, l1_loss


class MultiScaleTemporalDetr(nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()
        d_model = cfg.model.d_model
        num_query = cfg.model.num_query
        self.backbone = build_segformerx(cfg)
        self.cfg = cfg

    def forward(self, vid_feat, vid_mask, txt_feat, txt_mask):
        vid_feat_lvls, vid_mask_lvls = self.backbone(vid_feat, vid_mask, txt_feat, txt_mask)
        # B, Lv, _ = vid_feat_lvls[i].shape[0]
        ret_dict = self.head(vid_feat_lvls, vid_mask_lvls)

        return ret_dict

    def compute_loss(self, batch_dict):
        loss = 0.0
        gt = batch_dict["gt"]
        if "intermediate_proposal" in batch_dict and "intermediate_score" in batch_dict:
            for score, proposal in zip(batch_dict["intermediate_score"], batch_dict["intermediate_proposal"]):
                score = torch.topk(score, k=self.cfg.train.topk, dim=1)
                loss += focal_loss(score, proposal, loss_weight=self)
                loss += l1_loss(pred_bds=)

    def inference(self,):
        pass