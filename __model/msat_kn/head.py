import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, repeat, rearrange
from kn_util.nn_utils.layers import MLP
from kn_util.nn_utils import clones
from __model.loss import l1_loss
from misc import calc_iou_score_gt


class MultiStageHead(nn.Module):

    def __init__(self, d_model, dist_ff_dim, text_ff_dim, vocab_size, dropout=0.1, loss_cfg=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.loss_cfg = loss_cfg

        self.text_clf = MLP(d_model, text_ff_dim, vocab_size, has_ln=False, dropout=dropout)
        self.dist_mlps = clones(MLP(d_model, dist_ff_dim, 1, has_ln=False, dropout=dropout), 3)
        self.moment_ffn = torch.nn.Sequential(
            torch.nn.Dropout(dropout, inplace=False),
            torch.nn.Linear(d_model, d_model * 3),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout, inplace=False),
        )
        self.stage_clf = nn.Linear(d_model, 3)

    def forward(self, vid_feat, txt_feat, word_label=None, word_mask=None, gt=None, gt_dist=None, mode="train"):
        B, Lv, D = vid_feat.shape

        word_logits = self.text_clf(txt_feat)

        hidden_stage = self.moment_ffn(vid_feat)
        hidden_st, hidden_ed, hidden_mid = torch.split(hidden_stage, self.d_model, dim=2)
        idxs = torch.arange(0, Lv)
        moment_embeddings_st = repeat(hidden_st, "b lv d -> b lv i d", i=Lv)
        mid_idxs = ((idxs[:, None] + idxs[None, :]) / 2).long().flatten()
        moment_embeddings_mid = repeat(hidden_mid[:, mid_idxs,], "i (lv lv1) d -> i lv lv1 d", lv=Lv)
        moment_embeddings_ed = repeat(hidden_ed, "b lv d -> b i lv d", i=Lv)
        moment_embeddings = torch.concat([moment_embeddings_st, moment_embeddings_ed, moment_embeddings_mid], dim=-1)
        moment_logits = self.stage_clf(moment_embeddings)

        moment_indices = torch.arange(Lv)
        moment_indices = repeat(moment_indices, "b lv i -> b lv lv1 i", lv1=Lv)
        moment_indices = torch.stack([moment_indices, moment_indices.transpose(2, 3)], dim=3)
        pred_bds_by_nclip = moment_indices + stage_logits[:, :, :2]

        if mode == "train":
            stage_logit_st = self.dist_mlps[0](vid_feat)
            stage_logit_ed = self.dist_mlps[1](vid_feat)
            stage_logit_mid = self.dist_mlps[2](vid_feat)
            stage_logits = torch.concat([stage_logit_st, stage_logit_ed, stage_logit_mid], dim=1)
            stage_logits = stage_logits.squeeze(2).sigmoid()

            losses = dict()
            loss = 0
            pred_bds_by_nclip_flatten = rearrange(pred_bds_by_nclip, "b lv lv1 i -> (b lv lv1) i")
            gt_flatten = rearrange(gt, "b i -> b k i", k=Lv * Lv)
            gt_flatten_by_nclip = gt_flatten * Lv
            stage_score = stage_logits.sigmoid()
            moment_score = moment_logits.sigmoid()
            moment_score_flatten = rearrange(moment_score, "b lv lv1 -> b (lv lv1)")
            valid_indices_filter = pred_bds_by_nclip_flatten[:, :, 0] <= pred_bds_by_nclip_flatten[:, :, 1]

            if self.loss_cfg.get("word_mask_loss", 0):
                word_logits_flatten = rearrange(word_logits, "b lt nv -> (b l) nv")
                word_mask_flatten = word_mask.flatten()
                word_label_flatten = word_label.flatten()
                word_logits_selected = word_logits_flatten[word_mask_flatten]
                word_mask_loss = F.cross_entropy(word_logits_selected, word_label_flatten, reduction="mean")
                losses["word_mask_loss"] = word_mask_loss
                loss += self.loss_cfg["word_mask_loss"] * word_mask_loss

            if self.loss_cfg.get("iou_loss", 0):
                iou_flatten = calc_iou_score_gt(pred_bds_by_nclip_flatten, gt_flatten_by_nclip)
                iou_loss = F.cross_entropy(moment_score_flatten, iou_flatten,
                                           reduction="none") * (iou_flatten - moment_score_flatten)**2
                iou_loss = iou_loss[valid_indices_filter].mean()
                losses["iou_loss"] = iou_loss
                loss += self.loss_cfg["iou_loss"] + iou_loss

            if self.loss_cfg.get("stage_loss", 0):
                stage_loss = F.kl_div(torch.log(stage_score), gt_dist, reduction="mean")
                losses["stage_loss"] = stage_loss
                loss += self.loss_cfg["stage_loss"] * stage_loss

            if self.loss_cfg.get("reg_loss", 0):
                gt_offset = (gt * Lv)[:, None, None, :] - moment_indices
                gt_offset_flatten = rearrange(gt_offset, "b lv lv1 i -> b (lv lv1) i")
                reg_loss = torch.abs(gt_offset_flatten - moment_score_flatten)
                reg_loss = reg_loss[valid_indices_filter].mean()
                losses["reg_loss"] = reg_loss
                loss += self.loss_cfg["reg_loss"] * reg_loss

            return losses

        else:
            pred_bds_flatten = pred_bds_by_nclip_flatten / Lv
            pred_bds_valid_flatten = pred_bds_flatten[valid_indices_filter]
            moment_score_valid_flatten = moment_score[valid_indices_filter]
            batch_idxs = rearrange(torch.arange(B), "b -> b k", k=Lv * Lv)
            batch_idxs_valid = batch_idxs[valid_indices_filter]

            ret_dict = dict(pred_bds_raw=pred_bds_valid_flatten,
                            score_raw=moment_score_valid_flatten,
                            batch_idxs=batch_idxs_valid)

            return ret_dict
