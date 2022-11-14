import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from einops import rearrange, repeat
from detectron2.config import instantiate

class SetCriterion(nn.Module):
    def __init__(self, loss_cfgs) -> None:
        super().__init__()
        self.loss_cfgs = loss_cfgs

    def forward(self, **kwargs):
        tot_loss = 0.0
        for loss_cfg in self.loss_cfgs:
            # dispatch output to multiple losses
            for k, v in kwargs.items():
                if hasattr(loss_cfg, k):
                    loss_cfg[k] = kwargs[k]
            tot_loss += instantiate(loss_cfg)
        return tot_loss


def l1_loss(pred_bds, gt, loss_weight=1.0):
    if isinstance(pred_bds, torch.Tensor):
        # B, Nc, _2 = preds.shape
        # B, _2 = gt.shape
        return torch.mean(pred_bds - gt[:, None, :])
    else:
        loss = 0.0
        for idx, pred_bd in enumerate(pred_bds):
            # NC, _2 = pred.shape
            loss = loss + 0.5 * (
                torch.abs(pred_bd[:, 0] - gt[idx, 0])
                + torch.abs(pred_bd[:, 1] - gt[idx, 1])
            )
        loss /= len(pred_bds)
    return loss * loss_weight


def dist_loss(pred_logits, gt_span, loss_weight=1.0, loss_fn=F.kl_div):
    if isinstance(pred_logits, torch.Tensor):
        B, Nc, _2, Llgt = pred_logits.shape
        expand_gt = repeat(gt_span, "b i llgt -> b nc i llgt", nc=Nc)
        pred_score = torch.softmax(pred_logits, dim=-1)
        return loss_fn(pred_score, expand_gt, reduction="mean") * loss_weight
    else:
        loss = 0.0
        for logits in pred_logits:
            Nc, _2, Llgt = logits.shape
            score = torch.softmax(logits, dim=-1)
            loss += loss_fn(pred_score, expand_gt, reduction="mean")
        return loss * loss_weight

def calc_iou_score_gt(pred_bds, gt, type="iou"):
    """make sure the range between [0, 1) to make loss function happy"""
    min_ed = torch.minimum(pred_bds[:, 1], gt[:, 1])
    max_ed = torch.maximum(pred_bds[:, 1], gt[:, 1])
    min_st = torch.minimum(pred_bds[:, 0], gt[:, 0])
    max_st = torch.maximum(pred_bds[:, 0], gt[:, 0])
    
    I = torch.maximum(min_ed - max_st, torch.zeros_like(min_ed, dtype=torch.float, device=pred_bds.device))
    area_pred = pred_bds[1] - pred_bds[1]
    area_gt = gt[1] - gt[0]
    U = area_pred + area_gt - I
    Ac = max_ed - min_st

    iou = I / U

    if type == "iou":
        return iou
    elif type == "giou":
        return 0.5 * (iou + U / Ac)
    else:
        raise NotImplementedError()


def focal_loss(iou_scores, pred_bds, gt, alpha=2, iou_type="iou", loss_weight=1.0):
    loss = 0.0
    if isinstance(pred_bds, torch.Tensor):
        B, Nc, _2 = pred_bds.shape
        pred_bds_flatten = rearrange("b nc i -> (b nc) i")
        gt_flatten = repeat(gt, "b i -> (b nc) i", nc=Nc)
        iou_gt_flatten = calc_iou_score_gt(pred_bds_flatten, gt_flatten, type=iou_type)
        iou_scores_flatten = rearrange("b nc -> (b nc)")
        loss = F.binary_cross_entropy(iou_scores_flatten, iou_gt_flatten, reduction="none")
        loss = loss * ((iou_scores_flatten - iou_gt_flatten) ** alpha)
        loss = torch.mean(loss)
        return loss * loss_weight
    else:
        loss = 0.0
        tot_cand = 0
        for iou_score, pred_bd, cur_gt in zip(iou_scores, pred_bds, gt):
            Nc, _2 = pred_bd.shape
            gt_flatten = repeat(cur_gt, "i -> nc i", nc=Nc)
            iou_gt = calc_iou_score_gt(pred_bd, gt_flatten)
            cur_loss = F.binary_cross_entropy(iou_score, iou_gt, reduction="none")
            cur_loss = cur_loss * ((iou_score - iou_gt) ** alpha)
            loss += cur_loss
            tot_cand += Nc
        return loss / tot_cand * loss_weight