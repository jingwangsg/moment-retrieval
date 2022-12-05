import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import nms
from .transformer import VisualLinguisticTransformer
from .head import MultiStageHead


class MultiStageAggregateTransformer(nn.Module):

    def __init__(self, transformer: VisualLinguisticTransformer, head: MultiStageHead, cfg) -> None:
        super().__init__()
        self.transformer = transformer
        self.head = head
        self.cfg = cfg

    def generate_gt_dist(self, gt):
        cfg = self.cfg
        mid_gt = (gt[:, 0] + gt[:, 1]) / 2
        expanded_gt = torch.concat([gt, mid_gt[:, None]], dim=1)
        expanded_gt_by_nclip = cfg.num_clip * expanded_gt
        alpha = torch.tensor([cfg.loss_cfg.alpha_s, cfg.loss_cfg.alpha_s, cfg.loss_cfg.alpha_m], device=gt.device)
        sigma = alpha[None, :] * (expanded_gt_by_nclip[..., 1] - expanded_gt_by_nclip[..., 0])[:, None]
        gt_dist = -(torch.arange(cfg.num_clip, device=gt.device)[None, :, None] -
                    expanded_gt_by_nclip[:, None, :])**2 / (2 * (sigma[:, None, :]**2))
        gt_dist = torch.exp(gt_dist)
        return gt_dist

    def forward(self,
                vid_feat,
                vid_mask,
                txt_feat,
                txt_mask,
                word_label=None,
                word_mask=None,
                gt=None,
                mode="inference"):
        gt_dist = self.generate_gt_dist(gt)
        vid_feat, txt_feat = self.transformer(vid_feat, vid_mask, txt_feat, txt_mask)
        if mode == "train":
            losses = self.head(vid_feat=vid_feat, txt_feat=txt_feat, gt=gt, gt_dist=gt_dist, mode="train")
            return losses
        else:
            ret_dict = self.head(vid_feat=vid_feat, txt_feat=txt_feat, mode="inference")
            nms_pred_bds, nms_score = nms(ret_dict["pred_bds_raw"],
                                          ret_dict["score_raw"],
                                          ret_dict["batch_idxs"],
                                          iou_threshold=self.cfg.iou_threshold)
            ret_dict.update(dict(pred_bds=nms_pred_bds, score=nms_score))

            return ret_dict