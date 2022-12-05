import torch

from torchmetrics import Metric

from misc import calc_iou_score_gt
from einops import repeat
from torchmetrics import Metric


class AverageMeter(Metric):
    higher_is_better = False
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, val):
        self.sum += val
        self.n += 1

    def compute(self):
        return self.sum / self.n


class RankMIoUAboveN(Metric):
    higher_is_better = True
    full_state_update = True

    def __init__(self, m, n) -> None:
        super().__init__()
        self.m = m
        self.n = n
        self.add_state("hit", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_sample", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, pred_bds, scores, gt):
        B = len(pred_bds)
        pred_bds_batch = pred_bds
        gt_batch = gt
        scores_batch = scores

        for i in range(B):
            pred_bds = pred_bds_batch[i]
            gt = gt_batch[i]
            scores = scores_batch[i]

            _, sorted_index = torch.sort(scores)
            pred_bds = pred_bds[sorted_index][:self.m]

            Nc, _2 = pred_bds.shape
            expand_gt = repeat(gt, "i -> nc i", nc=Nc)
            ious = calc_iou_score_gt(pred_bds, expand_gt)
            is_hit = torch.sum(ious >= self.n)
            self.hit += (is_hit > 0).float()
            self.num_sample += 1

    def compute(self):
        return self.hit / self.num_sample


class TrainEvaluater:

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def update_all(self, losses):
        if not hasattr(self, "metrics"):
            self.metrics = dict()
            for loss_nm, loss_val in losses.items():
                metric = AverageMeter().to(loss_val.device)
                self.metrics[loss_nm] = metric
        for loss_nm, metric in self.metrics.items():
            metric.update(losses[loss_nm])

    def compute_all(self):
        ret_dict = dict()
        for loss_nm, metric in self.metrics.items():
            ret_dict["train/" + loss_nm] = metric.compute().item()
            metric.reset()

        return ret_dict


class ValTestEvaluater:

    def __init__(self, cfg, domain="val"):
        assert domain in ("val", "test")
        self.metrics = dict()
        self.cfg = cfg
        self.domain = domain
        ms = cfg.eval.ms
        ns = cfg.eval.ns
        for m in ms:
            for n in ns:
                self.metrics[f"Rank{m}@IoU={n:.1f}"] = RankMIoUAboveN(m=m, n=n)

    def update_all(self, outputs):
        for metric_nm, metric in self.metrics.items():
            if isinstance(metric, RankMIoUAboveN):
                boxxes = outputs["boxxes"]
                scores = outputs["scores"]
                gt = outputs["gt"]
                metric.update(boxxes, scores, gt)

    def compute_all(self, losses):
        for metric_nm, metric in self.metrics.items():
            ret_dict = dict()
            ret_dict[self.domain + "/" + metric_nm] = metric.compute().item()
        return ret_dict
