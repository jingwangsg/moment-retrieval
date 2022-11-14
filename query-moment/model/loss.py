import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SetCriterion(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        pass


def l1loss(preds: List[torch.Tensor], gt, loss_weight=1.0):
    loss = 0.0
    for idx, pred in enumerate(preds):
        # NC, _2 = pred.shape
        loss = loss + torch.mean(
            torch.abs(pred[:, 0] - gt[idx, 0]) + torch.abs(pred[:, 1] - gt[idx, 1])
        )
    loss /= len(preds)
    return loss * loss_weight


def focal_loss(preds: List[torch.Tensor], gt, loss_weight=1.0):
    pass
