import torch

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