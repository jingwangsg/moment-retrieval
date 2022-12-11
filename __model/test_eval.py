import numpy as np


def iou(pred, gt):  # require pred and gt is numpy
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    if not pred_is_list:
        pred = [pred]
    if not gt_is_list:
        gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = 1.0 * inter / union
    if not gt_is_list:
        overlap = overlap[:, 0]
    if not pred_is_list:
        overlap = overlap[0]
    return overlap


def nms(dets, thresh=0.4, top_k=-1):
    """Pure Python NMS baseline."""
    if len(dets) == 0:
        return []
    order = np.arange(0, len(dets), 1)
    dets = np.array(dets)
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    lengths = x2 - x1
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) == top_k:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep]


# ipdb> pred_bds
# tensor([[-3.1651e-01,  4.0106e+02]])
# ipdb> expand_gt
# tensor([[ 6.8367, 16.5646]]
# iou = [0.02423654]


def eval(segments, data):
    tious = [0.3, 0.5, 0.7]
    recalls = [1, 5]

    eval_result = [[[] for _ in recalls] for _ in tious]
    max_recall = max(recalls)
    average_iou = []
    for seg, dat in zip(segments, data):
        seg = nms(seg, thresh=0.37, top_k=max_recall).tolist()
        overlap = iou(seg, [dat['times']])
        average_iou.append(np.mean(np.sort(overlap[0])[-3:]))

        for i, t in enumerate(tious):
            for j, r in enumerate(recalls):
                eval_result[i][j].append((overlap > t)[:r].any())
    eval_result = np.array(eval_result).mean(axis=-1)
    miou = np.mean(average_iou)

    return eval_result, miou


from kn_util.file import load_pickle

data = load_pickle("/export/home2/kningtg/WORKSPACE/MSAT/data.pkl")
segments = load_pickle("/export/home2/kningtg/WORKSPACE/MSAT/segments.pkl")
x1 = eval(segments, data)

import sys
import os.path as osp

sys.path.insert(0, osp.join(osp.dirname(__file__), ".."))
from evaluater import ValTestEvaluater
from omegaconf import OmegaConf
import torch
from misc import nms as nms_mmcv

segments = [torch.tensor(seg) for seg in segments]
scores = [torch.arange(len(segments[_]) - 1, -1, -1) for _ in range(len(segments))]
batch_idxs = [torch.tensor([i] * int(scores[i].shape[0])) for i in range(len(scores))]

segments = torch.cat(segments, dim=0).float()
scores = torch.cat(scores, dim=0).float()
batch_idxs = torch.cat(batch_idxs, dim=0).long()

nms_boxxes, nms_scores = nms_mmcv(segments, scores, batch_idxs, 0.37)
'''
(array([[0.05613165, 0.33814247],
       [0.01487827, 0.1886835 ],
       [0.00608656, 0.06875564]]), 0.08562581858831726)
'''

cfg = OmegaConf.create(dict(eval=dict(ms=[1, 5], ns=[0.3, 0.5, 0.7], best_monitor="val/R1@IoU=0.7", is_best="max")))
evaluater = ValTestEvaluater(cfg, domain="val")

outputs = dict(boxxes=nms_boxxes, scores=nms_scores, gt=torch.tensor([dat["times"] for dat in data]))
evaluater.update_all(outputs)
x2 = evaluater.compute_all()

import ipdb

ipdb.set_trace()  #FIXME
