from kn_util.file import load_pickle
import numpy as np
import sys; import os.path as osp; sys.path.insert(0, osp.join(osp.dirname(__file__), ".."))
dets = load_pickle("/export/home2/kningtg/WORKSPACE/MSAT/dets.pkl")

def nms(dets, thresh=0.4, top_k=-1):
    """Pure Python NMS baseline."""
    if len(dets) == 0: return []
    order = np.arange(0,len(dets),1)
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

x1 = nms(dets, 0.37)

from misc import nms as nms_mmcv

import torch
dets = torch.from_numpy(dets).float()
scores = torch.arange(dets.shape[0]-1, -1, -1).float()
batch_idxs = torch.zeros((dets.shape[0],), dtype=torch.int32)
x2 = nms_mmcv(dets, scores, batch_idxs, iou_threshold=0.37)

import ipdb; ipdb.set_trace() #FIXME
