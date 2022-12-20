from omegaconf import OmegaConf
from ..runtime import paths, flags
import os.path as osp
from kn_util.config.common import adamw, reduce_lr_on_plateau
from kn_util.config import LazyCall as L
from kn_util.config import eval_str
from models.backbone.segformerx import SegFormerX, SegFormerXFPN
from models.arch.temporal_dab import *

data = dict(datapipe="mst_detr",
            dataset="tacos",
            dataset_dir=osp.join("${paths.data_dir}", "${data.dataset}"),
            max_len_video=512,
            target_stride=4,
            word_mask_rate=0.0,
            vid_hdf5="i3d.hdf5",
            vid_hdf5_key_template="{video_id}")

eval = dict(ms=[1, 5], ns=[0.3, 0.5, 0.7], best_monitor="R1@IoU=0.7", is_best="max")

train = dict(num_workers=8,
             num_epochs=100,
             eval_epoch_interval=1,
             batch_size=32,
             optimizer=adamw(lr=1e-4, weight_decay=0.000),
             clip_grad=10.0,
             lr_scheduler=reduce_lr_on_plateau(factor=0.5, patience=5),
             val_interval=1.0,
             print_interval=0.2)
