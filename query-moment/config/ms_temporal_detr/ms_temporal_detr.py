from omegaconf import OmegaConf
from detectron2.config import LazyCall as L
from ..common.runtime import paths, flags
import os.path as osp
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from ..common.pipelines.default_pipeline import pipeline

data = dict(
    dataset="tacos",
    dataset_dir=osp.join("${paths.data_dir}", "${data.dataset}"),
    **pipeline)

train = dict(
    prefetch_factor=6,
    num_workers=8,
    # prefetch_factor=2,
    # num_workers=0,
    max_epochs=12,
    eval_epoch_interval=1,
    batch_size=16,
    optimizer=L(AdamW)(params=None, lr=3e-4),
    # lr_scheduler=(StepLR)()
)

from model.arch import ms_temporal_detr
from model.backbone import segformerx