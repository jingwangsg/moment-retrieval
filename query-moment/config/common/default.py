from omegaconf import OmegaConf
from .pipelines.clip import pipeline_cfg
from .runtime import G
import os.path as osp

data = dict(dataset="tacos", dataset_dir=osp.join("${G.data_dir}", "${data.dataset}"))
data.update(pipeline_cfg)
del pipeline_cfg

train = dict(
    prefetch_factor=6, num_workers=8, num_epoch=12, eval_epoch_interval=1, batch_size=16
)
