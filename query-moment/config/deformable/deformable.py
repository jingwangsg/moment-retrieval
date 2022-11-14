from omegaconf import OmegaConf
from detectron2.config import LazyCall as L
from ..common.pipelines.clip import pipeline_cfg
from ..common.runtime import G
import os.path as osp
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from detectron2.solver.build import get_default_optimizer_params

data = dict(dataset="tacos", dataset_dir=osp.join("${G.data_dir}", "${data.dataset}"))
data.update(pipeline_cfg)
del pipeline_cfg

train = dict(
    prefetch_factor=6,
    num_workers=8,
    num_epoch=12,
    eval_epoch_interval=1,
    batch_size=16,
    optimizer=L(AdamW)(params=None, lr=3e-4),
    # lr_scheduler=(StepLR)()
)