from omegaconf import OmegaConf
from detectron2.config import LazyCall as L
from ..common.runtime import paths, flags
import os.path as osp
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from ..common.pipelines.msat_pipeline import pipeline as msat_pipeline

data = dict(dataset="tacos",
            dataset_dir=osp.join("${paths.data_dir}", "${data.dataset}"),
            pipeline_verbose=False,
            msat_pipeline=)

data["to_multiple_pad_video"] = 16
data["video_max_len"] = 128

train = dict(
    prefetch_factor=6,
    num_workers=8,
    max_epochs=50,
    eval_epoch_interval=1,
    batch_size=16,
    optimizer=L(AdamW)(params=None, lr=1e-4),
    val_monitor="val/Rank1@IoU=07",
    clip_grad=2.0
    # lr_scheduler=(StepLR)()
)

from model.msat import VisualLinguisticTransformer, MultiStageAggregateTransformer, MultiStageHead

model_cfg = dict(d_model=1024,
                 num_clip="${data.video_max_len}",
                 dropout=0.1,
                 loss_cfg=dict(stage_loss=0.3,
                               reg_loss=1.0,
                               iou_loss=200.0,
                               word_mask_loss=0.2,
                               alpha_s=0.25,
                               alpha_m=0.21))

model = L(MultiStageAggregateTransformer)(transformer=None, head=None, cfg=model_cfg)
model.transformer = L(VisualLinguisticTransformer)(d_model="${..d_model}",
                                                   num_heads=16,
                                                   ff_dim=2048,
                                                   num_layers=6,
                                                   max_len_vid=128,
                                                   max_len_txt=30,
                                                   input_size_vid=1024,
                                                   input_size_txt=300,
                                                   dropout="${..dropout")
model.head = L(MultiStageHead)(d_model=1024,
                               dist_ff_dim=1024,
                               text_ff_dim=1024,
                               vocab_size=1054,
                               dropout="${..dropout}",
                               loss_cfg="${..loss_cfg}")
