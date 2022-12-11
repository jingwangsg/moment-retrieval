from models.msat_off.modeling import TAN, TLocVLBERT, FrameAvgPool
from ..common.runtime import flags, paths
from detectron2.config import LazyCall as L
import os.path as osp
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

data = dict(
    dataset="tacos",
    dataset_dir=osp.join("${paths.data_dir}", "${data.dataset}"),
    datapipe="${model_cfg.arch}",
    max_len_video=256,
    target_stride=2,
    vid_hdf5="i3d.hdf5",
    vid_hdf5_key_template="{video_id}",
    txt_hdf5="roberta-large.txt.hdf5",
    txt_hdf5_key_template="{text_id}/last_hidden_state",
)

eval = dict(ms=[1, 5], ns=[0.3, 0.5, 0.7], best_monitor="val/R1@IoU=0.7", is_best="max")

train = dict(num_workers=8,
             num_epochs=50,
             eval_epoch_interval=1,
             batch_size=16,
             optimizer=L(AdamW)(params=None, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.000),
             clip_grad=10.0,
            #  lr_scheduler=L(StepLR)(optimizer=None, step_size=5),
             val_interval=1.0,
             print_interval=0.2)

model_cfg = dict(arch="msat",
                 w_stage_loss=0.3,
                 w_reg_loss=1.0,
                 w_iou_loss=200.0,
                 num_clips="${data.max_len_video}//${data.target_stride}",
                 nms_threshold=0.37)

model = L(TAN)(frame_layer=L(FrameAvgPool)(kernel_size=2, stride=2),
               bert_layer=L(TLocVLBERT)(dataset="${data.dataset}",
                                        visual_size=1024,
                                        hidden_size=1024,
                                        input_size_txt=1024,
                                        num_hidden_layers=6,
                                        num_attention_heads=32,
                                        intermediate_size=512,
                                        hidden_act="gelu",
                                        hidden_dropout_prob=0.1,
                                        attention_probs_dropout_prob=0.1,
                                        max_position_embeddings=512,
                                        type_vocab_size=2,
                                        initializer_range=0.02,
                                        visual_scale_text_init=1.0,
                                        visual_scale_object_init=1.0,
                                        visual_ln=False,
                                        word_embedding_frozen=False,
                                        with_pooler=True,
                                        classifier_type="2fc",
                                        classifier_dropout=0.1,
                                        classifier_hidden_size=512),
               cfg="${model_cfg}")
