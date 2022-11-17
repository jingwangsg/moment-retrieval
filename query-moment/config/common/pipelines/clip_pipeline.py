from detectron2.config import LazyCall as L
from data.processor import *
from transformers import AutoModel, AutoTokenizer
from transformers.models.clip import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPFeatureExtractor
from kn_util.data import HFImageModelWrapper
import os.path as osp
from .default_bundle import processors as default_processors
import torch.nn.functional as F
from einops import rearrange

pretrained = "${data.pretrained}"
dataset = "${data.dataset}"
dataset_dir = "${data.dataset_dir}"
cache_dir = "${paths.cache_dir}"
vid_feat_type = "${data.video_feat_type}"


delete_unfound_video = L(BatchLambda)(_lambda=L(FilterFunctionBuilder)(dataset="${data.dataset}"))

rename = L(Rename)(
    from_keys=[
        "video_frame_paths.last_hidden_state", "text.last_hidden_state"
    ],
    to_keys=["video_feat", "text_feat"])
"""pipeline collections"""

pipeline = dict(
    pretrained="openai/clip-vit-large-patch14-336",
    video_max_len=512,
    collater="simple",
    pre_processors=[delete_unfound_video],
    default_processors)
