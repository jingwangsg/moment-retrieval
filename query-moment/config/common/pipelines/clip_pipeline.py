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

vid_clip_file = osp.join(dataset_dir, pretrained + ".vid.hdf5")
txt_clip_file = osp.join(dataset_dir, pretrained + ".txt.hdf5")

load_video_clip = L(HDF5Loader)(hdf5_file=vid_clip_file, from_key="video_id", key_template="{}/pooler_output")
load_text_clip = L(HDF5Loader)(hdf5_file=txt_clip_file, from_key="text_id")
delete_unfound_video = L(HDF5Checker)(hdf5_file=vid_clip_file)

rename = L(Rename)(from_keys=["video_id.hdf5", "text_id.hdf5"], to_keys=["video_feat", "text_feat"])
collect = L(Collect)(from_keys=["text_feat", "text_mask", "vid_feat", "vid_mask", "gt"])

pipeline = dict(pretrained="clip-vit-large-patch14-336",
                video_max_len=512,
                collater="simple",
                pre_processors=[delete_unfound_video],
                processors=[load_video_clip, load_text_clip, rename, *default_processors, collect])
