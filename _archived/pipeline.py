from kn_util.data import (
    merge_list_to_tensor,
    fix_tensor_to_float32,
    collect_features_from_sample_list,
)
from .processor import *
from .processor import test_pipeline_signal as _signal
import copy
from typing import Dict, List, Any
import numpy as np
from kn_util.general import registry
from kn_util.debug import SignalContext
from collections import OrderedDict


class SimpleCollater:
    """collect feature from sample list into dict of tensor"""

    def __init__(self, cfg, processors, is_train) -> None:
        self.cfg = cfg
        self.processors = processors
        self.is_train = is_train

    def get_feature_dict(self, batch) -> Dict[str, List[np.ndarray]]:
        return collect_features_from_sample_list(batch)

    def __call__(self, _batch):
        batch = copy.deepcopy(_batch)
        batch = apply_processors(batch, self.processors)
        feature_dict = self.get_feature_dict(batch)
        return fix_tensor_to_float32(merge_list_to_tensor(feature_dict))

@registry.register_pipeline("default")
class DefaultPipeline:

    def __init__(self, cfg):
        self.cfg = cfg

    def build_preprocessor():
        pass


def build_default_bundle(cfg):
    sample_video = SequenceSampler(axis=0, max_len=cfg.data.video_max_len, from_key="video_feat"),
    pad_video = SequencePad(from_key="video_feat.sample", axis=0, fill_value=0.0)
    pad_text = SequencePad(from_key="text_feat", axis=0, fill_value=0)
    rename = Rename(from_keys=[
        "text_feat.pad",
        "text_feat.mask",
        "video_feat.sample.pad",
        "video_feat.sample.mask",
    ],
                    to_keys=["text_feat", "text_mask", "vid_feat", "vid_mask"])

    # text_feat, vid_feat -> text_feat, text_mask, vid_feat, vid_mask
    processors = [sample_video, pad_video, pad_text, rename]
    return processors


@registry.register_pipeline("clip")
class CLIPPipeline:

    def __init__(self, cfg, is_train) -> None:
        self.cfg = cfg
        self.is_train = is_train

    def build_preprocessor(self):
        return []

    def build_collater(self):
        Dcfg = self.cfg.data
        clip_txt_hdf5 = Dcfg.clip_pretrained + ".txt.hdf5"
        clip_vid_hdf5 = Dcfg.clip_pretrained + ".vid.hdf5"
        dataset_dir = Dcfg.paths.dataset_dir
        load_video_feat = HDF5Loader(hdf5_file=osp.join(dataset_dir, clip_txt_hdf5), from_key="video_id")
        load_text_feat = HDF5Loader(hdf5_file=osp.join(dataset_dir, clip_vid_hdf5), from_key="text_id")
        rename = Rename(["video_id.hdf5", "text_id.hdf5"], ["video_feat", "text_feat"])

        default_bundle = build_default_bundle(self.cfg)
        processors = [load_video_feat, load_text_feat, rename, *default_bundle]
        collater = SimpleCollater(self.cfg, processors=processors, is_train=self.is_train)
        return collater