from kn_util.data import general_pad, stack_list_to_tensor, fix_tensor_to_float32, collect_features_from_sample_list
from .processors import apply_processors
import copy
from typing import Dict, List, Any
import numpy as np
from kn_util.general import global_registry

@global_registry.register_collater("default")
class DefaultCollater:
    def __init__(self, cfg, processors, is_train) -> None:
        self.cfg = cfg
        self.processors = processors
        self.is_train = is_train
    
    def get_feature_dict(self, batch) -> Dict[List[np.ndarray]]:
        vid_feat, text_inds = collect_features_from_sample_list(batch, include_keys=["vid_feat", "text_inds"])
        vid_feat, vid_mask = general_pad(vid_feat, fill_value=0.0, axis=0)
        text_inds, text_mask = general_pad(text_inds, fill_value=0, axis=0)

        feature_dict = dict(vid_feat=vid_feat, vid_mask=vid_mask, text_inds=text_inds, text_mask=text_mask)
        return feature_dict

    
    def __call__(self, _batch):
        batch = copy.deepcopy(_batch)
        batch = apply_processors(batch, self.processors)
        feature_dict = self.get_feature_dict(batch)
        return fix_tensor_to_float32(stack_list_to_tensor(feature_dict))