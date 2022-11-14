from kn_util.data import (
    general_pad,
    stack_list_to_tensor,
    fix_tensor_to_float32,
    collect_features_from_sample_list,
)
from .processor import apply_processors
import copy
from typing import Dict, List, Any
import numpy as np
from kn_util.general import registry
from kn_util.debug import SignalContext
from collections import OrderedDict

_signal = "_TEST_PIPELINE_SIGNAL"

@registry.register_collater("default")
class DefaultCollater:
    def __init__(self, cfg, processors, is_train) -> None:
        self.cfg = cfg
        self.processors = processors
        self.is_train = is_train
        self.not_verbose_yet = True

    def get_feature_dict(self, batch) -> Dict[str, List[np.ndarray]]:
        keys = list(batch[0].keys())
        return dict(zip(keys, collect_features_from_sample_list(batch, keys=keys)))

    def __call__(self, _batch):
        batch = copy.deepcopy(_batch)
        with SignalContext(_signal, self.cfg.G.debug and self.not_verbose_yet):
            batch = apply_processors(batch, self.processors)
            self.not_verbose_yet = False  # only verbose on one batch
        feature_dict = self.get_feature_dict(batch)
        return fix_tensor_to_float32(stack_list_to_tensor(feature_dict))
