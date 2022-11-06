from kn_util.general import import_modules
import os.path as osp

cur_dir = osp.dirname(__file__)
import_modules(cur_dir, "data.processor")
import copy
from kn_util.debug import dict_diff
from kn_util.general import global_registry, get_logger
import time
from pprint import pformat

log = get_logger(__name__)


def apply_processors(batch, processors):
    elem_copy = copy.copy(batch[0])
    for processor in processors:
        _st = time.time()
        if getattr(processor, "is_batch_processor", False):
            batch = processor(batch)
        else:
            for idx, e in enumerate(batch):
                batch[idx] = processor(e)

        verbose = global_registry.get_object("_TEST_PIPELINE_SIGNAL", False)
        if verbose:
            log.info(
                f"\napply [processor] {type(processor).__name__} (costs {time.time() - _st:3f} s)\n"
                + dict_diff(elem_copy, batch[0])
            )
            elem_copy = copy.copy(batch[0])

    return batch


def build_processors(processors_cfg):
    processors = []
    for processor_cfg in processors_cfg:
        processors += [global_registry.build_from_cfg(processor_cfg, "processor")]
    log.info("\n===============processors built==============\n" + pformat(processors))

    return processors
