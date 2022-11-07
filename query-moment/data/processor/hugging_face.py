from kn_util.general import global_registry, get_logger
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor
import os.path as osp
import os
import torch
import hydra
import h5py
import subprocess

log = get_logger(__name__)


def cache_filter(suffixes=[]):
    def out_wrapper(fn):
        """
        from_key: [required]
        hash_key: [optional]
        cache_args: [optional]
            cache_dir: [required]
            load_to_memory: [optional] False
            verbose: [optional] False
            overwrite: [optional] False

        """

        def wrapper(self, result):
            cache_args = self.cache_args
            if not cache_args:  # no cache
                return fn(self, result)

            # use cache
            if cache_args:
                os.makedirs(cache_args["cache_dir"], exist_ok=True)
            cache_dir = cache_args["cache_dir"]
            hash_id = result.get(cache_args.get("hash_key"), hash(result[self.from_key]))
            cache_files = [osp.join(cache_dir, self.from_key + suffix, f"{hash_id}.npy") for suffix in suffixes]
            verbose = self.cache_args.get("verbose", False)
            load_to_memory = self.cache_args.get("load_to_memory", False)
            overwrite = self.cache_args.get("overwrite", False)
            # no matter whether loaded to memory, inference must run with cache configured

            load_success = False
            load_item = dict()
            for idx, (suffix, cache_file) in enumerate(zip(suffixes, cache_files)):
                if not overwrite and osp.exists(cache_file):  # succesful loaded
                    try:
                        to_key = self.from_key + suffix
                        load_item.update({to_key: np.load(cache_file)})
                        if verbose:
                            log.info(f"cache loaded from {cache_file}")
                    except:
                        # load failed, run inference
                        if verbose:
                            log.info(
                                f"{cache_file} is invalid and removed, inference will run"
                            )
                        subprocess.run(
                            f"rm -rf {cache_file}", shell=True
                        )  # delete invalid file
                    if idx == len(cache_files) - 1:
                        load_success = True
            if load_success:
                if load_to_memory:
                    result.update(load_item)
                    if verbose:
                        log.info(f"cache merged to result")
                return result

            # run and save
            result, load_item = fn(self, result, True)
            for cache_file, suffix in zip(cache_files, suffixes):
                os.makedirs(osp.dirname(cache_file), exist_ok=True)
                to_key = self.from_key + suffix
                np.save(cache_file, load_item[to_key])
            if verbose:
                log.info(f"cache stored to {cache_file}")
            if load_to_memory:
                result.update(load_item)
                if verbose:
                    log.info(f"cache merged to result")
            return result

        return wrapper

    return out_wrapper


@global_registry.register_processor("hf_tokenizer")
class HuggingfaceTokenizer:
    def __init__(self, pretrained=None, from_key=None) -> None:
        assert pretrained is not None
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.from_key = from_key

    def __call__(self, result):
        text = result[self.from_key]
        result[self.from_key + "_inds"] = self.tokenizer(text).input_ids
        return result


@global_registry.register_processor("hf_embedding")
class HuggingfaceEmbedding:
    def __init__(
        self,
        model_cls="AutoModel",
        extractor_cls="AutoTokenizer",
        pretrained=None,
        is_text=True,
        from_key=None,
        cache_args=None,
    ):
        """
        cache_args:
            cache_dir: [required]
            hash_key: [optional]
            verbose: [optional]
        """
        assert from_key is not None and pretrained is not None
        # to_embeddings=False means for latter use but not load it into memory now
        log.warn("[processor] hf_embedding may use cuda")
        self.is_text = is_text
        self.model_cls = model_cls
        self.extractor_cls = extractor_cls

        self.from_key = from_key
        self.cache_args = cache_args
        self.pretrained = pretrained

    def _build_with_hydra(self, built_cls):
        cfg = {
            "_target_": "transformers.{}.from_pretrained".format(built_cls),
            "pretrained_model_name_or_path": self.pretrained,
        }
        instance = hydra.utils.instantiate(cfg)
        return instance

    def _build_model_and_extractor(self):
        model = self._build_with_hydra(self.model_cls)
        extractor = self._build_with_hydra(self.extractor_cls)
        return model, extractor

    @torch.no_grad()
    @cache_filter(["_embeddings"])
    def __call__(self, result, split_load_item=False):
        # split out load_item for convenience for cache_filter
        # cache_filter is able to decide whether merge load_item into result or latter
        data = result[self.from_key]
        if not hasattr(self, "model"):
            model, extractor = self._build_model_and_extractor()
            model = model.cuda()
            model.eval()
            self.model = model
            self.extractor = extractor

        model = self.model.cuda()
        inputs = self.extractor(data, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        embeddings = model(**inputs).last_hidden_state.squeeze(0).cpu().detach().numpy()

        load_item = dict()
        load_item[self.from_key + "_embeddings"] = embeddings

        if split_load_item:
            return result, load_item
        else:
            result.update(load_item)
            return result


# @global_registry.register_processor("hf_clip_embedding")
# class HFClipEmbedding(HuggingfaceEmbedding):
#     """for single branch in CLIP only"""

#     def _build_model_and_extractor(self):
#         if self.is_text:
#             model = CLIPTextModel.from_pretrained(self.pretrained)
#             extractor = CLIPTokenizer.from_pretrained(self.pretrained)
#         else:
#             model = CLIPVisionModel.from_pretrained(self.pretrained)
#             extractor = CLIPVisionModel.from_pretrained(self.pretrained)
#         if self.cache_args:  # to differentiate cliptext and clipvision
#             suffix = "_txt" if self.is_text else "_vis"
#             self.cache_args += suffix

#         return model, extractor
