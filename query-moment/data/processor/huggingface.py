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


def cache_filter():
    def out_wrapper(fn):
        """
        from_key: [required]
        cache_args: [optional]
            cache_dir: [required]
            hash_key: [optional]
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
            cache_file = osp.join(cache_dir, f"{hash_id}.hdf5")
            verbose = self.cache_args.get("verbose", False)
            load_to_memory = self.cache_args.get("load_to_memory", False)
            overwrite = self.cache_args.get("overwrite", False)
            # no matter whether loaded to memory, inference must run with cache configured

            if not overwrite and osp.exists(cache_file):  # succesful loaded
                try:
                    with h5py.File(cache_file, "r") as hdf5_handler:
                        load_item = dict()
                        for k in hdf5_handler.keys():
                            load_item.update({k: np.array(hdf5_handler[k])})
                    if verbose:
                        log.info(f"cache loaded from {cache_file}")
                    if load_to_memory:
                        result.update(load_item)
                        if verbose:
                            log.info(f"cache merged to result")
                    return result
                except:
                    # load failed, run inference
                    if verbose:
                        log.info(
                            f"{cache_file} is invalid and removed, inference will run"
                        )
                    subprocess.run(
                        f"rm -rf {cache_file}", shell=True
                    )  # delete invalid file

            # run and save
            result, load_item = fn(self, result, True)
            with h5py.File(cache_file, "w") as hdf5_handler:
                for k in load_item.keys():
                    hdf5_handler.create_dataset(k, data=load_item[k])
            if verbose:
                log.info(f"cache stored to {cache_file}")
            if load_to_memory:
                result.update(load_item)
                if verbose:
                    log.info(f"cache merged to result")
            return result

        return wrapper

    return out_wrapper


class HuggingfaceExtractor:
    def __init__(self, extractor=None, from_key=None) -> None:
        assert extractor is not None
        self.extractor = extractor
        self.from_key = from_key

    def __call__(self, result):
        data = result[self.from_key]
        result[self.from_key + "_ext"] = self.extractor(data)
        return result


class HuggingfaceEmbedding:
    def __init__(
        self,
        model=None,
        extractor=None,
        from_key=None,
        cache_args=None,
    ):
        """
        cache_args:
            cache_dir: [required]
            hash_key: [optional]
            verbose: [optional]
        """
        assert from_key is not None and model is not None and extractor is not None
        # to_embeddings=False means for latter use but not load it into memory now
        log.warn("[processor] hf_embedding may use cuda")
        self.model = model
        self.extractor = extractor

        self.from_key = from_key
        self.cache_args = cache_args

    @torch.no_grad()
    @cache_filter()
    def __call__(self, result, split_load_item=False):
        # split out load_item for convenience for cache_filter
        # cache_filter is able to decide whether merge load_item into result or latter
        data = result[self.from_key]
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
