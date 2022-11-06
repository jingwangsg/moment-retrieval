from kn_util.general import global_registry
from kn_util.data import generate_sample_indices
from kn_util.file import load_pickle
import os.path as osp
import h5py
import numpy as np
import decord as de
import os


@global_registry.register_processor("batch.load_or_cache")
class LoadOrCache:
    is_batch_processor = True

    def __init__(self, cache_path) -> None:
        self.cache_path = cache_path

    def __call__(self, batch):
        if osp.exists(self.cache_path):
            try:
                self.datasets = load_pickle(self.cache_path)
                load_fail = True
            except:
                load_fail = False

            if not load_fail:  # safe load for big file
                return
        global_registry.register_object(
            "_CACHE_SIGNAL", True
        )  # cache signal for later caching
        return batch


@global_registry.register_processor("load_hdf5")
class HDF5Loader:
    def __init__(self, hdf5_file, path_template="{}", from_key=None) -> None:
        assert from_key
        self.handle = h5py.File(hdf5_file, "r")
        self.from_key = from_key
        self.path_template = path_template

    def __call__(self, result):
        ind = result[self.from_key]
        result[self.from_key + "_hdf5"] = self.handle[self.path_template.format(ind)]

        return result


@global_registry.register_processor("load_npy")
class NumpyLoader:
    def __init__(self, npy_dir=None, path_template="{}.npy", hash_key=None):
        assert npy_dir and hash_key
        self.npy_dir = npy_dir
        self.hash_key = hash_key
        self.path_template = path_template

    def __call__(self, result):
        hash_entry = result[self.hash_key]
        result[self.hash_key + "_npy"] = np.load(
            osp.join(self.npy_dir, self.path_template.format(hash_entry))
        )
        return result


@global_registry.register_processor("load_video_decord")
class DecodeVideoLoader:
    def __init__(
        self,
        max_len=None,
        stride=None,
        path_format="{}.avi",
        from_key="video_id",
        cache_dir=None,
    ) -> None:
        assert (max_len is not None) ^ (stride is not None)
        self.stride = stride
        self.max_len = max_len

        self.path_format = path_format
        self.from_key = from_key
        self.cache_dir = cache_dir
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

    def __call__(self, result):
        video_id = result[self.from_key]
        if self.cache_dir:
            cache_file = osp.join(self.cache_dir, f"{video_id}.npy")
            if osp.exists(cache_file):
                arr = np.load(cache_file)
                result[self.from_key + "_imgs"] = arr
                return result
        video_path = self.video_format.format(video_id)
        vr = de.VideoReader(video_path)
        tot_len = len(vr)

        sampled_indices = generate_sample_indices(
            tot_len, self.max_len, self.stride, output_stride=True
        )
        arr = vr.get_batch(sampled_indices).asnumpy()

        if self.cache_dir:
            np.save(cache_file, arr)

        result[self.from_key + "_imgs"] = arr
        return result
