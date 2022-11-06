from kn_util.general import global_registry
import h5py
import numpy as np

@global_registry.register_processor("glove_tokenize")
class GloveTokenizer:
    def __init__(self) -> None:
        pass

@global_registry.register_processor("load_hdf5")
class LoadHDF5:
    def __init__(self, hdf5_file, from_key) -> None:
        self.handle = h5py.File(hdf5_file, "r")
        self.from_key = from_key

    def __call__(self, result):
        ind = result[self.from_key]
        result[self.from_key + "_hdf5"] = self.handle[ind]

        return result


@global_registry.register_processor("seq_sample")
class SequenceSampler:
    def __init__(self, axis=0, max_len=None, stride=None, from_key=None) -> None:
        assert (max_len is not None) ^ (stride is not None)
        assert from_key is not None
        self.max_len = max_len
        self.stride = stride
        self.from_key = from_key
        self.axis = axis

    def __call__(self, result):
        """seq should be a numpy ndarray"""
        seq = result[self.from_key]
        axis = self.axis
        tot_len = seq.shape[axis]
        if self.max_len is not None:
            stride = int(np.ceil((tot_len - 1) / (self.max_len - 1)))
        else:
            stride = self.stride

        dim_shape = len(seq.shape)
        slices = [
            slice(0, seq.shape[_]) if _ != axis else slice(0, tot_len - 1, stride)
            for _ in range(dim_shape)
        ]
        sampled_seq = seq[tuple(slices)]
        slices[axis] = -1
        last = seq[tuple(slices)]
        return np.concatenate([sampled_seq, last], axis=axis)


def apply_processors(batch, processors):
    for e in batch:
        for processor in processors:
            e = processor(e)

    return batch


def build_processors(processors_cfg):
    processors = []
    for processor_cfg in processors_cfg:
        processors += [global_registry.build_from_cfg(processor_cfg, "processor")]

    return processors
