import torch
from kn_util.general import global_registry, get_logger
from kn_util.data import delete_noisy_char, generate_sample_indices, general_pad
from kn_util.file import load_pickle
from kn_util.debug import explore_content, dict_diff
import h5py
import numpy as np
import decord as de
import os
import os.path as osp
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModel
from torchtext.data.utils import get_tokenizer
import torchtext
import copy
from functools import partial
from pprint import pformat
import time

log = get_logger(__name__)


@global_registry.register_processor("glove_tokenize")
class GloveTokenizer:
    def __init__(
        self,
        glove="glove.6B.300d",
        vocab_file=None,
        upload_vocab_key=None,
        tokenizer="split",
        from_key=None,
        cache_dir=None,
        to_words=False,
        to_indices=False,
        to_embeddings=False,
    ) -> None:
        assert from_key is not None
        assert cache_dir is not None
        assert to_words or to_indices or to_embeddings
        self.from_key = from_key
        self.vocab_file = vocab_file
        self.glove = glove
        self.upload_vocab_key = upload_vocab_key
        self.cache_dir = cache_dir
        self.to_words = to_words
        self.to_indices = to_indices
        self.to_embeddings = to_embeddings

        if tokenizer == "split":
            self.tokenizer = lambda s: delete_noisy_char(s).lower().split()
        else:
            self.tokenizer = get_tokenizer(tokenizer)

        self._load_vocab()

    def _load_vocab(self):
        pretrained_vocab = torchtext.vocab.pretrained_aliases[self.glove](
            cache=self.cache_dir
        )
        if self.vocab_file:
            with open(self.vocab_file, "r") as f:
                lines = f.readlines()
            itos = [w.strip() for w in lines]
            extracted_indicies = [pretrained_vocab.stoi.get(w, 1) for w in itos[1:]]
            vectors = pretrained_vocab.vectors[extracted_indicies]
            vectors = torch.concat(
                [torch.zeros((1, vectors.shape[-1]), dtype=vectors.dtype), vectors], dim=0
            )
        else:
            itos = ["<unk>", "<pad>"] + pretrained_vocab.itos
            vectors = pretrained_vocab.vectors
            vectors = torch.concat(
                [
                    torch.zeros((1, vectors.shape[-1]), dtype=vectors.dtype),  # <pad>
                    pretrained_vocab["<unk>"].unsqueeze(0),  # <unk>
                    vectors,
                ],
                dim=0,
            )

        stoi = {w: idx for idx, w in enumerate(itos)}
        self.itos = itos
        self.stoi = stoi
        self.vectors = vectors.float().numpy()

        log.info(f"glove vocab built with {len(itos)} words")

        if self.upload_vocab_key:
            global_registry.register_object(self.upload_vocab_key, (itos, vectors))

        del pretrained_vocab

    def __call__(self, result):
        text = result[self.from_key]
        text_tok = self.tokenizer(text)
        text_inds = np.array([self.stoi.get(w, 1) for w in text_tok])
        text_embeddings = np.stack([self.vectors[ind] for ind in text_inds], axis=0)
        if self.to_words:
            result[self.from_key + "_tok"] = text_tok
        if self.to_indices:
            result[self.from_key + "_inds"] = text_inds
        if self.to_embeddings:
            result[self.from_key + "_embs"] = text_embeddings

        return result


@global_registry.register_processor("batch.seq_pad")
class SequencePad:
    is_batch_processor = True

    def __init__(self, from_key, return_mask=True, **kwargs) -> None:
        assert from_key
        self.from_key = from_key
        self.kwargs = kwargs
        self.return_mask = return_mask

    def __call__(self, batch) -> None:
        seqs = [e[self.from_key] for e in batch]
        ret = general_pad(seqs, return_mask=self.return_mask, **self.kwargs)
        if self.return_mask:
            padded_val, padded_mask = ret
        else:
            padded_val = ret
        for idx, e in enumerate(batch):
            e[self.from_key + "_pad"] = padded_val[idx]
            if self.return_mask:
                e[self.from_key + "_mask"] = padded_mask[idx]

        return batch


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

            if not load_fail:
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
        pretrained=None,
        is_text=True,
        from_key=None,
        hash_key=None,
        cache_dir=None,
        to_embeddings=False,
    ):
        assert from_key is not None and pretrained is not None
        assert cache_dir or to_embeddings
        # to_embeddings=False means for latter use but not load it into memory now
        log.warn("[processor] hf_embedding may use cuda")
        self.cache_dir = cache_dir
        self.is_text = is_text
        model = AutoModel.from_pretrained(pretrained)
        model.eval()
        self.model = model
        self.extractor = (
            AutoTokenizer.from_pretrained(pretrained)
            if is_text
            else AutoFeatureExtractor.from_pretrained(pretrained)
        )
        self.to_embeddings = to_embeddings
        self.from_key = from_key
        self.hash_key = hash_key

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

    @torch.no_grad()
    def __call__(self, result):
        data = result[self.from_key]
        hash_id = hash(data) if self.hash_key is None else result[self.hash_key]

        if self.cache_dir:
            cache_file = osp.join(self.cache_dir, f"{hash_id}.npy")
            if osp.exists(cache_file):
                try:
                    embeddings = np.load(cache_file)
                    load_fail = False
                except:
                    load_fail = True
                if not load_fail and self.to_embeddings:
                    result[self.from_key + "_embeddings"] = embeddings
                return result

        model = self.model.cuda()
        inputs = self.extractor(data, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        embeddings = (
            model(**inputs).last_hidden_state.squeeze(0).cpu().detach().numpy()
        )
        if self.cache_dir:
            np.save(cache_file, embeddings)
            log.info(f"embeddings save to {cache_file}")

        if self.to_embeddings:
            result[self.from_key + "_embeddings"] = embeddings

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
        result[self.from_key + "_sample"] = np.concatenate(
            [sampled_seq, np.expand_dims(last, axis)], axis=axis
        )
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
        if not self.cache_dir:
            cache_file = osp.join(self.cache_dir, f"{video_id}.npy")
            if osp.exists(cache_file):
                try:
                    arr = np.load(cache_file)
                    load_fail = False
                except:
                    load_fail = True
                if not load_fail:
                    result[self.from_key + "_imgs"] = arr
                    return result
        video_path = self.video_format.format(video_id)
        vr = de.VideoReader(video_path)
        tot_len = len(vr)

        sampled_indices = generate_sample_indices(
            tot_len, self.max_len, self.stride, output_stride=True
        )
        arr = vr.get_batch(sampled_indices).asnumpy()

        if not self.cache_dir:
            np.save(cache_file, arr)

        result[self.from_key + "_imgs"] = arr
        return result


@global_registry.register_processor("delete")
class Delete:
    def __init__(self, from_keys=None) -> None:
        assert from_keys
        self.from_keys = from_keys

    def __call__(self, result):

        for k in self.from_keys:
            del result[k]
        return result


@global_registry.register_processor("rename")
class Rename:
    def __init__(self, from_keys=None, to_keys=None) -> None:
        assert from_keys and to_keys
        self.from_keys = from_keys
        self.to_keys = to_keys

    def __call__(self, result):
        _result = dict()
        for from_key, to_key in zip(self.from_keys, self.to_keys):
            _result[to_key] = result[from_key]
            del result[from_key]
        result.update(_result)

        return result


@global_registry.register_processor("collect")
class Collect:
    def __init__(self, from_keys=[]):
        self.from_keys = from_keys

    def __call__(self, result):
        _result = dict()
        for k in self.from_keys:
            _result[k] = result[k]
        return _result


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
