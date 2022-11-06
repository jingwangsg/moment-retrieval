import torch
from kn_util.general import global_registry, get_logger
from kn_util.data import delete_noisy_char, generate_sample_indices
import h5py
import numpy as np
import decord as de
import os
import os.path as osp
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModel
from torchtext.data.utils import get_tokenizer
import torchtext

log = get_logger(__name__)


@global_registry.register_processor("glove_tokenize")
class GloveTokenizer:
    def __init__(
        self,
        glove="",
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
        self.to_words = to_words
        self.to_indices = to_indices
        self.to_embeddings = to_embeddings

        if tokenizer == "split":
            self.tokenizer = lambda s: delete_noisy_char(s).split()
        else:
            self.tokenizer = get_tokenizer(tokenizer)

    def _load_vocab(self):
        pretrained_vocab = torchtext.vocab.pretrained_aliases[self.glove](
            specials=["<pad>", ["<unk>"]]
        )
        if self.vocab_file:
            with open(self.vocab_file, "r") as f:
                lines = f.readlines()
            itos = [w.strip() for w in lines]
            extracted_indicies = [pretrained_vocab.stoi.get(w, 1) for w in itos]
            vectors = pretrained_vocab.vectors[extracted_indicies]
        else:
            itos = pretrained_vocab.itos
            vectors = pretrained_vocab.vectors

        stoi = {w: idx for idx, w in enumerate(itos)}
        self.itos = itos
        self.stoi = stoi
        self.vectors = vectors

        if self.upload_vocab_key:
            global_registry.register_object(self.upload_vocab_key, vectors)

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


@global_registry.register_processor("load_hdf5")
class LoadHDF5:
    def __init__(self, hdf5_file, from_key) -> None:
        self.handle = h5py.File(hdf5_file, "r")
        self.from_key = from_key

    def __call__(self, result):
        ind = result[self.from_key]
        result[self.from_key + "_hdf5"] = self.handle[ind]

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
        self, pretrained=None, is_text=True, from_key=None, hash_key=None, cache_dir=None
    ):
        assert from_key is not None and pretrained is not None
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
        self.from_key = from_key
        self.hash_key = hash_key

        if cache_dir is not None:
            os.makedirs(cache_dir)

    @torch.no_grad()
    def __call__(self, result):
        data = result[self.from_key]
        hash_id = hash(data) if self.hash_key is None else result[self.hash_key]

        if not self.cache_dir:
            cache_file = osp.join(self.cache_dir, f"{hash_id}.npy")
            if osp.exists(cache_file):
                try:
                    embeddings = np.load(cache_file)
                    load_fail = False
                except:
                    load_fail = True
                if not load_fail:
                    result[self.from_key + "_embeddings"] = embeddings
                return result

        model = model.cuda()
        inputs = self.extractor(data, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        embeddings = (
            self.model(**inputs).last_hidden_state.squeeze(0).cpu().detach().numpy()
        )
        if not self.cache_dir:
            np.save(cache_file, embeddings)

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
        return np.concatenate([sampled_seq, last], axis=axis)


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
