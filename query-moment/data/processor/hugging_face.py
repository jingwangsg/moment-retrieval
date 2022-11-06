from kn_util.general import global_registry, get_logger
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor
import os.path as osp
import os
import torch

log = get_logger(__name__)


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
        embeddings = model(**inputs).last_hidden_state.squeeze(0).cpu().detach().numpy()
        if self.cache_dir:
            np.save(cache_file, embeddings)
            log.info(f"embeddings save to {cache_file}")

        if self.to_embeddings:
            result[self.from_key + "_embeddings"] = embeddings

        return result
