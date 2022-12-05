"""default pipeline with word masking"""
from data.processor import *
from detectron2.config import LazyCall as L
import os.path as osp
from .default_bundle import processors as default_processors
from kn_util.data import mask_safe

cache_dir = "${paths.cache_dir}"
dataset = "${data.dataset}"
dataset_dir = "${data.dataset_dir}"
pretrained = "${data.pretrained}"

tokenize_glove_txt = L(GloveTokenizer)(
    cache_dir=osp.join(cache_dir, ".glove"),
    to_embeddings=True,
    to_indices=True,
    from_key="text",
    #    vocab_file="${data.vocab_file}",
    upload_vocab_key="glove_vocab")

hdf5_file = "${data.dataset_dir}/${data.video_feat_type}.hdf5"
hdf5_key_template = "{video_id}"
delete_feature_nonexist = L(HDF5Checker)(hdf5_file=hdf5_file, key_template=hdf5_key_template)
load_video_hdf5 = L(HDF5Loader)(hdf5_file=hdf5_file, from_key="video_id")


def get_word_mask_op(result):
    text_mask = result["text.embs.mask"]
    valid_length = np.sum(text_mask)
    word_mask_inds = np.random.choice(np.arange(valid_length), size=int(np.ceil(valid_length * 0.15)))
    word_mask = np.zeros_like(text_mask)
    word_mask[word_mask_inds] = True
    # word_mask = mask_safe(mask)
    result["word_mask"] = word_mask
    return result


get_word_mask = L(Lambda)(_lambda=get_word_mask_op)

sample_video = L(SequenceSampler)(axis=0, max_len="${data.video_max_len}", from_key="video_id.hdf5")
pad_video = L(SequencePad)(from_key="video_id.hdf5.sample", axis=0, fill_value=0.0)
pad_text = L(SequencePad)(from_key="text.embs", axis=0, fill_value=0)
pad_word_label = L(SequencePad)(from_key="text.inds", axis=0, fill_value=0, return_mask=True)

rename = L(Rename)(from_keys=[
    "text.embs.pad",
    "text.embs.mask",
    "text.inds.pad",
    "video_id.hdf5.sample.pad",
    "video_id.hdf5.sample.mask",
],
                   to_keys=["txt_feat", "txt_mask", "word_label", "vid_feat", "vid_mask"])

collect = L(Collect)(from_keys=["txt_feat", "txt_mask", "word_label", "word_mask", "vid_feat", "vid_mask", "gt"])

pipeline = dict(
    video_feat_type="i3d",
    video_max_len=128,
    collater="simple",
    # vocab_file=osp.join(dataset_dir, "vocab.txt"),
    processors=[load_video_hdf5, sample_video, pad_video, pad_text, pad_word_label, get_word_mask, rename, collect],
    pre_processors=[tokenize_glove_txt, delete_feature_nonexist])
