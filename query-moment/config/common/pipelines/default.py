from data.processor import *
from detectron2.config import LazyCall as L
import os.path as osp

glove_tokenize = L(GloveTokenizer)(
    vocab_file="${data.dataset_dir}/annot/vocab.txt",
    from_key="text",
    to_indices=True,
    upload_vocab_key="glove_vocab",
    cache_dir="${G.cache_dir}",
)

load_video_hdf5 = L(HDF5Loader)(
    hdf5_file="${data.dataset_dir}/${data.video_feat_type}.hdf5", from_key="video_id"
)

sample_video = L(SequenceSampler)(axis=0, max_len=128, from_key="video_id_hdf5")
pad_video = L(SequencePad)(from_key="video_id_hdf5_sample", axis=0, fill_value=0.0)
pad_text = L(SequencePad)(from_key="text_inds", axis=0, fill_value=0)
rename = L(Rename)(
    from_keys=[
        "text_inds_pad",
        "text_inds_mask",
        "video_id_hdf5_sample_pad",
        "video_id_hdf5_sample_mask",
    ],
    to_keys=["text_inds", "text_mask", "vid_feat", "vid_mask"],
)
collect = L(Collect)(from_keys=["text_inds", "text_mask", "vid_feat", "vid_mask", "gt"])

pipeline_cfg = dict(
    video_feat_type="i3d",
    pre_processors=[glove_tokenize, pad_text],
    processors=[load_video_hdf5, sample_video, pad_video, rename, collect],
    collater="default",
)
