from data.processor import *
from detectron2.config import LazyCall as L
import os.path as osp
from .default_bundle import processors as default_processors

cache_dir = "${paths.cache_dir}"
dataset = "${data.dataset}"
pretrained = "${data.pretrained}"

infer_glove_txt = L(GloveTokenizer)(
    cache_dir=osp.join(cache_dir, ".glove"),
    to_embeddings=True,
    from_key="text")

hdf5_file = "${data.dataset_dir}/${data.video_feat_type}.hdf5"
hdf5_key_template = "{video_id}"
delete_feature_nonexist = L(HDF5Checker)(
    hdf5_file=hdf5_file, key_template=hdf5_key_template)
load_video_hdf5 = L(HDF5Loader)(hdf5_file=hdf5_file, from_key="video_id")

rename = L(Rename)(
    from_keys=["text.embs", "video_id.hdf5"],
    to_keys=["text_feat", "video_feat"])

collect = L(Collect)(
    from_keys = ["text_feat", "text_mask", "vid_feat", "vid_mask", "gt"]
)

pipeline = dict(
    video_feat_type="i3d",
    video_max_len=2048,
    collater="simple",
    processors=[load_video_hdf5, rename, *default_processors, collect],
    pre_processors=[infer_glove_txt, delete_feature_nonexist])
