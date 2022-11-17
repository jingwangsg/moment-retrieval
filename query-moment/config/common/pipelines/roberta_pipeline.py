from data.processor import *
from detectron2.config import LazyCall as L
import os.path as osp
from .default_bundle import processors as default_processors

cache_dir = "${paths.cache_dir}"
dataset = "${data.dataset}"
pretrained = "${data.pretrained}"
infer_roberta_txt = L(HuggingfaceInference)(
    model=L(AutoModel.from_pretrained)(
        pretrained_model_name_or_path="roberta-large"),
    extractor=L(AutoTokenizer.from_pretrained)(
        pretrained_model_name_or_path="roberta-large"),
    from_key="text",
    hash_key="text_id",
    cache_hdf5=osp.join(cache_dir, dataset, pretrained + ".hdf5"))

load_video_hdf5 = L(HDF5Loader)(
    hdf5_file="${data.dataset_dir}/${data.video_feat_type}.hdf5",
    from_key="video_id")

rename = L(Rename)(
    from_keys=["video_id.hdf5", "text.last_hidden_state"],
    to_keys=["video_feat", "text_feat"])

pipeline = dict(
    pretrained="roberta-large",
    video_feat_type="i3d",
    video_max_len=2048,
    collater="simple",
    processors=[load_video_hdf5, rename] + default_processors,
    pre_processors=[infer_roberta_txt])
