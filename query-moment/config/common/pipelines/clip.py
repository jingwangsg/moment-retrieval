from detectron2.config import LazyCall as L
from data.processor import *
from transformers import AutoModel, AutoTokenizer
from transformers.models.clip import CLIPTextModel, CLIPTokenizer
import os.path as osp
from ..runtime import G

pipeline_cfg = dict(
    pretrained="openai/clip-vit-large-patch14-336",
    video_feat_type="clip.video.avg.avg",
    video_max_len=128,
    collater="default",
)

pretrained = "${data.pretrained}"
dataset = "${data.dataset}"
load_hf_embeddings_txt_no_load = L(HuggingfaceEmbedding)(
    model=L(CLIPTextModel.from_pretrained)(pretrained_model_name_or_path=pretrained),
    extractor=L(CLIPTokenizer.from_pretrained)(pretrained_model_name_or_path=pretrained),
    from_key="text",
    cache_args=dict(
        cache_dir=osp.join("${G.cache_dir}", dataset, pretrained),
        hash_key="sample_id",
        load_to_memory=False,
        overwrite=False,
    ),
)

load_hf_embeddings_txt_load = copy.deepcopy(load_hf_embeddings_txt_no_load)
load_hf_embeddings_txt_load["cache_args"]["load_to_memory"] = True
load_video_hdf5 = L(HDF5Loader)(
    hdf5_file=osp.join("${data.dataset_dir}", "${data.video_feat_type}.hdf5"),
    path_template="{}/patch",
    from_key="video_id",
)
sample_video = L(SequenceSampler)(
    axis=0, max_len="${data.video_max_len}", from_key="video_id_hdf5"
)
pad_txt = L(SequencePad)(
    from_key="text_embeddings", fill_value=0.0, axis=0, return_mask=True
)
pad_video = L(SequencePad)(
    from_key="video_id_hdf5_sample", fill_value=0.0, axis=0, return_mask=True
)
rename = L(Rename)(
    from_keys=[
        "video_id_hdf5_sample_pad",
        "video_id_hdf5_sample_mask",
        "text_embeddings_pad",
        "text_embeddings_mask",
    ],
    to_keys=["vid_feat", "vid_mask", "text_feat", "text_mask"],
)
collect = L(Collect)(from_keys=["vid_feat", "vid_mask", "text_feat", "text_mask", "gt"])

pipeline_cfg.update(
    dict(
        pre_processors=[load_hf_embeddings_txt_no_load],
        processors=[
            load_hf_embeddings_txt_load,
            pad_txt,
            load_video_hdf5,
            sample_video,
            pad_video,
            rename,
            collect,
        ],
    ),
)
del (
    load_hf_embeddings_txt_no_load,
    load_hf_embeddings_txt_load,
    load_video_hdf5,
    pad_txt,
    pad_video,
    rename,
    collect,
)
