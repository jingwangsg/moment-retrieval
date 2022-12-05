import os.path as osp
from .datapipe import *
from torchdata.datapipes.iter import IterableWrapper
from operator import itemgetter
import numpy as np


def get_word_mask(result, mask_rate=0.15):
    text_inds = result["text.inds"]

    length_text = len(text_inds)
    num_mask = int(np.ceil(length_text * 0.15))
    word_mask = np.zeros_like(text_inds, dtype=bool)
    if num_mask:
        mask_inds = np.random.choice(np.arange(length_text), size=num_mask)
        word_mask[mask_inds] = True

    result["word_mask"] = word_mask

    return result


def build_datapipe_default(cfg, split):
    dataset_dir = cfg.data.dataset_dir
    max_len_video = cfg.data.max_len_video
    vid_hdf5 = osp.join(dataset_dir, cfg.data.vid_hdf5)
    vid_hdf5_key_template = cfg.data.vid_hdf5_key_template
    batch_size = cfg.train.batch_size
    vocab_file = osp.join(dataset_dir, "annot", "vocab.txt")
    cache_dir = cfg.paths.cache_dir
    is_train = (split == "train")
    use_word_mask = cfg.data.get("use_word_mask", False) and is_train

    # parse dataset
    if cfg.data.dataset == "tacos":
        annot_path = osp.join(dataset_dir, "annot", split + ".json")
        annot_path_dp = IterableWrapper([annot_path])
        dataset_dp = annot_path_dp.open_files("r", encoding="utf-8").parse_json_files().map(
            itemgetter(1)).parse_tacos_tsgv()
    if cfg.data.dataset == "activitynet":
        annot_path = osp.join(dataset_dir, "annot", split + ".json")
        annot_path_dp = IterableWrapper([annot_path])
        dataset_dp = annot_path_dp.open_files("r", encoding="utf-8").parse_json_files().map(
            itemgetter(1)).parse_activitynet_tsgv()

    # filter nonexisted hdf5 key
    dataset_dp = dataset_dp.filter_by_hdf5_key(hdf5_file=vid_hdf5, key_template=vid_hdf5_key_template)

    if split == "train":
        dataset_dp = dataset_dp.shuffle()

    # tokenize text
    dataset_dp = dataset_dp.tokenize_glove(from_key="text",
                                           vocab_file=vocab_file,
                                           cache_dir=osp.join(cache_dir, ".glove"),
                                           to_embeddings=True,
                                           to_indices=True)

    if is_train:
        dataset_dp = dataset_dp.shuffle()
    # load video feature
    dataset_dp = dataset_dp.load_hdf5(hdf5_file=vid_hdf5, key_template=vid_hdf5_key_template, output_key_prefix="video")
    # sample video feature
    dataset_dp = dataset_dp.sample_sequence(from_key="video.hdf5", axis=0, max_len=max_len_video, inplace=True)

    # word mask
    if use_word_mask:
        dataset_dp = dataset_dp.map(get_word_mask)

    # batchify
    dataset_dp = dataset_dp.batch(batch_size).rows2columnar()

    # pad
    dataset_dp = dataset_dp.pad_sequence(from_key="video.hdf5", axis=0, fill_value=0.0)
    dataset_dp = dataset_dp.pad_sequence(from_key="text.embs", axis=0, fill_value=0.0)
    dataset_dp = dataset_dp.pad_sequence(from_key="text.inds", axis=0, fill_value=0, return_mask=False)

    # collect
    if use_word_mask:
        dataset_dp = dataset_dp.pad_sequence(from_key="word_mask", axis=0, fill_value=False, return_mask=False)
        dataset_dp = dataset_dp.collect(
            ["video.hdf5.pad", "video.hdf5.mask", "text.embs.pad", "text.embs.mask", "text.inds.pad", "word_mask.pad"],
            ["vid_feat", "vid_mask", "txt_feat", "txt_mask", "word_labels", "word_mask"])
    else:
        dataset_dp = dataset_dp.collect(["video.hdf5.pad", "video.hdf5.mask", "text.embs.pad", "text.embs.mask"],
                                        ["vid_feat", "vid_mask", "txt_feat", "txt_mask"])

    dataset_dp = dataset_dp.collate(default_collate_fn)
    return dataset_dp
