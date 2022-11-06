from transformers.models.clip import CLIPVisionModel, CLIPFeatureExtractor
import os.path as osp
import os
import argparse
import decord as de
from PIL import Image
from typing import List
from functools import partial
import torch
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm

sys.path.insert(0, osp.join(osp.dirname(__file__), ".."))
from kn_util.general import global_registry
from kn_util.nn import freeze_module
from kn_util.data import generate_sample_indices
import glob
import h5py
import subprocess
import numpy as np
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ActivityNet", "TACoS", "Charades"])
    parser.add_argument("--stride", default=16, type=int)
    parser.add_argument("--pretrained", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--max-len", default=512, type=int)
    parser.add_argument("--temporal-pool", default="avg", type=str)
    parser.add_argument("--spatial-pool", default="avg", type=str)

    return parser.parse_args()


def inference_single_video(model, video_path, hdf5_handle: h5py.File, args):
    model_config = global_registry.get_object("model_config")
    vr = de.VideoReader(video_path)
    tot_len = len(vr)

    sampled_index = generate_sample_indices(tot_len, max_len=args.max_len)
    sampled_arr = vr.get_batch(sampled_index).asnumpy()
    sampled_image = [Image.fromarray(sampled_arr[i]) for i in range(sampled_arr.shape[0])]

    video_id = osp.basename(video_path)[:-4]

    extractor = CLIPFeatureExtractor.from_pretrained(args.pretrained)
    collate_fn = partial(extractor, return_tensors="pt")
    dataloader = DataLoader(
        sampled_image,
        batch_size=args.stride,
        collate_fn=collate_fn,
        num_workers=6,
        prefetch_factor=5,
    )
    cls_hidden_list = []
    patch_hidden_list = []
    for batch in dataloader:
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = model(**batch)
        outputs = outputs.last_hidden_state
        cls_hidden = outputs[:, 0, :]
        num_pixel = model_config.image_size // model_config.patch_size
        patch_hidden = outputs[:, 1:, :]
        patch_hidden_flatten = patch_hidden.reshape(patch_hidden.shape[0], -1)
        patch_hidden_flatten = patch_hidden_flatten.transpose(0, 1).unsqueeze(0)
        temporal_pool = F.avg_pool1d if args.temporal_pool == "avg" else F.max_pool1d
        if patch_hidden_flatten.shape[0] >= 8:
            patch_hidden = temporal_pool(patch_hidden_flatten, kernel_size=4, stride=4)
        else:
            patch_hidden = torch.mean(patch_hidden_flatten, dim=0, keepdim=True)

        patch_hidden = patch_hidden.squeeze(0).transpose(0, 1)
        patch_hidden = patch_hidden.reshape(
            (patch_hidden.shape[0], num_pixel, num_pixel, model_config.hidden_size)
        )
        patch_hidden = patch_hidden.permute([0, 3, 1, 2])
        spatial_pool = F.avg_pool2d if args.spatial_pool == "avg" else F.max_pool2d
        patch_hidden = spatial_pool(patch_hidden, kernel_size=4, stride=4).permute([0, 2, 3, 1])

        cls_hidden_list += [cls_hidden.cpu().detach().numpy()]
        patch_hidden_list += [patch_hidden.cpu().detach().numpy()]
    cls_hidden = np.concatenate(cls_hidden_list, axis=0)
    patch_hidden = np.concatenate(patch_hidden_list, axis=0)
    cur_handle = hdf5_handle.create_group(video_id)
    cur_handle.create_dataset("cls", data=cls_hidden)
    cur_handle.create_dataset("patch", data=patch_hidden)


if __name__ == "__main__":
    args = parse_args()
    video_dir = osp.expanduser(f"~/DATASET/TSGV_Data/videos/{args.dataset}")
    hdf5_path = osp.expanduser(
        f"~/DATASET/TSGV_Data/clip/{args.dataset}/clip.video.{args.temporal_pool}.{args.spatial_pool}.hdf5"
    )
    subprocess.run(f"rm -rf {hdf5_path}", shell=True)

    model = CLIPVisionModel.from_pretrained(args.pretrained)
    model = model.cuda()
    model.eval()
    freeze_module(model)
    global_registry.register_object("model_config", model.config)
    # collate_fn = partial(collate_fn_builder, extractor=extractor)

    video_paths = glob.glob(osp.join(video_dir, "*"))
    os.makedirs(osp.dirname(hdf5_path), exist_ok=True)
    hdf5_handle = h5py.File(hdf5_path, "w")
    for video_path in tqdm(video_paths):
        inference_single_video(model, video_path, hdf5_handle, args)
    hdf5_handle.close()
