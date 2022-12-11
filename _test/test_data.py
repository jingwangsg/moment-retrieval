import os.path as osp
import sys
import os.path as osp

sys.path.insert(0, "/export/home2/kningtg/WORKSPACE/moment-retrieval/query-moment")
from data.build import build_dataloader, build_datapipe
from tqdm import tqdm
import argparse


def test_dp_default():
    data_dir = "/export/home2/kningtg/WORKSPACE/moment-retrieval/data-bin/raw"
    dataset_dir = osp.join(data_dir, "${data.dataset}")
    cache_dir = "/export/home2/kningtg/WORKSPACE/moment-retrieval/data-bin/cache"
    from omegaconf import OmegaConf

    cfg = dict(data=dict(datapipe="msat",
                         dataset="activitynet",
                         dataset_dir=dataset_dir,
                         vid_hdf5="i3d.hdf5",
                         vid_hdf5_key_template="{video_id}",
                         txt_hdf5="roberta-large.txt.hdf5",
                         txt_hdf5_key_template="{text_id}/last_hidden_state",
                         max_len_video=256,
                         target_stride=2),
               train=dict(batch_size=16, num_workers=8, prefetch_factors=5),
               paths=dict(cache_dir=cache_dir),
               model_cfg=dict(arch="msat"))

    cfg = OmegaConf.create(cfg)
    datapipe = build_datapipe(cfg, "train")
    train_x = iter(datapipe).__next__()
    # datapipe = build_datapipe_default(cfg, "val")
    # val_x = iter(datapipe).__next__()
    # datapipe = build_datapipe_default(cfg, "test")
    # test_x = iter(datapipe).__next__()

    import ipdb
    ipdb.set_trace()  #FIXME


def test_dataloader():
    data_dir = "/export/home2/kningtg/WORKSPACE/moment-retrieval/data-bin/raw"
    dataset_dir = osp.join(data_dir, "${data.dataset}")
    cache_dir = "/export/home2/kningtg/WORKSPACE/moment-retrieval/data-bin/cache"
    from omegaconf import OmegaConf
    cfg = dict(data=dict(datapipe="default",
                         dataset="activitynet",
                         dataset_dir=dataset_dir,
                         vid_hdf5="i3d.hdf5",
                         vid_hdf5_key_template="{video_id}",
                         use_word_mask=True,
                         max_len_video=128),
               train=dict(batch_size=16, num_workers=8, prefetch_factor=5),
               paths=dict(cache_dir=cache_dir),
               flags=dict(ddp=False))
    cfg = OmegaConf.create(cfg)

    train_loader = build_dataloader(cfg, "train")
    # train_x = iter(train_loader).__next__()
    for x in tqdm(train_loader):
        pass
    val_loader = build_dataloader(cfg, "val")
    val_x = iter(val_loader).__next__()
    test_loader = build_dataloader(cfg, "test")
    test_x = iter(test_loader).__next__()
    import ipdb
    ipdb.set_trace()  #FIXME

if __name__ == "__main__":
    # test_distributed_dataloader()
    test_dp_default()
    # test_dataloader()
