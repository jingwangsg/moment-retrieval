from torchdata.dataloader2.reading_service import PrototypeMultiProcessingReadingService
from torchdata.dataloader2.dataloader2 import DataLoader2
# from torch.utils.data import DataLoader
# from .datapipe.parse import *
from models import *
from kn_util.general import registry


def build_datapipe(cfg, split):
    return registry.build_datapipe(cfg.model_cfg.arch, cfg=cfg, split=split)
    # if cfg.data.datapipe == "default":
    #     return build_datapipe_default(cfg, split=split)
    # if cfg.data.datapipe == "msat":
    #     return build_datapipe_msat(cfg, split=split)


def build_dataloader(cfg, split="train"):
    assert split in ["train", "test", "val"]
    datapipe = build_datapipe(cfg, split=split)

    reading_service = None
    if cfg.train.num_workers > 0:
        reading_service = PrototypeMultiProcessingReadingService(num_workers=cfg.train.num_workers)

    dataloader = DataLoader2(datapipe, reading_service=reading_service)
    dataloader.num_batches = len(datapipe)
    return dataloader
