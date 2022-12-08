from torchdata.dataloader2.reading_service import PrototypeMultiProcessingReadingService
from torchdata.dataloader2.dataloader2 import DataLoader2
from torch.utils.data import DataLoader
from .default import build_datapipe_default
from .datapipe.parse import *



def build_datapipe(cfg, split):
    if cfg.data.datapipe == "default":
        return build_datapipe_default(cfg, split=split)


def build_dataloader(cfg, split="train"):
    assert split in ["train", "test", "val"]
    datapipe = build_datapipe(cfg, split=split)

    reading_service = None
    if cfg.train.num_workers > 0:
        reading_service = PrototypeMultiProcessingReadingService(num_workers=cfg.train.num_workers)

    return DataLoader2(datapipe, reading_service=reading_service)

