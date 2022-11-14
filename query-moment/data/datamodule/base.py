import sys
import os.path as osp
import copy
from kn_util.debug import SignalContext

sys.path.insert(0, osp.join(osp.dirname(__file__), "../.."))
from torch.utils.data import (
    SequentialSampler,
    RandomSampler,
    DistributedSampler,
    DataLoader,
)
from data.processor import build_processors, apply_processors
from kn_util.general import registry, get_logger
from kn_util.debug import explore_content
import pytorch_lightning as pl

log = get_logger(__name__)
_signal = "_TEST_PIPELINE_SIGNAL"


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
    
    def prepare_data(self) -> None:
        self.load_data()
        self.preprocess()

    def load_data(self):
        """to be implemented in sub-class"""
        self.dataset = dict()

    def preprocess(self):
        if not hasattr(self.cfg.data, "pre_processors"):
            return
        self.pre_processor = build_processors(self.cfg.data.pre_processors)

        for domain in ["train", "val", "test"]:
            dataset = self.datasets[domain]

            if domain != "train":
                registry.set_object(_signal, False)

            with SignalContext(_signal, self.cfg.G.debug and domain == "train"):
                self.datasets[domain] = apply_processors(
                    dataset,
                    self.pre_processor,
                    tqdm_args=dict(desc=f"preprocess {domain}", total=len(dataset)),
                )

    def _build_sampler(self, domain):
        if domain == "train":
            return RandomSampler(self.datasets[domain])
        else:
            return SequentialSampler(self.datasets[domain])

    def build_dataloaders(self, domain):
        cfg = self.cfg
        processors = build_processors(cfg.data.processors)
        self.dataloaders = dict()
        collater = registry.build_collater(
            cfg.data.collater,
            cfg=cfg,
            processors=processors,
            is_train=(domain == "train"),
        )
        sampler = self._build_sampler(domain)
        return DataLoader(
            self.datasets[domain],
            batch_size=cfg.train.batch_size,
            sampler=sampler,
            prefetch_factor=cfg.train.prefetch_factor,
            num_workers=cfg.train.num_workers,
            collate_fn=collater,
        )

    def train_dataloader(self):
        return self.build_dataloaders("train")
    
    def val_dataloader(self):
        return self.build_dataloaders("val")
    
    def test_dataloader(self):
        return self.build_dataloaders("test")
    
    
