import sys
import os.path as osp
import copy
from kn_util.debug import SignalContext
import torch
from tqdm import tqdm

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


class TSGVDataModule(pl.LightningDataModule):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.load_data()
        # self.prepare_data()

    def prepare_data(self) -> None:
        if self.cfg.data.pipeline_verbose:
            self.sanity_check()
        self.preprocess()

    def load_data(self):
        """to be implemented in sub-class"""
        self.dataset = dict()

    def sanity_check(self):
        cfg = self.cfg
        x = copy.copy(self.datasets["train"][:4])
        with SignalContext(_signal, True):
            pre_processor = build_processors(self.cfg.data.pre_processors, verbose=True)
            x = apply_processors(x, pre_processor)
            collater = self.build_collater("train")
            x = collater(x)
            log.info("\n" + explore_content(x, "collater output"))

        del pre_processor
        del collater
        del x

    def preprocess(self):
        if not hasattr(self.cfg.data, "pre_processors"):
            return
        self.pre_processor = build_processors(self.cfg.data.pre_processors)

        for domain in ["train", "val", "test"]:
            dataset = self.datasets[domain]

            self.datasets[domain] = apply_processors(
                dataset,
                self.pre_processor,
                tqdm_args=dict(
                    desc=f"preprocess {domain}", total=len(dataset)),
            )

    def build_collater(self, domain, verbose=False):
        cfg = self.cfg
        processors = build_processors(cfg.data.processors, verbose=verbose)
        self.dataloaders = dict()
        collater = registry.build_collater(
            cfg.data.collater,
            cfg=cfg,
            processors=processors,
            is_train=(domain == "train"),
        )
        return collater

    def build_sampler(self, domain):
        if domain == "train":
            return RandomSampler(self.datasets[domain])
        else:
            return SequentialSampler(self.datasets[domain])

    def build_dataloaders(self, domain):
        cfg = self.cfg
        collater = self.build_collater(domain)
        sampler = self.build_sampler(domain)
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
