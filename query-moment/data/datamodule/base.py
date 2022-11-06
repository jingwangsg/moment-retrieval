from torch.utils.data import SequentialSampler, RandomSampler, DistributedSampler, DataLoader
from ..processors import build_processors, apply_processors


class BaseDataModule:
    def __init__(self, cfg):
        self.cfg = cfg
        self.load_data()
        self.preprocess()
        self.build_dataloaders()

    def load_data(self):
        """to be implemented in sub-class"""
        self.dataset = dict()

    def preprocess(self):
        if not hasattr(self.cfg.data, "pre_processors"):
            return
        pre_processor = build_processors(self.cfg.data.pre_processors)

        for domain in ["train", "val", "test"]:
            dataset = self.dataset[domain]
            self.dataset[domain] = apply_processors(dataset, pre_processor)

    def _build_sampler(self, domain):
        if domain == "train":
            if self.cfg.get("ddp", False):
                return DistributedSampler(self.dataset[domain])
            else:
                return RandomSampler(self.dataset[domain])
        else:
            return SequentialSampler(self.dataset[domain])

    def build_dataloaders(self):
        cfg = self.cfg
        self.dataloaders = dict()
        for domain in ["train", "val", "test"]:
            sampler = self._build_sampler(domain)
            self.dataloaders[domain] = DataLoader(
                self.dataset[domain],
                batch_size=cfg.train.batch_size,
                prefetch_factor=cfg.train.prefetch_factor,
                num_workers=cfg.train.num_workers,
            )

    def get_dataloader(self, domain):
        return self.dataloaders[domain]
