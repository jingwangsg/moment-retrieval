from tqdm import tqdm
from evaluater import TrainEvaluater, ValTestEvaluater
from omegaconf import OmegaConf, DictConfig
from torchdata.dataloader2 import DataLoader2
from pprint import pprint
from lightning_lite import LightningLite
from kn_util.config import instantiate
from data.build import build_dataloader

def train_one_epoch(model, train_loader: DataLoader2, val_loader: DataLoader2, train_evaluater: TrainEvaluater,
                   val_evaluater: ValTestEvaluater, cfg: DictConfig):

    for batch in tqdm(enumerate(train_loader)):
        losses = model(**batch, mode="train")
        train_evaluater.update_all(losses)

    train_metric_vals = train_evaluater.compute()
    pprint(train_metric_vals)

def train(cfg):
    lite = LightningLite()
    model = instantiate(cfg.model)
    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val")
    test_loader = build_dataloader(cfg, split="test")

    optimizer = instantiate(cfg.train.optimizer, params=model.parameters())

    lite.setup(train_loader)

