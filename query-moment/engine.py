import torch
import torch.nn as nn
from data.datamodule.base import BaseDataModule

def train_one_epoch(model, datamodule: BaseDataModule, cfg):
    train_loader = datamodule.get_dataloader("train")
    test_loader = datamodule.get_dataloader("test")
    val_loader = datamodule.get_dataloader("val")

def evaluate(model, dataloader, )

def inference(model, dataloader, )