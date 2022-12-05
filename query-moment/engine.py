from tqdm import tqdm
from evaluater import TrainEvaluater, ValTestEvaluater
from omegaconf import OmegaConf, DictConfig
from torchdata.dataloader2 import DataLoader2
from pprint import pprint

def train_one_epoch(model, train_loader: DataLoader2, val_loader: DataLoader2, train_evaluater: TrainEvaluater,
                   val_evaluater: ValTestEvaluater, cfg: DictConfig):

    for batch in tqdm(enumerate(train_loader)):
        losses = model(**batch, mode="train")
        train_evaluater.update_all(losses)

    train_metric_vals = train_evaluater.compute()
    pprint(train_metric_vals)

    

