from tqdm import tqdm
from evaluater import TrainEvaluater, ValTestEvaluater
from omegaconf import OmegaConf, DictConfig
from torchdata.dataloader2 import DataLoader2
from pprint import pprint
from lightning_lite import LightningLite
from kn_util.config import instantiate
from kn_util.amp import NativeScalerWithGradNormCount
from data.build import build_dataloader
from evaluater import TrainEvaluater, ValTestEvaluater, AverageMeter
import torch
from kn_util.general import get_logger
from kn_util.data import collection_to_device
from misc import dict2str
import time
from lightning_lite.utilities.seed import seed_everything
from kn_util.distributed import initialize_ddp_from_env
from kn_util.nn_utils import CheckPointer
from tqdm import tqdm
from torch import nn
import contextlib
from torch.optim.lr_scheduler import ReduceLROnPlateau


@torch.no_grad()
def evaluate(model, loader, evaluater, cfg):
    # evaluater = ValTestEvaluater(cfg, domain=domain)
    for batch in tqdm(loader, desc="evaluating"):
        batch = collection_to_device(batch, "cuda")
        out = model(**batch, mode="inference")
        out.update(batch)
        out = collection_to_device(out, "cpu")
        evaluater.update_all(out)

    metric_vals = evaluater.compute_all()

    return metric_vals


def train_one_epoch(model, train_loader, train_evaluater, val_loader, val_evaluater, ckpt, optimizer, lr_scheduler,
                    loss_scaler, cur_epoch, logger, cfg):
    grad_norm_meter = AverageMeter()
    batch_eta = AverageMeter()
    use_amp = cfg.flags.amp

    num_batches_train = train_loader.num_batches
    val_interval = int(cfg.train.val_interval * num_batches_train)
    print_interval = int(cfg.train.print_interval * num_batches_train)

    for idx, batch in enumerate(train_loader):
        st = time.time()
        model.train()
        # update_grad = ((idx + 1) % cfg.train.accum_grad_steps == 0)
        update_grad = True
        use_print = ((idx + 1) % print_interval == 0)
        use_validate = ((idx + 1) % val_interval == 0)

        # forward
        with torch.autocast("cuda", enabled=use_amp):
            batch = collection_to_device(batch, "cuda")
            losses = model(**batch, mode="train")
            loss = losses["loss"]

        # backward & optimize
        if use_amp:
            grad_norm = loss_scaler(loss, optimizer, clip_grad=cfg.train.clip_grad, parameters=model.parameters())
            grad_norm = grad_norm.item()
        else:
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.clip_grad).item()
            optimizer.step()
        grad_norm_meter.update(grad_norm)

        if update_grad:
            optimizer.zero_grad()

        # update training metrics
        batch_eta.update(time.time() - st)
        losses = collection_to_device(losses, "cpu")
        train_evaluater.update_all(losses)

        if use_validate:
            st = time.time()
            metric_vals = evaluate(model, val_loader, val_evaluater, cfg)
            logger.info(f'{dict2str(metric_vals)}\t'
                        f'eta {time.time() - st:.4f}')

            # save checkpoint
            ckpt.save_checkpoint(model,
                                 optimizer,
                                 num_epochs=cur_epoch,
                                 metric_vals=metric_vals,
                                 lr_scheduler=lr_scheduler)

        if use_print:
            train_losses = train_evaluater.compute_all()
            losses_str = dict2str(train_losses)
            mem = torch.cuda.max_memory_allocated()
            grad_norm_avg = grad_norm_meter.compute()
            batch_eta_avg = batch_eta.compute()
            logger.info(f"Train Epoch{cur_epoch:4d} [{idx}/{num_batches_train}]\t"
                        f"{losses_str}\t"
                        f"grad_norm {grad_norm_avg:.4f}\t"
                        f"mem {mem / (1024.0 * 1024.0): .0f}MB\t"
                        f"batch_eta {batch_eta_avg: .4f}")
