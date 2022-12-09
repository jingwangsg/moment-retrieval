import torch
import argparse
from kn_util.general import registry, get_logger
from kn_util.config import LazyConfig
from kn_util.general import yapf_pformat
from engine import train_one_epoch, evaluate
from kn_util.config import instantiate, serializable
from kn_util.amp import NativeScalerWithGradNormCount
from data.build import build_dataloader
from evaluater import TrainEvaluater, ValTestEvaluater, AverageMeter
import torch
from kn_util.general import get_logger
from kn_util.nn_utils import CheckPointer
from kn_util.file import save_json, save_pickle
from misc import dict2str
from lightning_lite.utilities.seed import seed_everything
import time
from omegaconf import OmegaConf
import os.path as osp
from pprint import pformat
import subprocess


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("cfg", type=str)
    args.add_argument("--cfg-override", "-co", nargs="+", default=[])
    args.add_argument("--resume", action="store_true", default=False)
    args.add_argument("--no-multiproc", action="store_true", default=False)
    args.add_argument("--no-callback", action="store_true", default=False)
    args.add_argument("--exp", required=True, type=str)

    return args.parse_args()


def main(args):
    cfg = LazyConfig.load(args.cfg)
    LazyConfig.apply_overrides(cfg, args.cfg_override)
    cfg.flags.exp = args.exp

    if args.resume:
        pass
    else:
        subprocess.run(f"rm -rf {cfg.paths.work_dir}/*", shell=True)

    logger = get_logger(output_dir=cfg.paths.work_dir)

    logger.info(pformat(cfg))
    OmegaConf.save(cfg, osp.join(cfg.paths.work_dir, "config.yaml"), resolve=False)
    # resume training if possible

    # debug flag
    if args.no_multiproc:
        cfg.train.prefetch_factor = 2
        cfg.train.num_workers = 0

    # use_amp = cfg.flags.amp
    seed_everything(cfg.flags.seed)
    use_ddp = cfg.flags.ddp
    use_amp = cfg.flags.amp

    model = instantiate(cfg.model)
    model = model.cuda()
    with open("./tmp.txt", "w") as f:
        f.write("\n".join([f"{k} {v.flatten()[-1].item():.4f}" for k, v in model.state_dict().items()]))

    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val")
    test_loader = build_dataloader(cfg, split="test")

    train_evaluater = TrainEvaluater(cfg)
    val_evaluater = ValTestEvaluater(cfg, "val")

    optimizer = instantiate(cfg.train.optimizer, params=model.parameters())
    lr_scheduler = instantiate(cfg.train.lr_scheduler, optimizer=optimizer) if hasattr(cfg.train,
                                                                                       "lr_scheduler") else None
    loss_scaler = NativeScalerWithGradNormCount() if use_amp else None

    logger = get_logger(output_dir=cfg.paths.work_dir)

    num_epochs = cfg.train.num_epochs
    ckpt = CheckPointer(monitor=cfg.eval.best_monitor, work_dir=cfg.paths.work_dir, mode=cfg.eval.is_best)

    logger.info("==============START TRAINING=============")

    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, train_evaluater, val_loader, val_evaluater, ckpt, optimizer, lr_scheduler,
                        loss_scaler, epoch, logger, cfg)

    # evaluate on best validation
    st = time.time()
    ckpt.load_checkpoint(model, optimizer, lr_scheduler, mode="best")
    metric_vals = evaluate(model, val_loader, val_evaluater, "val", cfg)
    logger.info("=========BEST VALIDATION RESULT==========")
    logger.info(f'{dict2str(metric_vals)}\t', f'eta {time.time() - st:.4f}')


if __name__ == "__main__":
    args = parse_args()
    main(args)