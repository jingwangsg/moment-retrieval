import torch
import argparse
from kn_util.basic import registry, get_logger
from kn_util.config import LazyConfig
from kn_util.basic import yapf_pformat
from engine import train_one_epoch, evaluate
from kn_util.config import instantiate, serializable
from kn_util.nn_utils.amp import NativeScalerWithGradNormCount
from data.build import build_dataloader
from evaluater import TrainEvaluater, ValTestEvaluater, ScalarMeter
import torch
from kn_util.basic import get_logger
from kn_util.nn_utils import CheckPointer
from kn_util.basic.file import save_json, save_pickle
from misc import dict2str
from lightning_lite.utilities.seed import seed_everything
import time
from omegaconf import OmegaConf
import os.path as osp
from pprint import pformat
import subprocess
import wandb


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("cfg", type=str)
    args.add_argument("--cfg-override", "-co", nargs="+", default=[])
    args.add_argument("--resume", action="store_true", default=False)
    args.add_argument("--no-multiproc", action="store_true", default=False)
    args.add_argument("--no-callback", action="store_true", default=False)
    args.add_argument("--wandb", action="store_true", default=False)
    args.add_argument("--exp", required=True, type=str)

    return args.parse_args()


def main(args):
    cfg = LazyConfig.load(args.cfg)
    LazyConfig.apply_overrides(cfg, args.cfg_override)
    cfg.flags.exp = args.exp
    cfg.flags.wandb = args.wandb

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

    # wandb init
    if args.wandb:
        wandb.init(config=OmegaConf.to_container(cfg, resolve=True), project="query-moment", name=cfg.flags.exp)
        wandb.run.log_code(
            cfg.paths.root_dir,
            include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"),
            exclude_fn=lambda path: "logs/" in path,
        )

    # use_amp = cfg.flags.amp
    seed_everything(cfg.flags.seed)
    use_ddp = cfg.flags.ddp
    use_amp = cfg.flags.amp


    # build dataloader
    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val")
    test_loader = build_dataloader(cfg, split="test")

    # instantiate model
    model = instantiate(cfg.model)
    model = model.cuda()

    # build evaluater
    train_evaluater = TrainEvaluater(cfg)
    val_evaluater = ValTestEvaluater(cfg)
    test_evaluater = ValTestEvaluater(cfg)

    # build optimizer & scheduler
    optimizer = instantiate(cfg.train.optimizer, params=model.parameters())
    lr_scheduler = instantiate(cfg.train.lr_scheduler, optimizer=optimizer) if hasattr(cfg.train,
                                                                                       "lr_scheduler") else None

    # build amp loss scaler
    loss_scaler = NativeScalerWithGradNormCount() if use_amp else None

    logger = get_logger(output_dir=cfg.paths.work_dir)

    num_epochs = cfg.train.num_epochs
    ckpt = CheckPointer(monitor=cfg.eval.best_monitor, work_dir=cfg.paths.work_dir, mode=cfg.eval.is_best)

    logger.info("==============START TRAINING=============")

    for epoch in range(num_epochs):
        train_one_epoch(model=model,
                        train_loader=train_loader,
                        train_evaluater=train_evaluater,
                        val_loader=val_loader,
                        val_evaluater=val_evaluater,
                        test_loader=test_loader,
                        test_evaluater=test_evaluater,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        loss_scaler=loss_scaler,
                        ckpt=ckpt,
                        logger=logger,
                        cur_epoch=epoch,
                        cfg=cfg)

    # evaluate on best validation
    st = time.time()
    ckpt.load_checkpoint(model, optimizer, lr_scheduler, mode="best")
    metric_vals = evaluate(model, val_loader, val_evaluater, "val", cfg)
    logger.info("=========BEST VALIDATION RESULT==========")
    logger.info(f'{dict2str(metric_vals)}\t', f'eta {time.time() - st:.4f}')


if __name__ == "__main__":
    args = parse_args()
    main(args)