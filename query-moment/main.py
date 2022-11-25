import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar
from engine import MomentRetrievalModule
from kn_util.general import registry, get_logger
from kn_util.config import LazyConfig

log = get_logger(__name__)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("cfg", type=str)
    args.add_argument("--cfg-override", nargs="+")
    args.add_argument("--resume", action="store_true", default=False)
    args.add_argument("--accelerator", default="gpu", type=str)
    args.add_argument("--debug", action="store_true", default=False)

    return args.parse_args()


def main(args):
    cfg = LazyConfig.load(args.cfg)
    LazyConfig.apply_overrides(cfg, args.cfg_option)
    if args.resume:
        pl.Trainer.resume_from_checkpoint(cfg.paths.work_dir)
    callbacks = [
        ModelCheckpoint(
            cfg.paths.work_dir,
            monitor=cfg.train.val_monitor,
            verbose=True,
            save_weights_only=True,
        ),
        RichModelSummary(),
        RichProgressBar(refresh_rate=20),
    ]
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        callbacks=callbacks,
        gradient_clip_val=cfg.train.clip_grad,
        accelerator=args.accelerator,
    )

    module = MomentRetrievalModule(cfg)
    datamodule = registry.build_datamodule(cfg.data.dataset, cfg=cfg)

    trainer.fit(model=module, datamodule=datamodule)
    trainer.validate(ckpt_path="best")



if __name__ == "__main__":
    args = parse_args()
    main(args)