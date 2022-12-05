import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger
from engine import MomentRetrievalModule
from kn_util.general import registry, get_logger
from kn_util.config import LazyConfig
from pprint import pformat

log = get_logger(__name__)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("cfg", type=str)
    args.add_argument("--cfg-override", "-co", nargs="+")
    args.add_argument("--resume", action="store_true", default=False)
    args.add_argument("--accelerator", default="gpu", type=str)
    args.add_argument("--no-multiproc", action="store_true", default=False)
    args.add_argument("--no-callback", action="store_true", default=False)

    return args.parse_args()


def main(args):
    cfg = LazyConfig.load(args.cfg)
    LazyConfig.apply_overrides(cfg, args.cfg_override)

    print(pformat(cfg))

    if args.resume:
        pl.Trainer.resume_from_checkpoint(cfg.paths.work_dir)

    if args.no_multiproc:
        # args.accelerator = "cpu"
        cfg.train.prefetch_factor = 2
        cfg.train.num_workers = 0

    if args.no_callback:
        callbacks = []
        logger = False

    else:
        callbacks = [
            ModelCheckpoint(
                cfg.paths.work_dir,
                monitor=cfg.train.val_monitor,
                verbose=True,
                save_weights_only=True,
            ),
            RichModelSummary(),
            RichProgressBar(refresh_rate=20)
        ]
        logger = WandbLogger(name=cfg.flags.exp, project="query-moment")

    trainer = pl.Trainer(max_epochs=cfg.train.max_epochs,
                         callbacks=callbacks,
                         gradient_clip_val=cfg.train.clip_grad,
                         accelerator=args.accelerator,
                         logger=logger)
    from kn_util.general import registry
    # registry.register_object("output_proposal", True)
    # registry.register_object("output_grad", True)

    # trainer = pl.Trainer(max_epochs=-1,
    #                      callbacks=callbacks,
    #                      gradient_clip_val=cfg.train.clip_grad,
    #                      accelerator=args.accelerator,
    #                      overfit_batches=1)

    module = MomentRetrievalModule(cfg)
    datamodule = registry.build_datamodule(cfg.data.dataset, cfg=cfg)
    # datamodule.datasets["train"] = datamodule.datasets["train"][:1]

    # trainer.fit(model=module,
    #             train_dataloaders=datamodule.train_dataloader(),
    #             val_dataloaders=datamodule.train_dataloader())
    trainer.fit(model=module, datamodule=datamodule)
    # trainer.validate(ckpt_path="best")


if __name__ == "__main__":
    args = parse_args()
    main(args)