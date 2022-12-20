from tqdm import tqdm
from evaluater import TrainEvaluater, ValTestEvaluater, ScalarMeter, Evaluater
import torch
from kn_util.basic import add_prefix_dict, global_get, global_set
from kn_util.data import collection_to_device
from misc import dict2str
import time
from tqdm import tqdm
from torch import nn
import contextlib
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import copy


@torch.no_grad()
def evaluate(model, loader, evaluater, cfg):
    # evaluater = ValTestEvaluater(cfg, domain=domain)
    for idx, batch in enumerate(tqdm(loader, desc="evaluating")):
        batch = collection_to_device(batch, "cuda")
        out = model(**batch, mode="inference")
        out.update(batch)
        out = collection_to_device(out, "cpu")
        evaluater.update_all(out)

    metric_vals = evaluater.compute_all()

    return metric_vals


def train_one_epoch(
    model,
    train_loader,
    train_evaluater: Evaluater,
    val_loader: Evaluater,
    val_evaluater,
    ckpt,
    optimizer,
    lr_scheduler,
    loss_scaler,
    cur_epoch,
    logger,
    cfg,
    test_loader=None,
):
    use_amp = cfg.flags.amp
    use_wandb = cfg.flags.wandb

    num_batches_train = train_loader.num_batches
    val_interval = int(cfg.train.val_interval * num_batches_train)
    print_interval = int(cfg.train.print_interval * num_batches_train)
    validate_on_test = test_loader is not None

    global_step = global_get("global_step", 0)

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

        if update_grad:
            optimizer.zero_grad()

        # update training metrics
        train_evaluater.update_scalar("grad_norm", grad_norm)
        train_evaluater.update_scalar("batch_eta", time.time() - st)
        losses = collection_to_device(losses, "cpu")
        train_evaluater.update_all(losses)

        if use_validate:
            st = time.time()
            metric_vals = evaluate(model, val_loader, val_evaluater, cfg)

            # only display all losses for training
            ks = list(metric_vals.keys())
            for nm in ks:
                if nm.endswith("loss") and nm != "loss":
                    metric_vals.pop(nm)

            logger.info(f'Evaluated Val\t'
                        f'{dict2str(metric_vals)}\t'
                        f'eta {time.time() - st:.4f}s')
            if use_wandb:
                wandb.log(add_prefix_dict(metric_vals, "val/"), step=global_step)

            if validate_on_test:
                st = time.time()
                metric_vals = evaluate(model, test_loader, val_evaluater, cfg)

                # only display all losses for training
                ks = list(metric_vals.keys())
                for nm in ks:
                    if nm.endswith("loss") and nm != "loss":
                        metric_vals.pop(nm)

                logger.info(f'Evaluated Test\t'
                            f'{dict2str(metric_vals)}\t'
                            f'eta {time.time() - st:.4f}s')
                if use_wandb:
                    wandb.log(add_prefix_dict(metric_vals, "test/"), step=global_step)

            # save checkpoint
            ckpt.save_checkpoint(model=model,
                                 optimizer=optimizer,
                                 num_epochs=cur_epoch,
                                 metric_vals=metric_vals,
                                 lr_scheduler=lr_scheduler,
                                 loss_scaler=loss_scaler)

            # reduce lr on plateau
            if isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(metric_vals["loss"])

        if use_print:
            train_metrics = train_evaluater.compute_all()
            if use_wandb:
                wandb.log(add_prefix_dict(train_metrics, "train/"), step=global_step)
            mem = torch.cuda.max_memory_allocated()
            # metric_str = dict2str(train_metrics, ordered_keys=["train/loss", "train/grad_norm", "train/batch_eta"])
            other_loss = {k: v for k, v in train_metrics.items() if k.endswith("_loss")}
            logger.info(f"Train Epoch{cur_epoch:4d} [{idx+1}/{num_batches_train}]\t"
                        f"loss {train_metrics['loss']:.6f}\t"
                        f"{dict2str(other_loss)}\t"
                        f"grad_norm {train_metrics['grad_norm']:.4f}\t"
                        f"batch_eta {train_metrics['batch_eta']:.4f}s\t"
                        f"mem {mem / (1024 ** 2):.0f}MB\t")

        global_step += 1

        global_set("global_step", global_step)


def overfit_one_epoch(model, train_loader, train_evaluater: Evaluater, val_loader, val_evaluater, optimizer,
                      loss_scaler, cur_epoch, logger, cfg, overfit_sample):
    use_amp = cfg.flags.amp
    use_wandb = cfg.flags.wandb

    num_batches_train = train_loader.num_batches
    val_interval = int(cfg.train.val_interval * num_batches_train)
    print_interval = int(cfg.train.print_interval * num_batches_train)

    target_idx = [161]
    def cut_iter(cur_iter, samples=1):
        for idx, x in enumerate(cur_iter):
            if idx in target_idx:
                yield x
                break

    train_loader = cut_iter(train_loader, samples=overfit_sample)
    val_loader = cut_iter(val_loader, samples=overfit_sample)

    global_step = global_get("global_step", 0)

    for idx, batch in enumerate(train_loader):
        st = time.time()
        model.train()
        # update_grad = ((idx + 1) % cfg.train.accum_grad_steps == 0)
        update_grad = True
        use_print = True
        use_validate = ((idx + 1) == overfit_sample)

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

        if update_grad:
            optimizer.zero_grad()

        # update training metrics
        train_evaluater.update_scalar("grad_norm", grad_norm)
        train_evaluater.update_scalar("batch_eta", time.time() - st)
        losses = collection_to_device(losses, "cpu")
        train_evaluater.update_all(losses)

        if use_validate:
            metric_vals = evaluate(model, val_loader, val_evaluater, cfg)
            logger.info(f"Evaluated Train\t"
                        f"{dict2str(metric_vals)}\t")

        if use_print:
            train_metrics = train_evaluater.compute_all()
            if use_wandb:
                wandb.log(add_prefix_dict(train_metrics, "train/"), step=global_step)
            mem = torch.cuda.max_memory_allocated()
            # metric_str = dict2str(train_metrics, ordered_keys=["train/loss", "train/grad_norm", "train/batch_eta"])
            other_loss = {k: v for k, v in train_metrics.items() if k.endswith("_loss")}
            logger.info(f"Train Epoch{cur_epoch:4d} [{idx+1}/{overfit_sample}]\t"
                        f"loss {train_metrics['loss']:.6f}\t"
                        f"{dict2str(other_loss)}\t"
                        f"grad_norm {train_metrics['grad_norm']:.4f}\t"
                        f"batch_eta {train_metrics['batch_eta']:.4f}s\t"
                        f"mem {mem / (1024 ** 2):.0f}MB\t")

        global_step += 1

        global_set("global_step", global_step)
