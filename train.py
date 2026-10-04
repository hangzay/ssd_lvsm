# Copyright (c) 2026 Yihang Wu.

import copy
import importlib
import os
import time
import traceback

import torch
import torch.distributed as dist
import wandb
from rich import print
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from setup import init_config, init_distributed, init_wandb
from utils.data_utils import decoupled_nvs_collate_fn
from utils.metric_utils import visualize_intermediate_results
from utils.training_utils import auto_resume_job, create_lr_scheduler, create_optimizer, print_rank0


def validate_switches(config):
    """Reject combinations that do not correspond to the paper or original baseline."""
    decouple = bool(config.model.transformer.get("decouple", True))
    film = bool(config.model.transformer.get("film", True))
    dino = bool(config.training.get("dino", True))
    spatial_supervision = bool(config.training.get("spatial_supervision", True))
    if not decouple and (film or dino or spatial_supervision):
        raise ValueError("film, dino, and spatial_supervision require model.transformer.decouple=true")

    image_sizes = {
        int(config.model.image_tokenizer.image_size),
        int(config.model.pose_tokenizer.image_size),
        int(config.model.target_pose_tokenizer.image_size),
    }
    patch_sizes = {
        int(config.model.image_tokenizer.patch_size),
        int(config.model.pose_tokenizer.patch_size),
        int(config.model.target_pose_tokenizer.patch_size),
    }
    if len(image_sizes) != 1 or len(patch_sizes) != 1:
        raise ValueError("Image/pose/target tokenizers must use the same image_size and patch_size.")
    image_size = next(iter(image_sizes))
    patch_size = next(iter(patch_sizes))
    if image_size % patch_size != 0:
        raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}.")

    camera_cfg = config.model.get("camera_encoding", {})
    ray_encoding = camera_cfg.get("ray_encoding", "plucker")
    attention_encoding = camera_cfg.get("attention_encoding", "none")
    if ray_encoding not in ("plucker", "raymap"):
        raise ValueError(f"Unsupported ray encoding: {ray_encoding}")
    if attention_encoding not in ("none", "prope", "gta"):
        raise ValueError(f"Unsupported attention encoding: {attention_encoding}")
    if attention_encoding != "none" and "encoder_decoder" in config.model.class_name:
        raise ValueError(
            "PRoPE/GTA is not defined for encoder-decoder LVSM scene latents."
        )
    return config


def move_batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }




def build_dataloader(config, dataset_cls, batch_size, shuffle, drop_last):
    dataset = dataset_cls(config)
    sampler = DistributedSampler(dataset, shuffle=shuffle)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": config.training.num_workers,
        "persistent_workers": config.training.num_workers > 0,
        "pin_memory": False,
        "drop_last": drop_last,
        "sampler": sampler,
        "collate_fn": decoupled_nvs_collate_fn,
    }
    if config.training.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.training.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)
    return dataset, sampler, loader


def run_validation(model, val_loader, val_sampler, ddp_info, config, amp_dtype, step_idx):
    if val_loader is None:
        return
    val_sampler.set_epoch(step_idx)
    model.eval()

    totals = {
        "loss": torch.tensor(0.0, device=ddp_info.device),
        "psnr": torch.tensor(0.0, device=ddp_info.device),
        "batches": torch.tensor(0.0, device=ddp_info.device),
    }
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            batch = move_batch_to_device(batch, ddp_info.device)
            with torch.autocast(
                enabled=config.training.use_amp,
                device_type="cuda",
                dtype=amp_dtype[config.training.amp_dtype],
            ):
                ret = model(batch)
            totals["loss"] += ret.loss_metrics.loss.detach()
            totals["psnr"] += ret.loss_metrics.psnr.detach()
            totals["batches"] += 1.0
            max_batches = config.training.get("eval_max_batches", None)
            if max_batches is not None and idx + 1 >= max_batches:
                break

    for value in totals.values():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)

    if totals["batches"].item() > 0 and ddp_info.is_main_process:
        avg_loss = (totals["loss"] / totals["batches"]).item()
        avg_psnr = (totals["psnr"] / totals["batches"]).item()
        print(f"[Eval] step {step_idx:>6d} | avg_loss: {avg_loss:.6f} | avg_psnr: {avg_psnr:.6f}")
        wandb.log(
            {"eval/loss": avg_loss, "eval/psnr": avg_psnr, "eval/iter": step_idx},
            step=step_idx,
        )
    model.train()


config = validate_switches(init_config())
time_detail = bool(config.training.get("time_detail", False))
os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

ddp_info = init_distributed(seed=777)
dist.barrier()

if ddp_info.is_main_process:
    init_wandb(config)
dist.barrier()

torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "tf32": torch.float32,
}

dataset_module, dataset_class_name = config.training.get("dataset_name").rsplit(".", 1)
Dataset = importlib.import_module(dataset_module).__dict__[dataset_class_name]
dataset, sampler, dataloader = build_dataloader(
    config,
    Dataset,
    config.training.batch_size_per_gpu,
    shuffle=True,
    drop_last=True,
)
dataloader_iter = iter(dataloader)

val_loader = None
val_sampler = None
if config.training.get("eval_every", 0) > 0 and config.training.get("eval_list_path", None):
    val_config = copy.deepcopy(config)
    val_config.training.dataset_path = config.training.eval_list_path
    _, val_sampler, val_loader = build_dataloader(
        val_config,
        Dataset,
        config.training.get("eval_batch_size", config.training.batch_size_per_gpu),
        shuffle=False,
        drop_last=False,
    )

total_param_update_steps = config.training.train_steps
grad_accum_steps = config.training.grad_accum_steps
total_train_steps = total_param_update_steps * grad_accum_steps
total_batch_size = config.training.batch_size_per_gpu * ddp_info.world_size * grad_accum_steps
total_num_epochs = int(total_param_update_steps * total_batch_size / len(dataset))

model_module, model_class_name = config.model.class_name.rsplit(".", 1)
Model = importlib.import_module(model_module).__dict__[model_class_name]
model = DDP(
    Model(config).to(ddp_info.device),
    device_ids=[ddp_info.local_rank],
    find_unused_parameters=False,
)

optimizer, optimized_param_dict, _ = create_optimizer(
    model,
    config.training.weight_decay,
    config.training.lr,
    (config.training.beta1, config.training.beta2),
)
optim_param_list = list(optimized_param_dict.values())
lr_scheduler = create_lr_scheduler(
    optimizer,
    total_param_update_steps,
    config.training.warmup,
    scheduler_type=config.training.get("scheduler_type", "cosine"),
)

ckpt_load_path = config.training.get("resume_ckpt", "") or config.training.checkpoint_dir
optimizer, lr_scheduler, cur_train_step, cur_param_update_step = auto_resume_job(
    ckpt_load_path,
    model,
    optimizer,
    lr_scheduler,
    config.training.get("reset_training_state", False),
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=config.training.use_amp and config.training.amp_dtype == "fp16",
)
print_rank0(f"Grad scaler enabled: {scaler.is_enabled()}")
dist.barrier()

start_train_step = cur_train_step
model.train()

while cur_train_step < total_train_steps:
    tic = time.time()
    cur_epoch = int(cur_train_step * (total_batch_size / grad_accum_steps) // len(dataset))
    t_iter_start = time.perf_counter() if time_detail else None
    data_next_s = to_device_s = forward_s = backward_s = optim_s = 0.0

    try:
        t0 = time.perf_counter() if time_detail else None
        data = next(dataloader_iter)
        if time_detail:
            data_next_s = time.perf_counter() - t0
    except StopIteration:
        print(f"Rank {ddp_info.local_rank} reached dataloader end. Resetting epoch to {cur_epoch}.")
        sampler.set_epoch(cur_epoch)
        dataloader_iter = iter(dataloader)
        data = next(dataloader_iter)

    t0 = time.perf_counter() if time_detail else None
    batch = move_batch_to_device(data, ddp_info.device)
    if time_detail:
        to_device_s = time.perf_counter() - t0

    t0 = time.perf_counter() if time_detail else None
    with torch.autocast(
        enabled=config.training.use_amp,
        device_type="cuda",
        dtype=amp_dtype[config.training.amp_dtype],
    ):
        ret = model(batch)
    if time_detail:
        forward_s = time.perf_counter() - t0

    update_grads = (
        (cur_train_step + 1) % grad_accum_steps == 0
        or (cur_train_step + 1) == total_train_steps
    )
    t0 = time.perf_counter() if time_detail else None
    if update_grads:
        scaler.scale(ret.loss_metrics.loss / grad_accum_steps).backward()
    else:
        with model.no_sync():
            scaler.scale(ret.loss_metrics.loss / grad_accum_steps).backward()
    if time_detail:
        backward_s = time.perf_counter() - t0
    cur_train_step += 1

    vis_every = int(config.training.get("vis_every", 0))
    export_inter_results = vis_every > 0 and (
        (cur_train_step - 1) == start_train_step or cur_train_step % vis_every == 0
    )

    total_grad_norm = None
    if update_grads:
        t0 = time.perf_counter() if time_detail else None
        skip_optimizer_step = False
        if torch.isnan(ret.loss_metrics.loss) or torch.isinf(ret.loss_metrics.loss):
            print("NaN or Inf loss detected; skipping optimizer step.")
            skip_optimizer_step = True
            ret.loss_metrics.loss.data = torch.zeros_like(ret.loss_metrics.loss)

        if not skip_optimizer_step:
            scaler.unscale_(optimizer)
            with torch.no_grad():
                for param in optimized_param_dict.values():
                    if param.requires_grad and param.grad is not None:
                        param.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)

            total_grad_norm = 0.0
            if config.training.grad_clip_norm > 0:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    optim_param_list,
                    max_norm=config.training.grad_clip_norm,
                ).item()
                allowed = config.training.grad_clip_norm * config.training.get(
                    "allowed_gradnorm_factor",
                    5,
                )
                if total_grad_norm > allowed:
                    skip_optimizer_step = True
                    print(
                        f"WARNING: step {cur_train_step} grad norm "
                        f"{total_grad_norm} > {allowed}; skipping."
                    )

            if not skip_optimizer_step:
                scaler.step(optimizer)
                cur_param_update_step += 1

        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if time_detail:
            optim_s = time.perf_counter() - t0

    if time_detail:
        iter_s = time.perf_counter() - t_iter_start

    if ddp_info.is_main_process:
        loss_dict = {key: float(f"{value.item():.6f}") for key, value in ret.loss_metrics.items()}
        if (
            cur_train_step % config.training.print_every == 0
            or cur_train_step < 100 + start_train_step
        ):
            msg = (
                f"[Epoch {int(cur_epoch):>3d}/{total_num_epochs:>3d}] | "
                f"Forward step: {int(cur_train_step):>6d} | "
                f"Param update step: {int(cur_param_update_step):>6d} | "
                f"Iter Time: {time.time() - tic:.2f}s | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}\n"
            )
            msg += " | ".join(f"{key}: {value:.6f}" for key, value in loss_dict.items())
            if time_detail:
                msg += (
                    f"\n[timing] data={data_next_s:.4f} to_device={to_device_s:.4f} "
                    f"forward={forward_s:.4f} backward={backward_s:.4f} "
                    f"optim={optim_s:.4f} iter={iter_s:.4f}"
                )
            print(msg)

        if (
            cur_train_step % config.training.wandb_log_every == 0
            or cur_train_step < 200 + start_train_step
        ):
            log_dict = {
                "iter": cur_train_step,
                "forward_pass_step": cur_train_step,
                "param_update_step": cur_param_update_step,
                "lr": optimizer.param_groups[0]["lr"],
                "iter_time": time.time() - tic,
                "grad_norm": total_grad_norm,
                "epoch": cur_epoch,
            }
            log_dict.update({"train/" + key: value for key, value in loss_dict.items()})
            wandb.log(log_dict, step=cur_train_step)


        checkpoint_every = config.training.get("checkpoint_every", 0)
        if (
            checkpoint_every > 0
            and cur_train_step % checkpoint_every == 0
            or cur_train_step == total_train_steps
        ):
            checkpoint = {
                "model": {
                    key: value
                    for key, value in model.module.state_dict().items()
                    if not key.startswith(("dino_teacher.", "loss_computer."))
                },
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "fwdbwd_pass_step": cur_train_step,
                "param_update_step": cur_param_update_step,
            }
            os.makedirs(config.training.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(
                config.training.checkpoint_dir,
                f"ckpt_{cur_train_step:016}.pt",
            )
            torch.save(checkpoint, ckpt_path)
            print(
                f"Saved checkpoint at step {cur_train_step} "
                f"to {os.path.abspath(ckpt_path)}"
            )

        if export_inter_results:
            vis_path = os.path.join(
                config.training.checkpoint_dir,
                f"iter_{cur_train_step:08d}",
            )
            try:
                visualize_intermediate_results(vis_path, ret)
                print(f"Saved training visualization to {os.path.abspath(vis_path)}")
            except Exception as exc:
                print(f"[Visualization] Failed to save {vis_path}: {exc}")
                traceback.print_exc()

    if export_inter_results:
        torch.cuda.empty_cache()
        dist.barrier()

    eval_every = config.training.get("eval_every", 0)
    if eval_every > 0 and cur_train_step % eval_every == 0:
        run_validation(
            model,
            val_loader,
            val_sampler,
            ddp_info,
            config,
            amp_dtype,
            cur_train_step,
        )

dist.barrier()
dist.destroy_process_group()
