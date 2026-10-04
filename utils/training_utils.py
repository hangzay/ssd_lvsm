# Copyright (c) 2026 Yihang Wu.

import re

import torch
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
import torch.distributed as dist
import os
from rich import print
import traceback
from torch.nn.parallel import DistributedDataParallel as DDP


def print_rank0(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def format_number(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return str(num)

def create_optimizer(model, weight_decay, learning_rate, betas):
    all_param_dict = {name: param for name, param in model.named_parameters()}
    optimized_param_dict = {name: param for name, param in all_param_dict.items() if param.requires_grad}

    decay_params, nodecay_params = [], []
    for name, param in optimized_param_dict.items():
        if param.dim() == 1 or getattr(param, "_no_weight_decay", False):
            nodecay_params.append(param)
        else:
            decay_params.append(param)
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=True)

    if dist.get_rank() == 0:
        def get_module_name(name):
            parts = name.split(".")
            if len(parts) > 2 and parts[0] == "module":
                return parts[1] + "." + parts[2]
            return parts[0]

        print(f"Optimizer: AdamW, learning rate: {learning_rate}, weight decay: {weight_decay}, betas: {betas}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in optimized_param_dict.values())
        optim_module_names = sorted(set(get_module_name(name) for name in optimized_param_dict.keys()))
        frozen_module_names = sorted(set(get_module_name(name) for name in set(all_param_dict.keys()) - set(optimized_param_dict.keys())))

        print(f"Total parameters: {format_number(total_params)}, Trainable parameters: {format_number(trainable_params)}")
        print(f"Optimized parameters: {optim_module_names}")
        print(f"Frozen parameters: {frozen_module_names}")

    return optimizer, optimized_param_dict, all_param_dict

def create_lr_scheduler(optimizer, param_update_steps, warm_up_steps, scheduler_type='cosine'):
    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    elif scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    elif scheduler_type == 'constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, warm_up_steps)
    else:
        raise ValueError(f'Invalid scheduler type: {scheduler_type}')
    return scheduler



def find_checkpoints(load_path):
    if os.path.isdir(load_path):
        ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
        ckpt_names = sorted(ckpt_names, key=lambda x: x)
        ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
    else:
        if load_path.endswith(".pt"):
            ckpt_paths = [load_path]
        else:
            ckpt_paths = []
    return ckpt_paths


def remap_legacy_lvsm_state_dict(model_state, target_state):
    """Map legacy baseline-only parameter names to the consolidated model."""
    remapped_state = dict(model_state)
    renamed = {}
    fixed_aliases = {
        "image_token_decoder.0.weight": "decoder_norm.weight",
        "image_token_decoder.1.weight": "decoder_linear.weight",
    }

    for source_key in list(remapped_state):
        target_key = fixed_aliases.get(source_key)
        if target_key is None:
            match = re.fullmatch(r"transformer_blocks\.(\d+)\.norm([12])\.weight", source_key)
            if match is not None:
                target_key = (
                    f"transformer_blocks.{match.group(1)}."
                    f"norm_{match.group(2)}.weight"
                )
        if target_key is None or target_key not in target_state:
            continue
        if target_key in remapped_state:
            raise RuntimeError(
                f"Checkpoint contains both legacy '{source_key}' and current "
                f"'{target_key}' keys."
            )
        if remapped_state[source_key].shape != target_state[target_key].shape:
            raise RuntimeError(
                f"Cannot remap '{source_key}' to '{target_key}': checkpoint "
                f"shape={tuple(remapped_state[source_key].shape)}, target "
                f"shape={tuple(target_state[target_key].shape)}."
            )
        remapped_state[target_key] = remapped_state.pop(source_key)
        renamed[source_key] = target_key

    return remapped_state, renamed


def auto_resume_job(
    load_path,
    model,
    optimizer,
    lr_scheduler,
    reset_training_state
):
    """
    Resume training from the latest checkpoint in the specified directory.
    Returns the fwdbwd_pass_step and param_update_step.

    Args:
        load_path: If dir, load the last checkpoint in the directory.
            O.w., assume it's a ckpt and load it.
        model: model to be loaded
        optimizer: optimizer to be loaded
        lr_scheduler: lr scheduler to be loaded
        reset_training_state: whether to reset the training state

    Returns:
        optimizer, lr_scheduler, forward_pass_step, param_update_step

    """
    forward_pass_step = 0
    param_update_step = 0
    all_ckpt_paths = find_checkpoints(load_path)
    if len(all_ckpt_paths) == 0:
        print_rank0(f"No checkpoint found in {load_path}, we will start from scratch")
        return optimizer, lr_scheduler, forward_pass_step, param_update_step
    try:
        ckpt_path = all_ckpt_paths[-1]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except:
        traceback.print_exc()
        print_rank0(f"Failed to load {ckpt_path}, we will start from scratch")
        return optimizer, lr_scheduler, forward_pass_step, param_update_step

    # Teacher/loss helpers are external frozen state, not trainable checkpoint state.
    ignored_prefixes = ("dino_teacher.", "loss_computer.")
    model_state = {
        key: value
        for key, value in checkpoint["model"].items()
        if not key.startswith(ignored_prefixes)
    }
    target_model = model.module if isinstance(model, DDP) else model
    model_state, renamed = remap_legacy_lvsm_state_dict(
        model_state,
        target_model.state_dict(),
    )
    status = target_model.load_state_dict(model_state, strict=False)
    core_missing = [key for key in status.missing_keys if not key.startswith(ignored_prefixes)]
    core_unexpected = [key for key in status.unexpected_keys if not key.startswith(ignored_prefixes)]
    print_rank0(
        f"Loaded model from {os.path.abspath(ckpt_path)} | "
        f"legacy_keys_remapped={len(renamed)} | "
        f"core_missing={core_missing} | core_unexpected={core_unexpected}"
    )
    if renamed and (core_missing or core_unexpected):
        raise RuntimeError(
            "Legacy checkpoint remapping left unresolved core model keys: "
            f"missing={core_missing}, unexpected={core_unexpected}"
        )

    # resume training state
    if not reset_training_state:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            forward_pass_step = checkpoint["fwdbwd_pass_step"]
            param_update_step = checkpoint["param_update_step"]
            print_rank0(f"Resumed optimizer and lr_scheduler from {ckpt_path}")
        except:
            traceback.print_exc()
            print_rank0(f"Failed to load optimizer and lr_scheduler from {ckpt_path}")

    return optimizer, lr_scheduler, forward_pass_step, param_update_step
