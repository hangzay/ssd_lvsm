# Copyright (c) 2026 Yihang Wu.

import argparse
import datetime
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
import yaml
from easydict import EasyDict as edict
from omegaconf import OmegaConf


def process_overrides(overrides):
    """Normalize CLI overrides such as `a.b = value` into OmegaConf form."""
    combined = " ".join(overrides)
    fixed_string = re.sub(r"(\S+)\s*=\s*(\S+)", r"\1=\2", combined)
    return re.findall(r"[^\s=]+=\S+|\S+", fixed_string)


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    cli_overrides = OmegaConf.from_cli(process_overrides(args.overrides))
    config = OmegaConf.merge(config, cli_overrides)
    if "variant" in config.training or "variant" in config.model.transformer:
        raise ValueError(
            "Legacy variant settings are no longer supported. Use "
            "model.transformer.decouple, model.transformer.film, training.dino, "
            "and training.spatial_supervision; see README."
        )
    return edict(OmegaConf.to_container(config, resolve=True))


def init_distributed(seed=42):
    """Initialize NCCL distributed training and per-rank random seeds."""
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=3600),
        device_id=device,
    )

    process_seed = seed + global_rank
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)
    np.random.seed(process_seed)
    random.seed(process_seed)
    torch.backends.cudnn.benchmark = True

    return edict(
        {
            "local_rank": local_rank,
            "global_rank": global_rank,
            "world_size": world_size,
            "device": device,
            "is_main_process": global_rank == 0,
            "seed": process_seed,
        }
    )




def _plain_config(value):
    if isinstance(value, dict):
        return {key: _plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_config(item) for item in value]
    return value


def init_wandb(config):
    """Initialize optional W&B logging without requiring a key for offline runs."""
    mode = os.environ.get("WANDB_MODE", "offline")
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        key_path = config.training.get("api_key_path")
        if key_path and Path(key_path).is_file():
            with open(key_path, encoding="utf-8") as file:
                key = (yaml.safe_load(file) or {}).get("wandb")
            if key and not str(key).startswith("path_to_"):
                os.environ["WANDB_API_KEY"] = str(key)
    config_dict = _plain_config(config)
    wandb.init(project=config.training.wandb_project,
               name=config.training.wandb_exp_name,
               config=config_dict, mode=mode)
    Path(config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config_dict),
                   Path(config.training.checkpoint_dir) / "config.yaml")
