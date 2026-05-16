# Copyright (c) 2026 Yihang Wu.

import argparse
import copy
import datetime
import os
import random
import re
import shutil
import time
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
    return edict(OmegaConf.to_container(config, resolve=True))


def init_distributed(seed=42):
    """Initialize NCCL distributed training and per-rank random seeds."""
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

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


def local_backup_src_code(
    src_dir,
    dst_dir,
    max_size_MB=10.0,
    extension_to_backup=(".py", ".yaml", ".sh", ".bash", ".json"),
    exclude_dirs=("wandb", ".git", "checkpoints", "experiments", "__pycache__"),
    verbose=True,
):
    """Copy release source files into the checkpoint directory."""
    start_time = time.time()
    src_path = Path(src_dir).resolve()
    dst_path = Path(dst_dir).resolve()
    extension_set = set(extension_to_backup)
    ignore_paths = {(src_path / d).resolve() for d in exclude_dirs}
    max_bytes = int(max_size_MB * 1024 * 1024)

    if not src_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_path}")

    files = []
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(src_path):
        current_path = Path(dirpath).resolve()
        if current_path in ignore_paths or any(parent in ignore_paths for parent in current_path.parents):
            dirnames.clear()
            continue

        for filename in filenames:
            if os.path.splitext(filename)[1] not in extension_set:
                continue
            src_file = current_path / filename
            rel_path = current_path.relative_to(src_path)
            dst_file = dst_path / rel_path / filename
            try:
                file_size = src_file.stat().st_size
            except (FileNotFoundError, PermissionError) as exc:
                if verbose:
                    print(f"Warning: Could not access {src_file}: {exc}")
                continue
            total_size += file_size
            files.append((src_file, dst_file, file_size))

    if total_size > max_bytes:
        if verbose:
            print(f"Size limit exceeded: {total_size / (1024 * 1024):.2f} MB > {max_size_MB} MB")
            print("Largest files:")
            for src_file, _, size in sorted(files, key=lambda item: item[2], reverse=True)[:5]:
                print(f"{src_file}: {size / 1024:.1f} KB")
        raise ValueError(f"Size limit exceeded: {total_size / (1024 * 1024):.2f} MB > {max_size_MB} MB")

    if verbose:
        print(f"Backing up {len(files)} files ({total_size / (1024 * 1024):.2f} MB)")

    dst_path.mkdir(parents=True, exist_ok=True)
    successful_copies = 0
    for src_file, dst_file, _ in files:
        try:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            successful_copies += 1
        except Exception as exc:
            if verbose:
                print(f"Error copying {src_file} to {dst_file}: {exc}")

    if verbose:
        print(f"Backup completed: {successful_copies}/{len(files)} files copied in {time.time() - start_time:.2f}s")
    return successful_copies, total_size


def init_wandb_and_backup(config):
    assert os.path.exists(config.training.api_key_path), f"API key file does not exist: {config.training.api_key_path}"
    with open(config.training.api_key_path, "r", encoding="utf-8") as f:
        api_keys = edict(yaml.safe_load(f))
    assert api_keys.wandb is not None, "Wandb API key not found in api key file"
    os.environ["WANDB_API_KEY"] = api_keys.wandb

    wandb.init(
        project=config.training.wandb_project,
        name=config.training.wandb_exp_name,
        config=copy.deepcopy(config),
    )

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    target_dir = os.path.join(config.training.checkpoint_dir, "src", os.path.basename(cur_dir))
    os.makedirs(target_dir, exist_ok=True)
    extensions = (".py", ".yaml", ".sh", ".bash", ".json")
    local_backup_src_code(cur_dir, target_dir, extension_to_backup=extensions)

    config_save_path = os.path.join(config.training.checkpoint_dir, "config.yaml")
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(dict(config), f)

    wandb.run.log_code(target_dir, include_fn=lambda path: path.endswith(extensions))
