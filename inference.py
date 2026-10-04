# Copyright (c) 2026 Yihang Wu.

import importlib
import json
import os
import random
import subprocess
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from setup import init_config, init_distributed
from utils.data_utils import decoupled_nvs_collate_fn
from utils.metric_utils import export_results, summarize_evaluation
from utils.training_utils import find_checkpoints, remap_legacy_lvsm_state_dict


def validate_switches(config):
    """Reject combinations that do not correspond to the paper or original baseline."""
    decouple = bool(config.model.transformer.get("decouple", True))
    film = bool(config.model.transformer.get("film", True))
    dino = bool(config.training.get("dino", True))
    spatial_supervision = bool(config.training.get("spatial_supervision", True))
    if not decouple and (film or dino or spatial_supervision):
        raise ValueError("film, dino, and spatial_supervision require model.transformer.decouple=true")
    return config


def move_batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def configure_local_vgg16(config):
    local_vgg16_path = getattr(config.inference, "vgg16_weight_path", None)
    local_vgg16_path = os.path.abspath(
        local_vgg16_path or "metric_checkpoint/vgg16-397923af.pth"
    )
    if not os.path.exists(local_vgg16_path):
        print(
            f"[inference] WARN: local VGG16 weights not found at {local_vgg16_path}; "
            "torchvision may download them."
        )
        return

    original_loader = torch.hub.load_state_dict_from_url

    def load_state_dict_from_url_local_first(url, *args, **kwargs):
        if url.endswith("vgg16-397923af.pth"):
            print(f"[inference] Load VGG16 weights from local file: {local_vgg16_path}")
            return torch.load(local_vgg16_path, map_location="cpu", weights_only=True)
        return original_loader(url, *args, **kwargs)

    torch.hub.load_state_dict_from_url = load_state_dict_from_url_local_first


def _evaluation_sample_is_available(dataset, index, config):
    """Check filesystem-backed Objaverse samples before committing to a subset."""
    if not hasattr(dataset, "_get_scene_images"):
        return True
    scene_paths = getattr(dataset, "all_scene_paths", None)
    if scene_paths is None or index >= len(scene_paths):
        return False
    image_paths = dataset._get_scene_images(str(scene_paths[index]).strip())
    required_views = int(config.training.num_views)
    valid_views = sum(
        os.path.exists(os.path.splitext(image_path)[0] + ".npy")
        for image_path in image_paths
    )
    return valid_views >= required_views


def select_evaluation_subset(dataset, config):
    """Select a deterministic first/random subset of valid scenes."""
    max_samples = int(config.inference.get("max_samples", 0) or 0)
    sample_mode = str(config.inference.get("sample_mode", "first"))
    sample_seed = int(config.inference.get("sample_seed", 777))
    if sample_mode not in ("first", "random"):
        raise ValueError(f"Unsupported inference.sample_mode={sample_mode!r}")

    desired_count = len(dataset) if max_samples <= 0 else min(max_samples, len(dataset))
    candidate_indices = list(range(len(dataset)))
    if sample_mode == "random":
        random.Random(sample_seed).shuffle(candidate_indices)

    indices = []
    for index in candidate_indices:
        if _evaluation_sample_is_available(dataset, index, config):
            indices.append(index)
            if len(indices) == desired_count:
                break
    if not indices:
        dataset_path = config.inference.get(
            "test_dataset_path",
            config.training.get("dataset_path", "unknown"),
        )
        dataset_root = config.inference.get("test_objaverse_root", "unknown")
        raise FileNotFoundError(
            "No valid evaluation scenes were found. "
            f"test_dataset_path={dataset_path}, test_objaverse_root={dataset_root}. "
            "Set OBJAVERSE_TEST_LIST and OBJAVERSE_TEST_ROOT to the mounted RGB/pose data."
        )
    if len(indices) < desired_count:
        print(
            f"[inference] WARN: requested {desired_count} samples but found "
            f"only {len(indices)} valid scenes."
        )
    if indices == list(range(len(dataset))):
        return dataset, indices
    return Subset(dataset, indices), indices


def write_selection_manifest(out_dir, source_dataset, indices, config):
    paths = getattr(source_dataset, "all_scene_paths", None)
    samples = []
    for index in indices:
        sample = {"dataset_index": int(index)}
        if paths is not None and index < len(paths):
            sample["scene_path"] = str(paths[index])
        samples.append(sample)
    manifest = {
        "sample_mode": str(config.inference.get("sample_mode", "first")),
        "sample_seed": int(config.inference.get("sample_seed", 777)),
        "num_samples": len(indices),
        "samples": samples,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "selection.json"), "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


def load_inference_checkpoint(model, load_path):
    checkpoint_paths = find_checkpoints(load_path)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoint found at: {load_path}")
    checkpoint_path = checkpoint_paths[-1]
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint has no 'model' state: {checkpoint_path}")

    training_only_prefixes = (
        "dino_teacher.",
        "dino_proj_head.",
        "loss_computer.",
    )
    checkpoint_state = {
        key: value
        for key, value in checkpoint["model"].items()
        if not key.startswith(training_only_prefixes)
    }
    checkpoint_state, renamed = remap_legacy_lvsm_state_dict(
        checkpoint_state,
        model.state_dict(),
    )
    status = model.load_state_dict(checkpoint_state, strict=False)
    missing = [
        key for key in status.missing_keys
        if not key.startswith(training_only_prefixes)
    ]
    unexpected = [
        key for key in status.unexpected_keys
        if not key.startswith(training_only_prefixes)
    ]
    if missing or unexpected:
        raise RuntimeError(
            "Inference checkpoint is not structurally compatible: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return checkpoint_path, len(renamed)


def build_dataloader(dataset, sampler, config):
    num_workers = int(config.training.num_workers)
    loader_kwargs = {
        "batch_size": int(config.training.batch_size_per_gpu),
        "shuffle": False,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "pin_memory": False,
        "drop_last": False,
        "sampler": sampler,
        "collate_fn": decoupled_nvs_collate_fn,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(config.training.prefetch_factor)
    return DataLoader(dataset, **loader_kwargs)


config = validate_switches(init_config())
configure_local_vgg16(config)
os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

ddp_info = init_distributed(seed=777)
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
source_dataset = Dataset(config)
dataset, selected_indices = select_evaluation_subset(source_dataset, config)
sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
dataloader = build_dataloader(dataset, sampler, config)

model_module, model_class_name = config.model.class_name.rsplit(".", 1)
Model = importlib.import_module(model_module).__dict__[model_class_name]
base_model = Model(config).to(ddp_info.device)
checkpoint_path, remapped_keys = load_inference_checkpoint(
    base_model,
    config.training.checkpoint_dir,
)
model = DDP(base_model, device_ids=[ddp_info.local_rank])

if ddp_info.is_main_process:
    write_selection_manifest(
        config.inference_out_dir,
        source_dataset,
        selected_indices,
        config,
    )
    print(
        f"[inference] checkpoint={os.path.abspath(checkpoint_path)} | "
        f"legacy_keys_remapped={remapped_keys} | samples={len(selected_indices)} | "
        f"save={os.path.abspath(config.inference_out_dir)}"
    )
dist.barrier()

sampler.set_epoch(0)
model.eval()
total_infer_time = 0.0
timed_samples = 0
warmup_batches = int(config.inference.get("warmup_batches", 1))

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype[config.training.amp_dtype],
):
    for batch_idx, batch in enumerate(dataloader):
        batch_size = int(batch["image"].shape[0])
        batch = move_batch_to_device(batch, ddp_info.device)
        torch.cuda.synchronize()
        start = time.time()
        result = model(batch)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        if batch_idx >= warmup_batches:
            total_infer_time += elapsed
            timed_samples += batch_size
        if config.inference.get("render_video", False):
            result = model.module.render_video(
                result,
                **config.inference.render_video_config,
            )
        export_results(
            result,
            config.inference_out_dir,
            compute_metrics=config.inference.get("compute_metrics", False),
        )

timing = torch.tensor(
    [total_infer_time, float(timed_samples)],
    device=ddp_info.device,
    dtype=torch.float64,
)
dist.all_reduce(timing, op=dist.ReduceOp.SUM)
avg_sample_latency = timing[0].item() / max(timing[1].item(), 1.0)
torch.cuda.empty_cache()
dist.barrier()

if ddp_info.is_main_process:
    print(
        f"[inference] warmup_batches_per_rank={warmup_batches}, "
        f"timed_samples={int(timing[1].item())}, "
        f"avg_sample_latency={avg_sample_latency:.4f}s"
    )
    if config.inference.get("compute_metrics", False):
        summarize_evaluation(
            config.inference_out_dir,
            avg_sample_latency=avg_sample_latency,
        )
        if config.inference.get("generate_website", True):
            subprocess.run(
                [sys.executable, "generate_html.py", config.inference_out_dir],
                check=True,
            )

dist.barrier()
dist.destroy_process_group()
