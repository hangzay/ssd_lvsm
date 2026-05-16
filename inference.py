# Copyright (c) 2026 Yihang Wu.

import importlib
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from setup import init_config, init_distributed
from utils.metric_utils import export_results, summarize_evaluation


def apply_public_variant(config):
    variant = config.training.get("variant", None)
    is_re10k_decoder = (
        config.model.class_name == "model.decoder_only.DecoupledNVSDecoder"
        and config.training.dataset_name == "data.dataset_scene.Dataset"
    )

    if variant is not None and not is_re10k_decoder:
        raise ValueError("training.variant is only supported for re10k decoder-only.")
    if variant not in (None, "basic", "full"):
        raise ValueError("training.variant must be 'basic' or 'full'.")

    if variant == "basic":
        config.model.transformer.use_mod = False
        config.training.use_irepa = False
        config.training.use_spatial = False
    else:
        config.model.transformer.use_mod = True
        config.training.use_irepa = True
        config.training.use_spatial = True
    return config


def move_batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def configure_local_vgg16(config):
    local_vgg16_path = getattr(config.inference, "vgg16_weight_path", None)
    local_vgg16_path = os.path.abspath(local_vgg16_path or "metric_checkpoint/vgg16-397923af.pth")
    if not os.path.exists(local_vgg16_path):
        print(f"[inference] WARN: local VGG16 weights not found at {local_vgg16_path}; torchvision may download them.")
        return

    original_loader = torch.hub.load_state_dict_from_url

    def load_state_dict_from_url_local_first(url, *args, **kwargs):
        if url.endswith("vgg16-397923af.pth"):
            print(f"[inference] Load VGG16 weights from local file: {local_vgg16_path}")
            return torch.load(local_vgg16_path, map_location="cpu")
        return original_loader(url, *args, **kwargs)

    torch.hub.load_state_dict_from_url = load_state_dict_from_url_local_first


config = apply_public_variant(init_config())
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
dataset = Dataset(config)
sampler = DistributedSampler(dataset, shuffle=False)
dataloader = DataLoader(
    dataset,
    batch_size=config.training.batch_size_per_gpu,
    shuffle=False,
    num_workers=config.training.num_workers,
    prefetch_factor=config.training.prefetch_factor,
    persistent_workers=True,
    pin_memory=False,
    drop_last=False,
    sampler=sampler,
)

model_module, model_class_name = config.model.class_name.rsplit(".", 1)
Model = importlib.import_module(model_module).__dict__[model_class_name]
model = DDP(Model(config).to(ddp_info.device), device_ids=[ddp_info.local_rank])
model.module.load_ckpt(config.training.checkpoint_dir)

if ddp_info.is_main_process:
    print(f"Running inference; save results to: {config.inference_out_dir}")
dist.barrier()

sampler.set_epoch(0)
model.eval()

total_infer_time = 0.0
total_batches = 0
warmup_batches = config.inference.get("warmup_batches", 1)

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype[config.training.amp_dtype],
):
    for batch in dataloader:
        batch = move_batch_to_device(batch, ddp_info.device)
        torch.cuda.synchronize()
        t0 = time.time()
        result = model(batch)
        torch.cuda.synchronize()
        dt = time.time() - t0

        total_batches += 1
        if total_batches > warmup_batches:
            total_infer_time += dt
        if config.inference.get("render_video", False):
            result = model.module.render_video(result, **config.inference.render_video_config)
        export_results(result, config.inference_out_dir, compute_metrics=config.inference.get("compute_metrics", False))
    torch.cuda.empty_cache()

per_sample = 0.0
if ddp_info.is_main_process:
    effective_batches = max(total_batches - warmup_batches, 1)
    avg_batch = total_infer_time / effective_batches
    per_sample = avg_batch / config.training.batch_size_per_gpu
    print(
        f"[inference] warmup={warmup_batches}, effective_batches={effective_batches}, "
        f"avg_batch_latency={avg_batch:.4f}s, avg_sample_latency={per_sample:.4f}s"
    )
dist.barrier()

if ddp_info.is_main_process and config.inference.get("compute_metrics", False):
    summarize_evaluation(config.inference_out_dir, avg_sample_latency=per_sample)
    if config.inference.get("generate_website", True):
        os.system(f"python generate_html.py {config.inference_out_dir}")

dist.barrier()
dist.destroy_process_group()
