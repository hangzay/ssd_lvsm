#!/usr/bin/env python3
"""CUDA smoke tests for rebuttal camera encodings on both LVSM architectures."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from easydict import EasyDict as edict
from omegaconf import OmegaConf


CASES = (
    ("configs/re10k_decoder_only.yaml", "raymap", "none", False),
    ("configs/re10k_decoder_only.yaml", "raymap", "none", True),
    ("configs/re10k_decoder_only.yaml", "plucker", "prope", False),
    ("configs/re10k_decoder_only.yaml", "plucker", "prope", True),
    ("configs/re10k_decoder_only.yaml", "plucker", "gta", False),
    ("configs/re10k_decoder_only.yaml", "plucker", "gta", True),
    ("configs/re10k_encoder_decoder.yaml", "raymap", "none", False),
    ("configs/re10k_encoder_decoder.yaml", "raymap", "none", True),
    ("configs/objaverse_decoder_only.yaml", "raymap", "none", False),
    ("configs/objaverse_decoder_only.yaml", "raymap", "none", True),
    ("configs/objaverse_decoder_only.yaml", "plucker", "prope", False),
    ("configs/objaverse_decoder_only.yaml", "plucker", "prope", True),
    ("configs/objaverse_decoder_only.yaml", "plucker", "gta", False),
    ("configs/objaverse_decoder_only.yaml", "plucker", "gta", True),
    ("configs/objaverse_encoder_decoder.yaml", "raymap", "none", False),
    ("configs/objaverse_encoder_decoder.yaml", "raymap", "none", True),
)


def load_config(path: str, ray_encoding: str, attention_encoding: str, decouple: bool) -> edict:
    config = OmegaConf.load(path)
    for tokenizer_name in ("image_tokenizer", "pose_tokenizer", "target_pose_tokenizer"):
        config.model[tokenizer_name].image_size = 16
    config.model.transformer.d = 32
    config.model.transformer.d_head = 16
    config.model.transformer.decouple = decouple
    config.model.transformer.film = False
    if "n_layer" in config.model.transformer:
        config.model.transformer.n_layer = 1
    else:
        config.model.transformer.encoder_n_layer = 1
        config.model.transformer.decoder_n_layer = 1
        config.model.transformer.n_latent_vectors = 4
    config.model.camera_encoding.ray_encoding = ray_encoding
    config.model.camera_encoding.attention_encoding = attention_encoding
    config.training.dino = False
    config.training.spatial_supervision = False
    config.training.perceptual_loss_weight = 0.0
    config.training.lpips_loss_weight = 0.0
    config.training.grad_checkpoint_every = 0
    config.inference.if_inference = False
    return edict(OmegaConf.to_container(config, resolve=True))


def synthetic_batch(config: edict, device: torch.device) -> dict:
    batch_size = 1
    num_views = config.training.num_views
    image_size = config.model.image_tokenizer.image_size
    images = torch.rand(
        batch_size,
        num_views,
        3,
        image_size,
        image_size,
        device=device,
    )
    c2ws = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(batch_size, num_views, 1, 1)
    c2ws[:, :, 0, 3] = torch.linspace(-0.2, 0.2, num_views, device=device)
    intrinsics = torch.tensor(
        [image_size, image_size, image_size / 2, image_size / 2],
        device=device,
    ).view(1, 1, 4).repeat(batch_size, num_views, 1)
    indices = torch.zeros(batch_size, num_views, 2, dtype=torch.long, device=device)
    return {
        "image": images,
        "c2w": c2ws,
        "fxfycxcy": intrinsics,
        "index": indices,
        "scene_name": ["synthetic"],
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Camera encoding smoke tests require CUDA.")
    device = torch.device("cuda:0")

    for path, ray_encoding, attention_encoding, decouple in CASES:
        config = load_config(path, ray_encoding, attention_encoding, decouple)
        module_name, class_name = config.model.class_name.rsplit(".", 1)
        model_cls = getattr(importlib.import_module(module_name), class_name)
        model = model_cls(config).to(device)
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(synthetic_batch(config, device))
        loss = result.loss_metrics.loss
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss for {path} {ray_encoding}+{attention_encoding}: {loss}"
            )
        missing_grads = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        if missing_grads:
            raise RuntimeError(
                f"Trainable parameters without gradients for {path} "
                f"{ray_encoding}+{attention_encoding}: {missing_grads}"
            )
        print(
            f"ok config={path} decouple={decouple} ray={ray_encoding} "
            f"attention={attention_encoding} loss={loss.item():.6f} "
            f"shape={tuple(result.render.shape)}"
        )


if __name__ == "__main__":
    main()
