#!/usr/bin/env python3
"""Fast construction/forward checks for all LVSM architecture and dataset configs."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from easydict import EasyDict as edict
from omegaconf import OmegaConf


CONFIGS = (
    "configs/re10k_decoder_only.yaml",
    "configs/re10k_encoder_decoder.yaml",
    "configs/objaverse_decoder_only.yaml",
    "configs/objaverse_encoder_decoder.yaml",
)


def load_smoke_config(path: str, decouple: bool) -> edict:
    cfg = OmegaConf.load(path)
    cfg.model.image_tokenizer.image_size = 16
    cfg.model.pose_tokenizer.image_size = 16
    cfg.model.target_pose_tokenizer.image_size = 16
    cfg.model.transformer.d = 32
    cfg.model.transformer.d_head = 16
    cfg.model.transformer.decouple = decouple
    cfg.model.transformer.film = decouple
    if "n_layer" in cfg.model.transformer:
        cfg.model.transformer.n_layer = 1
    else:
        cfg.model.transformer.encoder_n_layer = 1
        cfg.model.transformer.decoder_n_layer = 1
        cfg.model.transformer.n_latent_vectors = 4
    cfg.training.dino = False
    cfg.training.spatial_supervision = False
    cfg.training.perceptual_loss_weight = 0.0
    cfg.training.lpips_loss_weight = 0.0
    cfg.training.grad_checkpoint_every = 0
    cfg.inference.if_inference = False
    return edict(OmegaConf.to_container(cfg, resolve=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward", action="store_true", help="also run one synthetic transformer block forward")
    args = parser.parse_args()

    for path in CONFIGS:
        for decouple in (False, True):
            cfg = load_smoke_config(path, decouple)
            module_name, class_name = cfg.model.class_name.rsplit(".", 1)
            model_cls = getattr(importlib.import_module(module_name), class_name)
            model = model_cls(cfg)
            params = sum(parameter.numel() for parameter in model.parameters())
            if args.forward:
                if not torch.cuda.is_available():
                    raise RuntimeError("--forward requires CUDA for xFormers attention")
                block = model.transformer_blocks[0] if hasattr(model, "transformer_blocks") else model.transformer_encoder[0]
                block = block.to(device="cuda", dtype=torch.bfloat16)
                inputs = torch.randn(1, 8, cfg.model.transformer.d, device="cuda", dtype=torch.bfloat16)
                output = block(inputs)
                assert output.shape == (1, 8, cfg.model.transformer.d)
                assert torch.isfinite(output).all()
            print(f"ok config={path} decouple={decouple} params={params:,}")


if __name__ == "__main__":
    main()
