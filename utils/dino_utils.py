"""Small DINOv3 helpers used by ssLVSM training."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torchvision.transforms import v2


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_dino_eval_transform(resize_size: int, crop_size: int) -> v2.Compose:
    """Build the DINOv3 image transform without depending on the DINOv3 source tree."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(resize_size, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(crop_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


def load_dino_teacher(model_type: str, weight_path: str | None = None, *, repo: str = "facebookresearch/dinov3") -> torch.nn.Module:
    """Load a DINOv3 teacher from torch hub, optionally with local weights."""
    hub_name = {
        "vitb16": "dinov3_vitb16",
        "vitl16": "dinov3_vitl16",
    }[model_type]
    if weight_path:
        weight_path = os.path.expanduser(weight_path)
        if not Path(weight_path).exists():
            raise FileNotFoundError(f"DINOv3 weight file not found: {weight_path}")
    source = "local" if Path(repo).is_dir() else "github"
    model = torch.hub.load(repo, hub_name, pretrained=False, source=source, trust_repo=True)
    if weight_path:
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model
