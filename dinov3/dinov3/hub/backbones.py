# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
from enum import Enum
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import torch

from .utils import DINOV3_BASE_URL


class Weights(Enum):
    LVD1689M = "LVD1689M"
    SAT493M = "SAT493M"


def is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in ("https", "file")


def convert_path_or_url_to_url(path: str) -> str:
    if is_url(path):
        return path
    return Path(path).expanduser().resolve().as_uri()


def _make_dinov3_vit_model_url(
    *,
    patch_size: int = 16,
    compact_arch_name: str = "vitb",
    version: Optional[str] = None,
    weights: Union[Weights, str] = Weights.LVD1689M,
    hash: Optional[str] = None,
):
    model_arch = f"{compact_arch_name}{patch_size}"
    version_suffix = f"_{version}" if version else ""
    weights_name = weights.value.lower()
    hash_suffix = f"-{hash}" if hash else ""
    model_dir = f"dinov3_{model_arch}"
    model_filename = f"dinov3_{model_arch}_pretrain_{weights_name}{version_suffix}{hash_suffix}.pth"
    return os.path.join(DINOV3_BASE_URL, model_dir, model_filename)


def _make_dinov3_vit(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    hash: Optional[str] = None,
    check_hash: bool = False,
    **kwargs,
):
    from ..models.vision_transformer import DinoVisionTransformer

    model = DinoVisionTransformer(**kwargs)
    if pretrained:
        if isinstance(weights, Weights):
            url = _make_dinov3_vit_model_url(
                patch_size=kwargs["patch_size"],
                compact_arch_name="vitb",
                weights=weights,
                hash=hash,
            )
        else:
            url = convert_path_or_url_to_url(weights)
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=check_hash)
        model.load_state_dict(state_dict, strict=True)
    else:
        model.init_weights()
    return model


def dinov3_vitb16(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if "hash" not in kwargs:
        kwargs["hash"] = "73cec8be"
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        check_hash=check_hash,
        **kwargs,
    )
