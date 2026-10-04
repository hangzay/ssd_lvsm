# Copyright (c) 2026 Yihang Wu.
#
# PRoPE/GTA attention transforms are adapted from the official PyTorch source:
# https://github.com/liruilong940607/prope/blob/main/prope/torch.py
# Copyright (c) Authors of "Cameras as Relative Positional Encoding".
# Upstream MIT notice: LICENSES/PRoPE-MIT.txt.
# Ray tokenization and LVSM camera-context integration are local extensions.

"""Plucker/raymap tokens and PRoPE relative camera attention."""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


RAY_ENCODINGS = ("plucker", "raymap")
ATTENTION_ENCODINGS = ("none", "prope", "gta")


def encode_rays(ray_o: torch.Tensor, ray_d: torch.Tensor, encoding: str) -> torch.Tensor:
    """Encode a world-space ray as a six-channel per-pixel camera map."""
    if encoding == "plucker":
        return torch.cat([torch.cross(ray_o, ray_d, dim=2), ray_d], dim=2)
    if encoding == "raymap":
        return torch.cat([ray_o, ray_d], dim=2)
    raise ValueError(f"Unsupported ray encoding '{encoding}'. Expected one of {RAY_ENCODINGS}.")


def intrinsics_from_fxfycxcy(fxfycxcy: torch.Tensor) -> torch.Tensor:
    """Convert [..., fx, fy, cx, cy] vectors to homogeneous 3x3 intrinsics."""
    intrinsics = torch.zeros(
        *fxfycxcy.shape[:-1],
        3,
        3,
        device=fxfycxcy.device,
        dtype=fxfycxcy.dtype,
    )
    intrinsics[..., 0, 0] = fxfycxcy[..., 0]
    intrinsics[..., 1, 1] = fxfycxcy[..., 1]
    intrinsics[..., 0, 2] = fxfycxcy[..., 2]
    intrinsics[..., 1, 2] = fxfycxcy[..., 3]
    intrinsics[..., 2, 2] = 1.0
    return intrinsics


def build_decoder_only_camera_context(
    input_data,
    target_data,
    *,
    num_target_views: int,
    patches_x: int,
    patches_y: int,
    image_width: int,
    image_height: int,
    attention_encoding: str,
) -> dict | None:
    """Align input/target cameras with decoder-only's repeated target batches."""
    if attention_encoding == "none":
        return None
    if attention_encoding not in ATTENTION_ENCODINGS:
        raise ValueError(
            f"Unsupported attention encoding '{attention_encoding}'. "
            f"Expected one of {ATTENTION_ENCODINGS}."
        )

    ref_c2ws = repeat(
        input_data.c2w,
        "b vi x y -> (b vt) vi x y",
        vt=num_target_views,
    )
    target_c2ws = rearrange(target_data.c2w, "b vt x y -> (b vt) 1 x y")
    c2ws = torch.cat([ref_c2ws, target_c2ws], dim=1)

    ref_intrinsics = intrinsics_from_fxfycxcy(input_data.fxfycxcy)
    target_intrinsics = intrinsics_from_fxfycxcy(target_data.fxfycxcy)
    ref_intrinsics = repeat(
        ref_intrinsics,
        "b vi x y -> (b vt) vi x y",
        vt=num_target_views,
    )
    target_intrinsics = rearrange(
        target_intrinsics,
        "b vt x y -> (b vt) 1 x y",
    )

    context = {
        "encoding": attention_encoding,
        "viewmats": torch.linalg.inv(c2ws),
        "intrinsics": torch.cat([ref_intrinsics, target_intrinsics], dim=1),
        "patches_x": patches_x,
        "patches_y": patches_y,
        "image_width": image_width,
        "image_height": image_height,
    }
    return context




class ProjectivePositionEncoding(torch.nn.Module):
    """Faithful PRoPE/GTA transform around scaled dot-product attention."""

    def __init__(
        self,
        head_dim: int,
        patches_x: int,
        patches_y: int,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
    ):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError("PRoPE requires attention head_dim divisible by 4.")
        self.head_dim = head_dim
        self.patches_x = patches_x
        self.patches_y = patches_y

        coeffs_x = _rope_precompute_coeffs(
            torch.tile(torch.arange(patches_x), (patches_y,)),
            freq_base=freq_base,
            freq_scale=freq_scale,
            feat_dim=head_dim // 4,
        )
        coeffs_y = _rope_precompute_coeffs(
            torch.repeat_interleave(torch.arange(patches_y), patches_x),
            freq_base=freq_base,
            freq_scale=freq_scale,
            feat_dim=head_dim // 4,
        )
        self.register_buffer("coeffs_x_cos", coeffs_x[0], persistent=False)
        self.register_buffer("coeffs_x_sin", coeffs_x[1], persistent=False)
        self.register_buffer("coeffs_y_cos", coeffs_y[0], persistent=False)
        self.register_buffer("coeffs_y_sin", coeffs_y[1], persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        camera_context: dict,
        *,
        dropout_p: float = 0.0,
        token_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoding = camera_context["encoding"]
        if encoding not in ("prope", "gta"):
            raise ValueError(f"Projective attention received encoding='{encoding}'.")
        if camera_context["patches_x"] != self.patches_x or camera_context["patches_y"] != self.patches_y:
            raise ValueError(
                "Camera context patch grid does not match the initialized PRoPE grid: "
                f"context=({camera_context['patches_y']}, {camera_context['patches_x']}), "
                f"module=({self.patches_y}, {self.patches_x})."
            )

        viewmats = camera_context["viewmats"]
        intrinsics = camera_context["intrinsics"] if encoding == "prope" else None
        batch, _, sequence_length, head_dim = q.shape
        cameras = viewmats.shape[1]
        expected_length = cameras * self.patches_x * self.patches_y
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(f"PRoPE requires matching Q/K/V shapes, got {q.shape}, {k.shape}, {v.shape}.")
        if head_dim != self.head_dim or sequence_length != expected_length:
            raise ValueError(
                f"PRoPE token layout mismatch: q={q.shape}, cameras={cameras}, "
                f"patches=({self.patches_y}, {self.patches_x}), expected_length={expected_length}."
            )
        if viewmats.shape != (batch, cameras, 4, 4):
            raise ValueError(f"Invalid viewmat shape {viewmats.shape}.")

        apply_q, apply_kv, apply_out = _prepare_apply_fns(
            head_dim=head_dim,
            viewmats=viewmats,
            intrinsics=intrinsics,
            image_width=camera_context["image_width"],
            image_height=camera_context["image_height"],
            coeffs_x=(self.coeffs_x_cos, self.coeffs_x_sin),
            coeffs_y=(self.coeffs_y_cos, self.coeffs_y_sin),
        )
        output = F.scaled_dot_product_attention(
            apply_q(q),
            apply_kv(k),
            apply_kv(v),
            dropout_p=dropout_p,
        )
        return apply_out(output)




def _prepare_apply_fns(
    *,
    head_dim: int,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor | None,
    image_width: int,
    image_height: int,
    coeffs_x: tuple[torch.Tensor, torch.Tensor],
    coeffs_y: tuple[torch.Tensor, torch.Tensor],
) -> tuple[Callable, Callable, Callable]:
    batch, cameras = viewmats.shape[:2]
    if intrinsics is not None:
        if intrinsics.shape != (batch, cameras, 3, 3):
            raise ValueError(f"Invalid intrinsics shape {intrinsics.shape}.")
        intrinsics_norm = torch.zeros_like(intrinsics)
        intrinsics_norm[..., 0, 0] = intrinsics[..., 0, 0] / image_width
        intrinsics_norm[..., 1, 1] = intrinsics[..., 1, 1] / image_height
        intrinsics_norm[..., 0, 2] = intrinsics[..., 0, 2] / image_width - 0.5
        intrinsics_norm[..., 1, 2] = intrinsics[..., 1, 2] / image_height - 0.5
        intrinsics_norm[..., 2, 2] = 1.0
        projection = torch.einsum(
            "...ij,...jk->...ik",
            _lift_intrinsics(intrinsics_norm),
            viewmats,
        )
        projection_inv = torch.einsum(
            "...ij,...jk->...ik",
            _invert_se3(viewmats),
            _lift_intrinsics(_invert_intrinsics(intrinsics_norm)),
        )
    else:
        projection = viewmats
        projection_inv = _invert_se3(viewmats)

    projection_t = projection.transpose(-1, -2)
    transforms_q = (
        (partial(_apply_tiled_matrix, matrix=projection_t), head_dim // 2),
        (partial(_apply_rope, coeffs=coeffs_x), head_dim // 4),
        (partial(_apply_rope, coeffs=coeffs_y), head_dim // 4),
    )
    transforms_kv = (
        (partial(_apply_tiled_matrix, matrix=projection_inv), head_dim // 2),
        (partial(_apply_rope, coeffs=coeffs_x), head_dim // 4),
        (partial(_apply_rope, coeffs=coeffs_y), head_dim // 4),
    )
    transforms_out = (
        (partial(_apply_tiled_matrix, matrix=projection), head_dim // 2),
        (partial(_apply_rope, coeffs=coeffs_x, inverse=True), head_dim // 4),
        (partial(_apply_rope, coeffs=coeffs_y, inverse=True), head_dim // 4),
    )
    return (
        partial(_apply_block_diagonal, transforms=transforms_q),
        partial(_apply_block_diagonal, transforms=transforms_kv),
        partial(_apply_block_diagonal, transforms=transforms_out),
    )


def _apply_tiled_matrix(feats: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    batch, num_heads, sequence_length, feature_dim = feats.shape
    cameras = matrix.shape[1]
    matrix_dim = matrix.shape[-1]
    if sequence_length % cameras != 0 or feature_dim % matrix_dim != 0:
        raise ValueError(
            f"Cannot tile matrix {matrix.shape} over features {feats.shape}."
        )
    matrix = matrix.to(device=feats.device, dtype=feats.dtype)
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape(
            batch,
            num_heads,
            cameras,
            -1,
            feature_dim // matrix_dim,
            matrix_dim,
        ),
    ).reshape(feats.shape)


def _rope_precompute_coeffs(
    positions: torch.Tensor,
    *,
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if feat_dim % 2 != 0:
        raise ValueError(f"RoPE feature dimension must be even, got {feat_dim}.")
    num_frequencies = feat_dim // 2
    frequencies = freq_scale * (
        freq_base
        ** (
            -torch.arange(num_frequencies, device=positions.device)[None, None, None]
            / num_frequencies
        )
    )
    angles = positions[None, None, :, None] * frequencies
    return torch.cos(angles), torch.sin(angles)


def _apply_rope(
    feats: torch.Tensor,
    coeffs: tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    cosine, sine = coeffs
    cosine = cosine.to(device=feats.device, dtype=feats.dtype)
    sine = sine.to(device=feats.device, dtype=feats.dtype)
    if cosine.shape[2] != feats.shape[2]:
        repeats = feats.shape[2] // cosine.shape[2]
        cosine = cosine.repeat(1, 1, repeats, 1)
        sine = sine.repeat(1, 1, repeats, 1)
    x, y = feats.chunk(2, dim=-1)
    if inverse:
        return torch.cat([cosine * x - sine * y, sine * x + cosine * y], dim=-1)
    return torch.cat([cosine * x + sine * y, -sine * x + cosine * y], dim=-1)


def _apply_block_diagonal(feats: torch.Tensor, transforms: tuple) -> torch.Tensor:
    functions, block_sizes = zip(*transforms)
    blocks = torch.split(feats, block_sizes, dim=-1)
    return torch.cat([function(block) for function, block in zip(functions, blocks)], dim=-1)


def _invert_se3(transforms: torch.Tensor) -> torch.Tensor:
    rotation_inv = transforms[..., :3, :3].transpose(-1, -2)
    output = torch.zeros_like(transforms)
    output[..., :3, :3] = rotation_inv
    output[..., :3, 3] = -torch.einsum(
        "...ij,...j->...i",
        rotation_inv,
        transforms[..., :3, 3],
    )
    output[..., 3, 3] = 1.0
    return output


def _lift_intrinsics(intrinsics: torch.Tensor) -> torch.Tensor:
    output = torch.zeros(
        intrinsics.shape[:-2] + (4, 4),
        device=intrinsics.device,
        dtype=intrinsics.dtype,
    )
    output[..., :3, :3] = intrinsics
    output[..., 3, 3] = 1.0
    return output


def _invert_intrinsics(intrinsics: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(intrinsics)
    output[..., 0, 0] = 1.0 / intrinsics[..., 0, 0]
    output[..., 1, 1] = 1.0 / intrinsics[..., 1, 1]
    output[..., 0, 2] = -intrinsics[..., 0, 2] / intrinsics[..., 0, 0]
    output[..., 1, 2] = -intrinsics[..., 1, 2] / intrinsics[..., 1, 1]
    output[..., 2, 2] = 1.0
    return output
