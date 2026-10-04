# Copyright (c) 2026 Yihang Wu.
#
# Backbone adapted from https://github.com/Haian-Jin/LVSM.
# Upstream terms: LICENSES/LVSM-CC-BY-NC-SA-4.0.md.
# This project adds semantic-spatial decoupling, modulation, and supervision.

import logging
import os
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from utils import camera_utils, data_utils
from .camera_conditioning import encode_rays
from .loss import LossComputer
from .transformer import DecoupledTransformerBlock, init_weights


DINO_VITB16_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
MOD_SCALE_INIT = 1.0


class DecoupledNVSEncoderDecoder(nn.Module):
    """Encoder-decoder feedforward NVS model with semantic-spatial decoupled tokens."""

    def __init__(self, config, logger=None):
        super().__init__()
        self.config = config
        self.logger = logger or self._get_default_logger()
        self.process_data = data_utils.ProcessData(config)
        transformer_cfg = self.config.model.transformer
        self.decouple = bool(transformer_cfg.get("decouple", True))
        self.film = bool(transformer_cfg.get("film", True))
        self.use_dino = bool(self.config.training.get("dino", True))
        self.use_spatial_supervision = bool(self.config.training.get("spatial_supervision", True))
        camera_cfg = self.config.model.get("camera_encoding", {})
        self.ray_encoding = camera_cfg.get("ray_encoding", "plucker")
        self.attention_encoding = camera_cfg.get("attention_encoding", "none")
        if self.attention_encoding != "none":
            raise ValueError(
                "PRoPE/GTA is defined for per-view token sequences and is not directly "
                "applicable to encoder-decoder LVSM's camera-less scene latents. "
                "Use ray_encoding=raymap for the encoder-decoder camera-encoding control."
            )
        if not self.decouple and (self.film or self.use_dino or self.use_spatial_supervision):
            raise ValueError("film, dino, and spatial_supervision require decouple=true")
        self._init_tokenizers()
        self._init_transformer()
        self.loss_computer = LossComputer(config) if not config.inference.if_inference else nn.Identity()
        self._init_irepa_head()
        self._init_dino_teacher()

    def _get_default_logger(self):
        logger = logging.getLogger("DecoupledNVS.EncoderDecoder")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    def _create_tokenizer(self, in_channels, patch_size, d_out):
        tokenizer = nn.Sequential(
            Rearrange("b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)", ph=patch_size, pw=patch_size),
            nn.Linear(in_channels * (patch_size**2), d_out, bias=False),
        )
        tokenizer.apply(init_weights)
        return tokenizer

    def _init_tokenizers(self):
        self.d_model = self.config.model.transformer.d
        self.d_half = self.d_model // 2
        assert self.d_model % 2 == 0

        image_cfg = self.config.model.image_tokenizer
        pose_cfg = self.config.model.pose_tokenizer
        target_pose_cfg = self.config.model.target_pose_tokenizer
        if self.decouple:
            self.image_tokenizer = self._create_tokenizer(image_cfg.in_channels, image_cfg.patch_size, self.d_half)
            self.pose_tokenizer = self._create_tokenizer(pose_cfg.in_channels, pose_cfg.patch_size, self.d_half)
            self.target_pose_tokenizer = self._create_tokenizer(
                target_pose_cfg.in_channels, target_pose_cfg.patch_size, self.d_half
            )
            self.decoder_norm_i = nn.LayerNorm(self.d_half, bias=False)
            self.decoder_norm_p = nn.LayerNorm(self.d_half, bias=False)
        else:
            self.image_tokenizer = self._create_tokenizer(
                image_cfg.in_channels + pose_cfg.in_channels, image_cfg.patch_size, self.d_model
            )
            self.target_pose_tokenizer = self._create_tokenizer(
                target_pose_cfg.in_channels, target_pose_cfg.patch_size, self.d_model
            )
            self.decoder_norm = nn.LayerNorm(self.d_model, bias=False)

        patch_size = target_pose_cfg.patch_size
        self.decoder_linear = nn.Linear(self.d_model, patch_size * patch_size * 3, bias=False)
        self.decoder_sigmoid = nn.Sigmoid()
        self.decoder_linear.apply(init_weights)

        image_size = image_cfg.image_size
        patch = image_cfg.patch_size
        self.grid_h = image_size // patch
        self.grid_w = self.grid_h
        self.tokens_per_view = self.grid_h * self.grid_w
        self.num_input_views = self.config.training.num_input_views

    def _make_block(self, cfg):
        return DecoupledTransformerBlock(
            cfg.d,
            cfg.d_head,
            decouple=self.decouple,
            film=self.film,
            film_scale_init=cfg.get("film_scale_init", MOD_SCALE_INIT),
        )

    def _init_transformer(self):
        cfg = self.config.model.transformer
        self.scene_latents = nn.Parameter(torch.randn(cfg.n_latent_vectors, cfg.d))
        nn.init.trunc_normal_(self.scene_latents, std=0.02)
        self.transformer_encoder = nn.ModuleList([self._make_block(cfg) for _ in range(cfg.encoder_n_layer)])
        self.transformer_decoder = nn.ModuleList([self._make_block(cfg) for _ in range(cfg.decoder_n_layer)])

        for blocks, _ in ((self.transformer_encoder, cfg.encoder_n_layer), (self.transformer_decoder, cfg.decoder_n_layer)):
            for idx, block in enumerate(blocks):
                std = 0.02 / (2 * (idx + 1)) ** 0.5
                block.apply(lambda module, init_std=std: init_weights(module, init_std))
                block.reset_film_parameters(cfg.get("film_scale_init", MOD_SCALE_INIT))

        self.transformer_input_layernorm_decoder = nn.LayerNorm(cfg.d, bias=False)
        self.logger.info(
            f"Initialized encoder-decoder LVSM: {cfg.encoder_n_layer}+{cfg.decoder_n_layer} layers, "
            f"decouple={self.decouple}, film={self.film}, dino={self.use_dino}, spatial_supervision={self.use_spatial_supervision}, "
            f"ray_encoding={self.ray_encoding}."
        )

    def _init_irepa_head(self):
        if self.config.inference.if_inference or not self.use_dino:
            return
        self.dino_gamma = self.config.training.get("dino_gamma", 0.60)
        self.dino_proj_head = nn.Conv2d(self.d_half, self.config.model.irepa.feature_dim, kernel_size=3, padding=1)
        self.dino_proj_head.apply(init_weights)

    def _init_dino_teacher(self):
        self.dino_teacher = None
        if self.config.inference.if_inference or not self.use_dino:
            return
        from utils.dino_utils import load_dino_teacher

        cfg = self.config.model.irepa
        self.dino_layer = cfg.dino_layer
        self.dino_feature_dim = cfg.feature_dim
        self.dino_image_size = cfg.get("image_size", 448)
        self.dino_teacher = load_dino_teacher(
            "vitb16", cfg.weight_path, repo=cfg.get("repo", "facebookresearch/dinov3")
        ).to(next(self.parameters()).device)
        self.dino_h = self.dino_image_size // 16
        self.dino_w = self.dino_h

    def train(self, mode=True):
        super().train(mode)
        self.loss_computer.eval()
        if self.dino_teacher is not None:
            self.dino_teacher.eval()
        return self

    def pass_layers(self, transformer_blocks, input_tokens, gradient_checkpoint=False, checkpoint_every=1, capture_dino_layer=None, capture_spatial_layer=None):
        dino_feat = None
        spatial_feat = None
        if not gradient_checkpoint:
            for layer_idx, layer in enumerate(transformer_blocks):
                input_tokens = layer(input_tokens)
                if capture_dino_layer is not None and layer_idx == capture_dino_layer:
                    dino_feat = input_tokens.clone()
                if capture_spatial_layer is not None and layer_idx == capture_spatial_layer:
                    spatial_feat = input_tokens.clone()
            return input_tokens, dino_feat, spatial_feat

        def process_group(tokens, start_idx, end_idx):
            for idx in range(start_idx, end_idx):
                tokens = transformer_blocks[idx](tokens)
            return tokens

        for start_idx in range(0, len(transformer_blocks), checkpoint_every):
            end_idx = min(start_idx + checkpoint_every, len(transformer_blocks))
            input_tokens = torch.utils.checkpoint.checkpoint(process_group, input_tokens, start_idx, end_idx, use_reentrant=False)
            if capture_dino_layer is not None and start_idx <= capture_dino_layer < end_idx:
                dino_feat = input_tokens.clone()
            if capture_spatial_layer is not None and start_idx <= capture_spatial_layer < end_idx:
                spatial_feat = input_tokens.clone()
        return input_tokens, dino_feat, spatial_feat

    def get_posed_input(self, images=None, ray_o=None, ray_d=None):
        pose_cond = encode_rays(ray_o, ray_d, self.ray_encoding)
        if images is None:
            batch_size, num_views, _, height, width = pose_cond.shape
            image = torch.zeros((batch_size, num_views, 3, height, width), device=pose_cond.device, dtype=pose_cond.dtype)
        else:
            image = images * 2.0 - 1.0
        return image, pose_cond

    def _decode_image_tokens(self, tokens):
        if self.decouple:
            semantic = self.decoder_norm_i(tokens[..., : self.d_half])
            spatial = self.decoder_norm_p(tokens[..., self.d_half :])
            tokens = torch.cat([semantic, spatial], dim=-1)
        else:
            tokens = self.decoder_norm(tokens)
        return self.decoder_sigmoid(self.decoder_linear(tokens))

    def _align_views_tensor(self, per_view_tensor, num_input_views):
        batch_size, num_views = per_view_tensor.shape[:2]
        rest_shape = per_view_tensor.shape[2:]
        num_target_views = num_views - num_input_views
        context = per_view_tensor[:, :num_input_views].repeat_interleave(num_target_views, dim=0)
        targets = per_view_tensor[:, num_input_views:].reshape(batch_size * num_target_views, 1, *rest_shape)
        return torch.cat([context, targets], dim=1)

    @torch.inference_mode()
    def _compute_dino_teacher_features(self, dino_images):
        if self.dino_teacher is None:
            return None
        batch_size, num_views = dino_images.shape[:2]
        x = dino_images.reshape(
            batch_size * num_views, 3, self.dino_image_size, self.dino_image_size
        )
        inter = self.dino_teacher.get_intermediate_layers(x=x, n=[self.dino_layer])[0]
        batch_views, num_tokens, feature_dim = inter.shape
        expected_tokens = self.dino_h * self.dino_w
        if num_tokens != expected_tokens:
            raise RuntimeError(f"DINO token count {num_tokens} != expected {expected_tokens}")
        grid = inter.view(
            batch_views, self.dino_h, self.dino_w, feature_dim
        ).permute(0, 3, 1, 2).contiguous()
        interp = F.interpolate(
            grid,
            size=(self.grid_h, self.grid_w),
            mode=self.config.model.irepa.get("interpolate_mode", "bilinear"),
            align_corners=False,
        )
        return interp.permute(0, 2, 3, 1).contiguous().view(
            batch_size, num_views, self.tokens_per_view, feature_dim
        ).to(torch.float32)

    def get_irepa_features(self, student_features, target_dino_features):
        if self.config.inference.if_inference or not self.use_dino or student_features is None:
            return {"student_features": None, "teacher_features": None}
        semantic = student_features[..., : self.d_half]
        mean_token = semantic.mean(dim=1, keepdim=True)
        semantic = (semantic - self.dino_gamma * mean_token) / (semantic.std(dim=1, keepdim=True) + 1e-6)
        batch_views, total_tokens, _ = semantic.shape
        view_feat = semantic.view(batch_views, self.grid_h, self.grid_w, self.d_half).permute(0, 3, 1, 2)
        projected = self.dino_proj_head(view_feat).permute(0, 2, 3, 1).reshape(batch_views, total_tokens, -1)
        return {"student_features": projected, "teacher_features": target_dino_features}

    def forward(self, data_batch, has_target_image=True):
        input, target = self.process_data(
            data_batch,
            has_target_image=has_target_image,
            target_has_input=self.config.training.target_has_input,
            compute_rays=True,
        )
        batch_size, num_input_views = input.image.shape[:2]
        num_target_views = target.ray_o.shape[1]
        num_latents = self.config.model.transformer.n_latent_vectors
        use_dino = (not self.config.inference.if_inference) and self.use_dino
        use_spatial_supervision = (not self.config.inference.if_inference) and self.use_spatial_supervision

        input_images, input_pose = self.get_posed_input(images=input.image, ray_o=input.ray_o, ray_d=input.ray_d)
        if self.decouple:
            input_tokens = torch.cat([self.image_tokenizer(input_images), self.pose_tokenizer(input_pose)], dim=-1)
        else:
            input_tokens = self.image_tokenizer(torch.cat([input_images, input_pose], dim=2))
        _, num_patches, _ = input_tokens.shape
        input_tokens = input_tokens.reshape(batch_size, num_input_views * num_patches, self.d_model)

        latent = self.scene_latents.expand(batch_size, -1, -1)
        encoder_input = torch.cat([latent, input_tokens], dim=1)
        checkpoint_every = max(1, self.config.training.grad_checkpoint_every)
        use_checkpoint = self.config.training.grad_checkpoint_every > 0
        encoder_output, _, enc_spatial = self.pass_layers(
            self.transformer_encoder,
            encoder_input,
            gradient_checkpoint=use_checkpoint,
            checkpoint_every=checkpoint_every,
            capture_spatial_layer=self.config.training.get("spatial_layer", 9) if use_spatial_supervision else None,
        )
        latent_tokens, _ = encoder_output.split([num_latents, num_input_views * num_patches], dim=1)

        _, target_pose = self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)
        target_pose_tokens = self.target_pose_tokenizer(target_pose)
        if self.decouple:
            target_semantic = torch.zeros(
                batch_size * num_target_views,
                num_patches,
                self.d_half,
                device=target_pose_tokens.device,
                dtype=target_pose_tokens.dtype,
            )
            target_tokens = torch.cat([target_semantic, target_pose_tokens], dim=-1)
        else:
            target_tokens = target_pose_tokens
        repeated_latent = repeat(latent_tokens, "b nl d -> (b vt) nl d", vt=num_target_views)
        decoder_input = self.transformer_input_layernorm_decoder(torch.cat([target_tokens, repeated_latent], dim=1))
        decoder_output, dino_feat, dec_spatial = self.pass_layers(
            self.transformer_decoder,
            decoder_input,
            gradient_checkpoint=use_checkpoint,
            checkpoint_every=checkpoint_every,
            capture_dino_layer=self.config.model.irepa.decoder_depth if use_dino else None,
            capture_spatial_layer=self.config.training.get("spatial_layer", 9) if use_spatial_supervision else None,
        )

        target_image_tokens, _ = decoder_output.split([num_patches, num_latents], dim=1)
        rendered = self._decode_image_tokens(target_image_tokens)
        height, width = target.image_h_w
        patch_size = self.config.model.target_pose_tokenizer.patch_size
        rendered = rearrange(
            rendered,
            "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=num_target_views,
            h=height // patch_size,
            w=width // patch_size,
            p1=patch_size,
            p2=patch_size,
            c=3,
        )
        out = {"decoder_rgb": rendered}

        if use_spatial_supervision:
            if enc_spatial is None or dec_spatial is None:
                raise RuntimeError("spatial_supervision=True but no spatial feature was captured.")
            enc_features = enc_spatial[:, num_latents:, self.d_half :].view(batch_size, num_input_views, self.grid_h, self.grid_w, self.d_half)
            enc_features = enc_features.repeat_interleave(num_target_views, dim=0)
            dec_features = dec_spatial[:, :num_patches, self.d_half :].view(batch_size * num_target_views, 1, self.grid_h, self.grid_w, self.d_half)
            out["spatial_features"] = torch.cat([enc_features, dec_features], dim=1)
            all_w2cs = torch.cat([input.spatial_w2c, target.spatial_w2c], dim=1)
            out["w2cs"] = self._align_views_tensor(all_w2cs, num_input_views)
            if "intrinsics" in input and "intrinsics" in target:
                intrinsics = torch.cat([input.intrinsics, target.intrinsics], dim=1)
                out["intrinsics"] = self._align_views_tensor(intrinsics.to(all_w2cs.device), num_input_views)
            out["gt_point"] = {
                "pts3d": self._align_views_tensor(torch.cat([input.pointmap, target.pointmap], dim=1).to(all_w2cs.device), num_input_views),
                "valid_mask": self._align_views_tensor(torch.cat([input.pointmap_mask, target.pointmap_mask], dim=1).to(all_w2cs.device), num_input_views),
            }

        irepa_features = {"student_features": None, "teacher_features": None}
        if use_dino and "dino_images" in target and dino_feat is not None:
            target_dino = self._compute_dino_teacher_features(target.dino_images)
            if target_dino is not None:
                target_dino = target_dino.reshape(batch_size * num_target_views, num_patches, self.dino_feature_dim)
                irepa_features = self.get_irepa_features(dino_feat[:, :num_patches, :], target_dino)

        loss_metrics = None
        if has_target_image and not self.config.inference.if_inference:
            loss_metrics = self.loss_computer(out, target.image, irepa_features)
        return edict(input=input, target=target, loss_metrics=loss_metrics, render=rendered)

    @torch.no_grad()
    def render_video(self, data_batch, traj_type="interpolate", num_frames=60, loop_video=False, order_poses=False):
        if data_batch.input is None:
            input, target = self.process_data(
                data_batch,
                has_target_image=False,
                target_has_input=self.config.training.target_has_input,
                compute_rays=True,
            )
            data_batch = edict(input=input, target=target)
        else:
            input, target = data_batch.input, data_batch.target

        batch_size, num_input_views = input.image.shape[:2]
        num_latents = self.config.model.transformer.n_latent_vectors
        device = input.image.device
        input_images, input_pose = self.get_posed_input(images=input.image, ray_o=input.ray_o, ray_d=input.ray_d)
        if self.decouple:
            input_tokens = torch.cat([self.image_tokenizer(input_images), self.pose_tokenizer(input_pose)], dim=-1)
        else:
            input_tokens = self.image_tokenizer(torch.cat([input_images, input_pose], dim=2))
        _, num_patches, _ = input_tokens.shape
        input_tokens = input_tokens.reshape(batch_size, num_input_views * num_patches, self.d_model)
        latent = self.scene_latents.expand(batch_size, -1, -1)
        encoder_output, _, _ = self.pass_layers(self.transformer_encoder, torch.cat([latent, input_tokens], dim=1))
        latent_tokens, _ = encoder_output.split([num_latents, num_input_views * num_patches], dim=1)

        c2ws = input.c2w
        fxfycxcy = input.fxfycxcy
        intrinsics = torch.zeros((c2ws.shape[0], c2ws.shape[1], 3, 3), device=device)
        intrinsics[:, :, 0, 0] = fxfycxcy[:, :, 0]
        intrinsics[:, :, 1, 1] = fxfycxcy[:, :, 1]
        intrinsics[:, :, 0, 2] = fxfycxcy[:, :, 2]
        intrinsics[:, :, 1, 2] = fxfycxcy[:, :, 3]
        if loop_video:
            c2ws = torch.cat([c2ws, c2ws[:, [0], :]], dim=1)
            intrinsics = torch.cat([intrinsics, intrinsics[:, [0], :]], dim=1)

        all_c2ws, all_intrinsics = [], []
        for batch_idx in range(batch_size):
            cur_c2ws, cur_intrinsics = camera_utils.get_interpolated_poses_many(
                c2ws[batch_idx, :, :3, :4], intrinsics[batch_idx], num_frames, order_poses=order_poses
            )
            all_c2ws.append(cur_c2ws.to(device))
            all_intrinsics.append(cur_intrinsics.to(device))
        all_c2ws = torch.stack(all_c2ws, dim=0)
        all_intrinsics = torch.stack(all_intrinsics, dim=0)
        homogeneous = torch.tensor([[[0, 0, 0, 1]]], device=device).expand(all_c2ws.shape[0], all_c2ws.shape[1], -1, -1)
        all_c2ws = torch.cat([all_c2ws, homogeneous], dim=2)
        all_fxfycxcy = torch.zeros((all_intrinsics.shape[0], all_intrinsics.shape[1], 4), device=device)
        all_fxfycxcy[:, :, 0] = all_intrinsics[:, :, 0, 0]
        all_fxfycxcy[:, :, 1] = all_intrinsics[:, :, 1, 1]
        all_fxfycxcy[:, :, 2] = all_intrinsics[:, :, 0, 2]
        all_fxfycxcy[:, :, 3] = all_intrinsics[:, :, 1, 2]
        ray_o, ray_d = self.process_data.compute_rays(all_c2ws, all_fxfycxcy, h=input.image.shape[-2], w=input.image.shape[-1], device=device)
        _, target_pose = self.get_posed_input(ray_o=ray_o, ray_d=ray_d)
        target_pose_tokens = self.target_pose_tokenizer(target_pose).reshape(
            batch_size, num_frames * num_patches, self.d_half if self.decouple else self.d_model
        )

        video = []
        patch_size = self.config.model.target_pose_tokenizer.patch_size
        for start in range(0, num_frames, 4):
            cur_views = min(4, num_frames - start)
            cur_pose = rearrange(
                target_pose_tokens[:, start * num_patches : (start + cur_views) * num_patches],
                "b (v p) d -> (b v) p d",
                v=cur_views,
            )
            if self.decouple:
                target_semantic = torch.zeros(
                    batch_size * cur_views, num_patches, self.d_half, device=device, dtype=cur_pose.dtype
                )
                cur_target = torch.cat([target_semantic, cur_pose], dim=-1)
            else:
                cur_target = cur_pose
            repeated_latent = repeat(latent_tokens, "b nl d -> (b cv) nl d", cv=cur_views)
            decoder_input = self.transformer_input_layernorm_decoder(torch.cat([cur_target, repeated_latent], dim=1))
            decoder_output, _, _ = self.pass_layers(self.transformer_decoder, decoder_input)
            target_tokens, _ = decoder_output.split([num_patches, num_latents], dim=1)
            frames = self._decode_image_tokens(target_tokens)
            frames = rearrange(
                frames,
                "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
                v=cur_views,
                h=target.image_h_w[0] // patch_size,
                w=target.image_h_w[1] // patch_size,
                p1=patch_size,
                p2=patch_size,
                c=3,
            ).cpu()
            video.append(frames)
        data_batch.video_rendering = torch.cat(video, dim=1)
        return data_batch

    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = sorted(name for name in os.listdir(load_path) if name.endswith(".pt"))
            ckpt_path = os.path.join(load_path, ckpt_names[-1])
        else:
            ckpt_path = load_path
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            traceback.print_exc()
            print(f"Failed to load {ckpt_path}")
            return None
        self.load_state_dict(checkpoint["model"], strict=False)
        print(f"[load_ckpt] Loaded {os.path.basename(ckpt_path)}")
        return 0
