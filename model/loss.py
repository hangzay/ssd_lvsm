# Copyright (c) 2026 Yihang Wu.

import os
from pathlib import Path

import lpips
import scipy.io
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from torchvision.models import vgg19

from model.loss_api import SpatialConsistencyLoss


class PerceptualLoss(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.vgg = self._build_vgg()
        self._load_weights()
        self.blocks = nn.ModuleList()
        output_indices = [0, 4, 9, 14, 23, 32]
        for i in range(len(output_indices) - 1):
            self.blocks.append(nn.Sequential(*list(self.vgg.features[output_indices[i] : output_indices[i + 1]])).to(device).eval())
        for param in self.vgg.parameters():
            param.requires_grad = False

    def _build_vgg(self):
        model = vgg19()
        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.MaxPool2d):
                model.features[i] = nn.AvgPool2d(kernel_size=2, stride=2)
        return model.to(self.device).eval()

    def _load_weights(self):
        weight_file = Path("./metric_checkpoint/imagenet-vgg-verydeep-19.mat")
        weight_file.parent.mkdir(exist_ok=True, parents=True)
        if dist.get_rank() == 0 and not weight_file.exists():
            os.system(f"wget https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat -O {weight_file}")
        dist.barrier()

        vgg_data = scipy.io.loadmat(weight_file)
        vgg_layers = vgg_data["layers"][0]
        layer_indices = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        filter_sizes = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
        with torch.no_grad():
            for i, layer_idx in enumerate(layer_indices):
                weights = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][0]).permute(3, 2, 0, 1)
                biases = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][1]).view(filter_sizes[i])
                self.vgg.features[layer_idx].weight = nn.Parameter(weights, requires_grad=False)
                self.vgg.features[layer_idx].bias = nn.Parameter(biases, requires_grad=False)

    def forward(self, pred_img, target_img):
        mean = torch.tensor([123.6800, 116.7790, 103.9390], device=pred_img.device).reshape(1, 3, 1, 1)
        pred = pred_img * 255.0 - mean
        target = target_img * 255.0 - mean
        loss = torch.mean(torch.abs(target - pred))
        for block, scale in zip(self.blocks, [2.6, 4.8, 3.7, 5.6, 0.15]):
            pred = block(pred)
            target = block(target)
            loss = loss + torch.mean(torch.abs(target - pred)) / scale
        return loss / 255.0


class LossComputer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.training.lpips_loss_weight > 0.0:
            if dist.get_rank() == 0:
                self.lpips_loss_module = self._freeze(lpips.LPIPS(net="vgg"))
            dist.barrier()
            if dist.get_rank() != 0:
                self.lpips_loss_module = self._freeze(lpips.LPIPS(net="vgg"))
        if config.training.perceptual_loss_weight > 0.0:
            self.perceptual_loss_module = self._freeze(PerceptualLoss())
        if config.training.use_spatial and config.training.spatial_loss_weight > 0.0:
            self.spatial_loss_module = SpatialConsistencyLoss(
                temperature=config.training.get("spatial_temperature", 0.1),
                gamma=config.training.get("spatial_gamma", 1.0),
                alpha=config.training.get("spatial_alpha", 0.2),
                num_input_views=config.training.num_input_views,
                min_valid_correspondences=config.training.get("spatial_min_valid", 100),
            )

    @staticmethod
    def _freeze(module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
        return module

    @staticmethod
    def mean_flat(x):
        return torch.mean(x, dim=list(range(1, len(x.size()))))

    def compute_dino_loss(self, irepa_features, device):
        student_features = irepa_features["student_features"]
        teacher_features = irepa_features["teacher_features"]
        if student_features is None or teacher_features is None:
            return torch.tensor(0.0, device=device)
        if student_features.shape != teacher_features.shape:
            raise ValueError(f"DINO feature shape mismatch: student={student_features.shape}, teacher={teacher_features.shape}")
        return self.mean_flat(F.smooth_l1_loss(student_features, teacher_features, reduction="none")).mean()

    def compute_rgb_losses(self, rendering, target):
        if target.size(2) == 4:
            target, _ = target.split([3, 1], dim=2)
        if rendering.size(2) == 4:
            rendering, _ = rendering.split([3, 1], dim=2)

        l2_loss = F.mse_loss(rendering, target)
        psnr = -10.0 * torch.log10(l2_loss + 1e-8)
        lpips_loss = torch.tensor(0.0, device=rendering.device)
        perceptual_loss = torch.tensor(0.0, device=rendering.device)

        if self.config.training.lpips_loss_weight > 0.0:
            lpips_loss = self.lpips_loss_module(
                rendering.reshape(-1, *rendering.shape[2:]) * 2.0 - 1.0,
                target.reshape(-1, *target.shape[2:]) * 2.0 - 1.0,
            ).mean()
        if self.config.training.perceptual_loss_weight > 0.0:
            perceptual_loss = self.perceptual_loss_module(
                rendering.reshape(-1, *rendering.shape[2:]),
                target.reshape(-1, *target.shape[2:]),
            )
        return l2_loss, psnr, lpips_loss, perceptual_loss

    def forward(self, out: dict, target: torch.Tensor, irepa_features: dict) -> edict:
        device = target.device
        rendering = out["decoder_rgb"]
        l2_loss, psnr, lpips_loss, perceptual_loss = self.compute_rgb_losses(rendering, target)

        dino_loss = torch.tensor(0.0, device=device)
        if self.config.training.use_irepa and self.config.training.dino_loss_weight > 0.0:
            dino_loss = self.compute_dino_loss(irepa_features, device)

        spatial_loss = torch.tensor(0.0, device=device)
        if self.config.training.use_spatial and self.config.training.spatial_loss_weight > 0.0:
            spatial_loss, _ = self.spatial_loss_module.compute_loss(out)

        loss = (
            self.config.training.l2_loss_weight * l2_loss
            + self.config.training.lpips_loss_weight * lpips_loss
            + self.config.training.perceptual_loss_weight * perceptual_loss
            + self.config.training.dino_loss_weight * dino_loss
            + self.config.training.spatial_loss_weight * spatial_loss
        )

        metrics = edict(loss=loss, psnr=psnr, l2_loss=l2_loss)
        if self.config.training.lpips_loss_weight > 0.0:
            metrics.lpips_loss = lpips_loss
        if self.config.training.perceptual_loss_weight > 0.0:
            metrics.perceptual_loss = perceptual_loss
        if self.config.training.use_irepa:
            metrics.dino_loss = dino_loss
        if self.config.training.use_spatial:
            metrics.spatial_loss = spatial_loss
        return metrics
