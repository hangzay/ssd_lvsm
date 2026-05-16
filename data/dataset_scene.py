# Copyright (c) 2026 Yihang Wu.
"""RealEstate10K-style multi-view dataset with optional DINO and DA3 supervision."""

import json
import os
import random
import traceback

import numpy as np
import PIL
import torch
import torch.nn.functional as F
import zarr
from dinov3.dinov3.data.transforms import make_classification_eval_transform
from torch.utils.data import Dataset

from utils.pose_utils import align_poses_umeyama


def _generate_points_from_depth(depth: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    batch, height, width = depth.shape[0], depth.shape[2], depth.shape[3]
    y, x = torch.meshgrid(
        torch.arange(0, height, dtype=torch.float32, device=depth.device),
        torch.arange(0, width, dtype=torch.float32, device=depth.device),
        indexing="ij",
    )
    pixel_coords = torch.stack((x, y, torch.ones_like(x)), dim=0).reshape(3, -1)
    pixel_coords = pixel_coords.unsqueeze(0).expand(batch, -1, -1)
    cam_coords = torch.bmm(torch.inverse(K), pixel_coords) * depth.view(batch, 1, -1)
    world_coords = torch.bmm(c2w[:, :3, :3], cam_coords) + c2w[:, :3, 3:4]
    return world_coords.view(batch, 3, height, width)


def backproject_depth_to_pointmap(depth: np.ndarray, K: np.ndarray, ext_w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_views = depth.shape[0]
    device = torch.device("cpu")
    depth_t = torch.from_numpy(depth).to(device=device, dtype=torch.float32).unsqueeze(1)
    K_t = torch.from_numpy(K).to(device=device, dtype=torch.float32)
    w2c_t = torch.from_numpy(ext_w2c).to(device=device, dtype=torch.float32)
    if w2c_t.shape[-2:] == (3, 4):
        pad_row = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=device).view(1, 1, 4)
        w2c_t = torch.cat([w2c_t, pad_row.expand(num_views, 1, 4)], dim=1)
    pointmap = _generate_points_from_depth(depth_t, K_t, torch.linalg.inv(w2c_t))
    return pointmap.detach().cpu().numpy(), depth > 0


class Dataset(Dataset):
    """Load scene JSON files and assemble multi-view training samples."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._json_cache = {}
        self._zarr_handles = {}

        try:
            with open(self.config.training.dataset_path, "r") as f:
                self.all_scene_paths = [path for path in f.read().splitlines() if path.strip()]
        except Exception as exc:
            print(f"Error reading dataset paths from '{self.config.training.dataset_path}'")
            raise exc

        self.inference = self.config.inference.get("if_inference", False)
        self.use_irepa = (not self.inference) and self.config.training.use_irepa
        self.use_spatial = (not self.inference) and self.config.training.use_spatial
        self.da3_cache_root = self.config.training.da3_cache_root

        self._dino_transform = None
        if self.use_irepa:
            dino_image_size = self.config.model.irepa.get("image_size", 224)
            dino_resize_size = int(256 * dino_image_size / 224)
            self._dino_transform = make_classification_eval_transform(
                resize_size=dino_resize_size,
                crop_size=dino_image_size,
            )

        if self.inference:
            self.view_idx_list = {}
            view_idx_path = self.config.inference.get("view_idx_file_path", None)
            if view_idx_path is not None and os.path.exists(view_idx_path):
                with open(view_idx_path, "r") as f:
                    self.view_idx_list = json.load(f)
                valid_scenes = {k for k, v in self.view_idx_list.items() if v is not None}
                self.all_scene_paths = [
                    scene for scene in self.all_scene_paths if os.path.splitext(os.path.basename(scene))[0] in valid_scenes
                ]

    def __len__(self):
        return len(self.all_scene_paths)

    def _get_da3_handles(self, scene_name: str):
        if scene_name not in self._zarr_handles:
            scene_dir = os.path.join(self.da3_cache_root, scene_name)
            depth_z = zarr.open(os.path.join(scene_dir, "depth.zarr"), mode="r")
            pose_z = zarr.open(os.path.join(scene_dir, "camera_pose.zarr"), mode="r")
            intr_z = zarr.open(os.path.join(scene_dir, "camera_intrinsics.zarr"), mode="r")
            self._zarr_handles[scene_name] = (depth_z, pose_z, intr_z)
        return self._zarr_handles[scene_name]

    def _load_da3_by_indices(self, scene_name: str, indices: np.ndarray):
        depth_z, pose_z, intr_z = self._get_da3_handles(scene_name)
        depth_np = np.array(depth_z[indices], dtype=np.float32)
        pose_np = np.array(pose_z[indices], dtype=np.float32)
        intr_np = np.array(intr_z[indices], dtype=np.float32)
        return depth_np, pose_np, intr_np

    def _build_pointmap_for_sample(self, depth_np, ext_teacher, K_teacher, input_c2ws):
        ext_ref_norm = torch.linalg.inv(input_c2ws).cpu().numpy().astype(np.float32)
        try:
            _, _, _, ext_teacher_aligned = align_poses_umeyama(
                ext_ref_norm,
                ext_teacher,
                return_aligned=True,
                ransac=False,
            )
        except Exception:
            traceback.print_exc()
            raise

        pointmap_aligned, valid_mask = backproject_depth_to_pointmap(depth_np, K_teacher, ext_teacher_aligned)
        pointmap_t = torch.from_numpy(pointmap_aligned).float().permute(0, 2, 3, 1).contiguous()
        pointmap_mask_t = torch.tensor(valid_mask, dtype=torch.bool)
        return pointmap_t, pointmap_mask_t, ext_teacher_aligned

    def preprocess_frames(self, frames_chosen, image_paths_chosen):
        resize_h = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size
        square_crop = self.config.training.get("square_crop", False)
        need_dino = self.use_irepa

        images_list = []
        intrinsics_np = np.zeros((len(frames_chosen), 4), dtype=np.float32)
        dino_images = [] if need_dino else None

        for idx, (frame, image_path) in enumerate(zip(frames_chosen, image_paths_chosen)):
            image = PIL.Image.open(image_path).convert("RGB")
            original_w, original_h = image.size
            if need_dino:
                dino_images.append(self._dino_transform(image))

            resize_w = int(resize_h / original_h * original_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)
            image = image.resize((resize_w, resize_h), resample=PIL.Image.LANCZOS)

            if square_crop:
                min_size = min(resize_h, resize_w)
                start_h = (resize_h - min_size) // 2
                start_w = (resize_w - min_size) // 2
                image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))
            else:
                start_w = start_h = 0

            images_list.append(np.array(image))
            fxfycxcy = np.array(frame["fxfycxcy"], dtype=np.float32)
            resize_ratio_x = resize_w / original_w
            resize_ratio_y = resize_h / original_h
            fxfycxcy *= np.array([resize_ratio_x, resize_ratio_y, resize_ratio_x, resize_ratio_y], dtype=np.float32)
            if square_crop:
                fxfycxcy[2] -= start_w
                fxfycxcy[3] -= start_h
            intrinsics_np[idx] = fxfycxcy

        images = torch.from_numpy(np.stack(images_list, axis=0).astype(np.float32) / 255.0).permute(0, 3, 1, 2).contiguous()
        intrinsics = torch.from_numpy(intrinsics_np)
        w2cs = np.stack([np.array(frame["w2c"], dtype=np.float32) for frame in frames_chosen])
        c2ws = torch.from_numpy(np.linalg.inv(w2cs).astype(np.float32))

        if need_dino:
            return images, intrinsics, c2ws, torch.stack(dino_images, dim=0)
        return images, intrinsics, c2ws

    def preprocess_poses(self, in_c2ws: torch.Tensor, scene_scale_factor: float = 1.35) -> torch.Tensor:
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1)
        avg_down = in_c2ws[:, :3, 1].mean(0)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1)

        avg_pose = torch.eye(4, device=in_c2ws.device)
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center
        in_c2ws = torch.linalg.inv(avg_pose) @ in_c2ws
        in_c2ws[:, :3, 3] /= scene_scale_factor * torch.max(torch.abs(in_c2ws[:, :3, 3]))
        return in_c2ws

    def view_selector(self, frames):
        if len(frames) < self.config.training.num_views:
            return None
        view_selector_config = self.config.training.view_selector
        min_frame_dist = view_selector_config.get("min_frame_dist", 25)
        max_frame_dist = min(len(frames) - 1, view_selector_config.get("max_frame_dist", 100))
        if max_frame_dist <= min_frame_dist:
            return None
        frame_dist = random.randint(min_frame_dist, max_frame_dist)
        if len(frames) <= frame_dist:
            return None
        start_frame = random.randint(0, len(frames) - frame_dist - 1)
        end_frame = start_frame + frame_dist
        sampled_frames = random.sample(range(start_frame + 1, end_frame), self.config.training.num_views - 2)
        return [start_frame, end_frame] + sampled_frames

    def _inference_indices(self, scene_name):
        view_cfg = self.view_idx_list[scene_name]
        base_ctx = view_cfg["context"]
        target_indices = view_cfg["target"]
        need_ctx = self.config.training.num_input_views
        if need_ctx == 1:
            context_indices = [base_ctx[0]]
        elif need_ctx == 2:
            context_indices = list(base_ctx)
        else:
            lo, hi = sorted(base_ctx[:2])
            step = (hi - lo) / (need_ctx - 1)
            used = set(base_ctx) | set(target_indices)
            inserted = []
            for k in range(1, need_ctx - 1):
                candidate = lo + int(round(step * k))
                while candidate in used and candidate < hi:
                    candidate += 1
                if candidate < hi and candidate not in used:
                    inserted.append(candidate)
                    used.add(candidate)
            context_indices = [base_ctx[0]] + inserted + [base_ctx[1]]
        return context_indices + list(target_indices)

    def __getitem__(self, idx):
        scene_path = self.all_scene_paths[idx].strip()
        if scene_path not in self._json_cache:
            with open(scene_path, "r") as f:
                self._json_cache[scene_path] = json.load(f)
        data_json = self._json_cache[scene_path]
        frames = data_json["frames"]
        scene_name = data_json["scene_name"]

        if self.inference and scene_name in self.view_idx_list:
            image_indices = self._inference_indices(scene_name)
        else:
            image_indices = self.view_selector(frames)
            if image_indices is None:
                return self.__getitem__(random.randint(0, len(self) - 1))

        image_paths_chosen = [frames[idx]["image_path"] for idx in image_indices]
        frames_chosen = [frames[idx] for idx in image_indices]
        if self.use_irepa:
            input_images, input_intrinsics, input_c2ws, dino_images = self.preprocess_frames(frames_chosen, image_paths_chosen)
        else:
            input_images, input_intrinsics, input_c2ws = self.preprocess_frames(frames_chosen, image_paths_chosen)

        input_c2ws = self.preprocess_poses(input_c2ws, self.config.training.get("scene_scale_factor", 1.35))
        image_indices_t = torch.tensor(image_indices).long().unsqueeze(-1)
        scene_indices = torch.full_like(image_indices_t, idx)

        out = {
            "image": input_images,
            "c2w": input_c2ws,
            "fxfycxcy": input_intrinsics,
            "index": torch.cat([image_indices_t, scene_indices], dim=-1),
            "scene_name": scene_name,
        }

        if self.use_spatial:
            num_views = input_intrinsics.shape[0]
            intrinsics_matrix = torch.zeros(num_views, 3, 3, dtype=input_intrinsics.dtype)
            intrinsics_matrix[:, 0, 0] = input_intrinsics[:, 0]
            intrinsics_matrix[:, 1, 1] = input_intrinsics[:, 1]
            intrinsics_matrix[:, 0, 2] = input_intrinsics[:, 2]
            intrinsics_matrix[:, 1, 2] = input_intrinsics[:, 3]
            intrinsics_matrix[:, 2, 2] = 1.0
            out["intrinsics"] = intrinsics_matrix

            z_indices = image_indices_t.view(-1).cpu().numpy().astype(np.int64)
            depth_np, pose_np, intr_np = self._load_da3_by_indices(scene_name, z_indices)
            try:
                pointmap_t, pointmap_mask_t, spatial_w2c = self._build_pointmap_for_sample(depth_np, pose_np, intr_np, input_c2ws)
            except Exception:
                print(f"align_poses_umeyama failed for scene '{scene_name}', skip sample.")
                return self.__getitem__(random.randint(0, len(self) - 1))
            out["pointmap"] = pointmap_t
            out["pointmap_mask"] = pointmap_mask_t
            out["spatial_w2c"] = torch.from_numpy(spatial_w2c).float()

        if self.use_irepa:
            out["dino_images"] = dino_images
        return out
