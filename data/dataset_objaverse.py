import glob
import json
import os
import random
from math import radians, tan

import numpy as np
import PIL
import torch
from dinov3.dinov3.data.transforms import make_classification_eval_transform

from data.dataset_scene import Dataset as SceneDataset


class Dataset(SceneDataset):
    """Objaverse rendered-view dataset with optional DINO and DA3 supervision."""

    _MAX_RETRIES = 50

    def __init__(self, config):
        torch.utils.data.Dataset.__init__(self)
        self.config = config
        self._zarr_handles = {}
        self._scene_file_cache = {}
        self.inference = config.inference.get("if_inference", False)

        if self.inference:
            dataset_path = config.inference.test_dataset_path
            self.objaverse_root = config.inference.test_objaverse_root
            self.da3_cache_root = config.inference.get("test_da3_cache_root", config.training.da3_cache_root)
        else:
            dataset_path = config.training.dataset_path
            self.objaverse_root = config.training.objaverse_root
            self.da3_cache_root = config.training.da3_cache_root

        self.fovy_deg = config.training.get("objaverse_fovy_deg", 49.1)
        self.umeyama_centre_thresh = config.training.get("umeyama_centre_thresh", 0.15)

        try:
            with open(dataset_path, "r") as f:
                self.all_scene_paths = [line.strip() for line in f if line.strip()]
        except Exception as exc:
            print(f"Error reading dataset paths from '{dataset_path}'")
            raise exc

        self.use_irepa = (not self.inference) and config.training.use_irepa
        self.use_spatial = (not self.inference) and config.training.use_spatial

        self._dino_transform = None
        if self.use_irepa:
            dino_image_size = config.model.irepa.get("image_size", 224)
            dino_resize_size = int(256 * dino_image_size / 224)
            self._dino_transform = make_classification_eval_transform(
                resize_size=dino_resize_size,
                crop_size=dino_image_size,
            )

        if self.inference:
            self.view_idx_list = {}
            view_idx_path = config.inference.get("view_idx_file_path", None)
            if view_idx_path is not None and os.path.exists(view_idx_path):
                with open(view_idx_path, "r") as f:
                    self.view_idx_list = json.load(f)
                valid_scenes = {k for k, v in self.view_idx_list.items() if v is not None}
                self.all_scene_paths = [scene for scene in self.all_scene_paths if scene.strip() in valid_scenes]

    def _resolve_scene_dir(self, scene_id: str) -> str:
        rendered = os.path.join(self.objaverse_root, "rendered", scene_id)
        if os.path.isdir(rendered):
            return rendered
        flat = os.path.join(self.objaverse_root, scene_id)
        if os.path.isdir(flat):
            return flat
        return rendered

    def _get_scene_images(self, scene_id: str):
        if scene_id not in self._scene_file_cache:
            scene_dir = self._resolve_scene_dir(scene_id)
            self._scene_file_cache[scene_id] = sorted(glob.glob(os.path.join(scene_dir, "*.png")))
        return self._scene_file_cache[scene_id]

    @staticmethod
    def _compute_fxfycxcy(width: int, height: int, fovy_deg: float):
        fy = 0.5 * height / tan(0.5 * radians(fovy_deg))
        return [float(fy), float(fy), float(width) / 2.0, float(height) / 2.0]

    @staticmethod
    def _load_w2c_4x4(npy_path: str):
        rt = np.load(npy_path).astype(np.float32)
        if rt.shape == (3, 4):
            mat = np.eye(4, dtype=np.float32)
            mat[:3, :4] = rt
            return mat.tolist()
        if rt.shape == (4, 4):
            return rt.tolist()
        raise ValueError(f"Unexpected RT shape {rt.shape} in {npy_path}")

    @staticmethod
    def _load_image_size(path: str):
        try:
            with PIL.Image.open(path) as image:
                return image.width, image.height
        except Exception:
            return 512, 512

    def view_selector(self, frames):
        num_views = self.config.training.num_views
        num_frames = len(frames)
        if num_frames < num_views:
            return None

        max_window = self.config.training.get("view_window_frames", num_frames // 2)
        max_window = max(num_views, min(max_window, num_frames))
        max_start = num_frames - max_window
        start = random.randint(0, max_start) if max_start > 0 else 0
        window = list(range(start, start + max_window))
        step = max(1, int(np.ceil(max_window / num_views)))
        indices = [window[min(idx * step, max_window - 1)] for idx in range(num_views)]
        if len(set(indices)) < num_views:
            indices = sorted(random.sample(window, num_views))
        return indices

    def preprocess_frames(self, frames_chosen, image_paths_chosen):
        resize_h = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size
        square_crop = self.config.training.get("square_crop", False)

        images_list = []
        intrinsics_np = np.zeros((len(frames_chosen), 4), dtype=np.float32)
        dino_images = [] if self.use_irepa else None

        for idx, (frame, image_path) in enumerate(zip(frames_chosen, image_paths_chosen)):
            image = PIL.Image.open(image_path).convert("RGB")
            original_w, original_h = image.size
            if self.use_irepa:
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

        if self.use_irepa:
            return images, intrinsics, c2ws, torch.stack(dino_images, dim=0)
        return images, intrinsics, c2ws

    def __getitem__(self, idx):
        for _ in range(self._MAX_RETRIES):
            result = self._try_getitem(idx)
            if result is not None:
                return result
            idx = random.randint(0, len(self) - 1)
        raise RuntimeError(f"[dataset_objaverse] Exhausted {self._MAX_RETRIES} retries, last idx={idx}")

    def _try_getitem(self, idx):
        scene_id = self.all_scene_paths[idx].strip()
        image_paths = self._get_scene_images(scene_id)
        if len(image_paths) < self.config.training.num_views:
            return None

        width, height = self._load_image_size(image_paths[0])
        fxfycxcy = self._compute_fxfycxcy(width, height, self.fovy_deg)
        frames = []
        for image_path in image_paths:
            stem = os.path.splitext(os.path.basename(image_path))[0]
            npy_path = os.path.join(os.path.dirname(image_path), f"{stem}.npy")
            if os.path.exists(npy_path):
                frames.append({"fxfycxcy": fxfycxcy, "w2c": self._load_w2c_4x4(npy_path), "_image_path": image_path})
        if len(frames) < self.config.training.num_views:
            return None

        if self.inference and scene_id in self.view_idx_list:
            view_cfg = self.view_idx_list[scene_id]
            image_indices = view_cfg["context"] + view_cfg["target"]
        else:
            image_indices = self.view_selector(frames)
        if image_indices is None or len(image_indices) != self.config.training.num_views:
            return None
        if max(image_indices) >= len(frames) or min(image_indices) < 0:
            return None

        frames_chosen = [frames[idx] for idx in image_indices]
        image_paths_chosen = [frames[idx]["_image_path"] for idx in image_indices]
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
            "scene_name": scene_id,
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

            depth_np, pose_np, intr_np = self._load_da3_by_indices(scene_id, np.array(image_indices, dtype=np.int64))
            try:
                pointmap_t, pointmap_mask_t, spatial_w2c = self._build_pointmap_for_sample(depth_np, pose_np, intr_np, input_c2ws)
            except Exception:
                print(f"align_poses_umeyama exception for scene '{scene_id}', skip.")
                return None

            aligned_c2ws = np.linalg.inv(spatial_w2c.astype(np.float64))
            ref_centres = input_c2ws[:, :3, 3].numpy().astype(np.float64)
            centre_err = np.linalg.norm(ref_centres - aligned_c2ws[:, :3, 3], axis=1).mean()
            if centre_err > self.umeyama_centre_thresh or np.isnan(centre_err):
                print(f"[umeyama] scene '{scene_id}' centre_err={centre_err:.4f} > thresh {self.umeyama_centre_thresh}, skip sample.")
                return None

            out["pointmap"] = pointmap_t
            out["pointmap_mask"] = pointmap_mask_t
            out["spatial_w2c"] = torch.from_numpy(spatial_w2c).float()

        if self.use_irepa:
            out["dino_images"] = dino_images
        return out
