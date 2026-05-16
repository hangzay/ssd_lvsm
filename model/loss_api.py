import torch
import torch.nn as nn
import torch.nn.functional as F


def sanitize_loss(loss_tensor: torch.Tensor, loss_name: str, hard_max: float = 100) -> torch.Tensor:
    if torch.isnan(loss_tensor).any() or torch.isinf(loss_tensor).any():
        print(f"[WARNING] {loss_name} has inf or nan. Setting those values to 0.")
        loss_tensor = torch.where(
            torch.isnan(loss_tensor) | torch.isinf(loss_tensor),
            torch.tensor(0.0, device=loss_tensor.device, dtype=loss_tensor.dtype),
            loss_tensor,
        )
    if hard_max is not None:
        loss_tensor = torch.clamp(loss_tensor, min=-hard_max, max=hard_max)
    return loss_tensor


def project_points_to_target_view(pointmap, target_w2cs, target_intrinsics, pointmap_mask):
    batch_views, num_input_views, height, width, _ = pointmap.shape
    num_target_views = target_w2cs.shape[1]
    device = pointmap.device

    points_h = torch.cat([pointmap, torch.ones(batch_views, num_input_views, height, width, 1, device=device, dtype=pointmap.dtype)], dim=-1)
    points_flat = points_h.reshape(batch_views, num_input_views, height * width, 4)

    projected = torch.zeros(batch_views, num_target_views, num_input_views, height, width, 2, device=device, dtype=pointmap.dtype)
    depths = torch.zeros(batch_views, num_target_views, num_input_views, height, width, device=device, dtype=pointmap.dtype)
    valid = torch.zeros(batch_views, num_target_views, num_input_views, height, width, device=device, dtype=torch.bool)

    for target_idx in range(num_target_views):
        w2c = target_w2cs[:, target_idx]
        intrinsics = target_intrinsics[:, target_idx]
        for input_idx in range(num_input_views):
            cam = torch.bmm(points_flat[:, input_idx], w2c.transpose(1, 2))[..., :3]
            z = cam[..., 2].clamp(min=1e-6)
            pix = torch.bmm(cam, intrinsics.transpose(1, 2))
            x = pix[..., 0] / z
            y = pix[..., 1] / z
            proj = torch.stack([x, y], dim=-1).reshape(batch_views, height, width, 2)
            depth = z.reshape(batch_views, height, width)
            in_bounds = (proj[..., 0] >= 0) & (proj[..., 0] < width) & (proj[..., 1] >= 0) & (proj[..., 1] < height)
            projected[:, target_idx, input_idx] = proj
            depths[:, target_idx, input_idx] = depth
            valid[:, target_idx, input_idx] = pointmap_mask[:, input_idx] & in_bounds & (depth > 0)
    return projected, depths, valid


def build_correspondence_map(projected_coords, depth_in_target, valid_projection, height, width):
    batch_views, num_target_views, num_input_views = projected_coords.shape[:3]
    device = projected_coords.device
    corr_map = torch.zeros(batch_views, num_target_views, height, width, 3, device=device, dtype=torch.long)
    corr_mask = torch.zeros(batch_views, num_target_views, height, width, device=device, dtype=torch.bool)
    z_buffer = torch.full((batch_views, num_target_views, height, width), float("inf"), device=device, dtype=depth_in_target.dtype)

    for batch_idx in range(batch_views):
        for target_idx in range(num_target_views):
            for input_idx in range(num_input_views):
                valid = valid_projection[batch_idx, target_idx, input_idx]
                if not valid.any():
                    continue
                coords = projected_coords[batch_idx, target_idx, input_idx]
                depth = depth_in_target[batch_idx, target_idx, input_idx]
                src_y, src_x = torch.nonzero(valid, as_tuple=True)
                px = coords[..., 0][valid].round().long().clamp(0, width - 1)
                py = coords[..., 1][valid].round().long().clamp(0, height - 1)
                cur_depth = depth[valid]
                old_depth = z_buffer[batch_idx, target_idx, py, px]
                update = cur_depth < old_depth
                if not update.any():
                    continue
                px = px[update]
                py = py[update]
                z_buffer[batch_idx, target_idx, py, px] = cur_depth[update]
                corr_map[batch_idx, target_idx, py, px, 0] = input_idx
                corr_map[batch_idx, target_idx, py, px, 1] = src_y[update]
                corr_map[batch_idx, target_idx, py, px, 2] = src_x[update]
                corr_mask[batch_idx, target_idx, py, px] = True
    return corr_map, corr_mask


class SpatialConsistencyLoss(nn.Module):
    """Cross-view cosine consistency loss for the decoupled spatial branch."""

    def __init__(
        self,
        temperature: float = 0.1,
        gamma: float = 1.0,
        alpha: float = 0.2,
        num_input_views: int = 3,
        min_valid_correspondences: int = 100,
    ):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma
        self.alpha = alpha
        self.num_input_views = num_input_views
        self.min_valid_correspondences = min_valid_correspondences

    def compute_loss(self, out: dict):
        spatial_features = out.get("spatial_features")
        if spatial_features is None:
            raise ValueError("spatial_features is required when use_spatial=True.")

        pointmap = out["gt_point"]["pts3d"]
        pointmap_mask = out["gt_point"]["valid_mask"]
        w2cs = out["w2cs"]
        intrinsics = out["intrinsics"]
        device = spatial_features.device
        zero_loss = (spatial_features * 0).mean()

        batch_views, num_total_views, feat_h, feat_w, _ = spatial_features.shape
        _, _, pm_h, pm_w, _ = pointmap.shape
        num_input_views = self.num_input_views
        num_target_views = num_total_views - num_input_views
        input_pointmap = pointmap[:, :num_input_views]
        input_mask = pointmap_mask[:, :num_input_views]
        target_w2cs = w2cs[:, num_input_views:]
        target_intrinsics = intrinsics[:, num_input_views:]

        if pm_h != feat_h or pm_w != feat_w:
            scale_h = feat_h / pm_h
            scale_w = feat_w / pm_w
            target_intrinsics = target_intrinsics.clone()
            target_intrinsics[:, :, 0, 0] *= scale_w
            target_intrinsics[:, :, 1, 1] *= scale_h
            target_intrinsics[:, :, 0, 2] *= scale_w
            target_intrinsics[:, :, 1, 2] *= scale_h

            pm = input_pointmap.permute(0, 1, 4, 2, 3).reshape(batch_views * num_input_views, 3, pm_h, pm_w)
            pm = F.interpolate(pm, size=(feat_h, feat_w), mode="bilinear", align_corners=False)
            input_pointmap = pm.reshape(batch_views, num_input_views, 3, feat_h, feat_w).permute(0, 1, 3, 4, 2)
            mask = input_mask.float().unsqueeze(2).reshape(batch_views * num_input_views, 1, pm_h, pm_w)
            mask = F.interpolate(mask, size=(feat_h, feat_w), mode="nearest")
            input_mask = mask.reshape(batch_views, num_input_views, feat_h, feat_w) > 0.5

        projected_coords, depth_in_target, valid_projection = project_points_to_target_view(
            input_pointmap,
            target_w2cs,
            target_intrinsics,
            input_mask,
        )
        if valid_projection.sum() < self.min_valid_correspondences:
            return zero_loss, {"loss_corr_feat": zero_loss, "num_correspondences": torch.tensor(0, device=device)}

        corr_map, corr_mask = build_correspondence_map(
            projected_coords,
            depth_in_target,
            valid_projection,
            feat_h,
            feat_w,
        )
        num_correspondences = corr_mask.sum()
        if num_correspondences < self.min_valid_correspondences:
            return zero_loss, {"loss_corr_feat": zero_loss, "num_correspondences": num_correspondences}

        input_features = spatial_features[:, :num_input_views]
        target_features = spatial_features[:, num_input_views:]
        losses = []
        for target_idx in range(num_target_views):
            valid_mask = corr_mask[:, target_idx]
            if not valid_mask.any():
                continue
            src_view_idx = corr_map[:, target_idx, :, :, 0]
            src_h_idx = corr_map[:, target_idx, :, :, 1]
            src_w_idx = corr_map[:, target_idx, :, :, 2]
            target_valid = target_features[:, target_idx][valid_mask]
            source_features = []
            for batch_idx in range(batch_views):
                batch_mask = valid_mask[batch_idx]
                if batch_mask.any():
                    source_features.append(input_features[batch_idx][src_view_idx[batch_idx][batch_mask], src_h_idx[batch_idx][batch_mask], src_w_idx[batch_idx][batch_mask]])
            if not source_features:
                continue
            source_valid = torch.cat(source_features, dim=0)
            feat_loss = 1.0 - (F.normalize(source_valid, dim=-1) * F.normalize(target_valid, dim=-1)).sum(dim=-1)
            losses.append(sanitize_loss(feat_loss, "spatial_corr_feat_loss").mean())

        if not losses:
            return zero_loss, {"loss_corr_feat": zero_loss, "num_correspondences": num_correspondences}
        loss = torch.stack(losses).mean()
        return loss, {"loss_corr_feat": loss, "num_correspondences": num_correspondences}
