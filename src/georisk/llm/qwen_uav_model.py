from dataclasses import dataclass
import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception as exc:
    raise ImportError(
        "Qwen3VLForConditionalGeneration is unavailable. "
        "Please install a transformers build that supports Qwen3-VL (e.g. >=4.57.0.dev0)."
    ) from exc

from .constants import DEFAULT_STOP_TOKEN, DEFAULT_TRAJ_TOKEN


logger = logging.getLogger(__name__)


@dataclass
class GeoRiskOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    trajectory_loss: Optional[torch.FloatTensor] = None
    stop_loss: Optional[torch.FloatTensor] = None
    stop_rank_loss: Optional[torch.FloatTensor] = None
    clearance_loss: Optional[torch.FloatTensor] = None
    da3_joint_loss: Optional[torch.FloatTensor] = None
    predicted_trajectory: Optional[torch.FloatTensor] = None
    predicted_stop_probs: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


def resolve_uav_hidden_size(config) -> int:
    return int(config.text_config.hidden_size)


class DA3JointProjector(nn.Module):
    def __init__(self, input_dim: int = 2048, teacher_dim: int = 1024):
        super().__init__()
        self.input_dim = int(input_dim)
        self.teacher_dim = int(teacher_dim)
        self.projector = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.input_dim),
            nn.GELU(),
            nn.Linear(self.input_dim, self.teacher_dim),
        )

    def forward(self, llm_image_tokens: torch.Tensor) -> torch.Tensor:
        
        return F.normalize(self.projector(llm_image_tokens.float()), dim=-1, eps=1e-6)


class GeoRiskForNavigation(Qwen3VLForConditionalGeneration):
    def __init__(
        self,
        config,
        stop_loss_weight: float = 0.1,
        stop_rank_loss_weight: float = 0.0,
        stop_rank_margin: float = 0.1,
        stop_rank_min_gap: float = 2.0,
        traj_horizon: int = 10,
        traj_execute_points: int = 5,
        use_clearance_supervision: bool = True,
        clearance_loss_weight: float = 10.0,
        clearance_safe_margin_m: float = 2.0,
        clearance_voxel_size_m: float = 100.0 / 255.0,
        clearance_local_range_m: float = 30.0,
        clearance_depth_pool_size: int = 8,
        clearance_max_voxels: int = 2048,
        clearance_path_samples_per_segment: int = 4,
        clearance_temperature: float = 0.25,
        clearance_depth_max_m: float = 100.0,
        use_da3_joint_supervision: bool = True,
        da3_joint_weight: float = 0.5,
        da3_joint_teacher_dim: int = 1024,
        **kwargs,
    ):
        expected_project = "GeoRisk"
        checkpoint_project = getattr(config, "project", None)
        if checkpoint_project not in (None, expected_project):
            raise ValueError("Initialize from base Qwen or this exact DA3-weight-0.5 project.")
        if float(getattr(config, "da3_joint_weight", da3_joint_weight)) != 0.5:
            raise ValueError("This project's DA3 distillation weight must be 0.5.")
        super().__init__(config)
        hidden_size = resolve_uav_hidden_size(config)
        reduced_size = max(64, hidden_size // 2)
        self.stop_loss_weight = float(stop_loss_weight)
        self.stop_rank_loss_weight = max(0.0, float(stop_rank_loss_weight))
        self.stop_rank_margin = max(0.0, float(stop_rank_margin))
        self.stop_rank_min_gap = max(0.0, float(stop_rank_min_gap))
        self.traj_horizon = max(1, int(getattr(config, "traj_horizon", traj_horizon)))
        self.traj_execute_points = max(1, int(getattr(config, "traj_execute_points", traj_execute_points)))
        self.use_clearance_supervision = bool(
            getattr(config, "use_clearance_supervision", use_clearance_supervision)
        )
        self.clearance_loss_weight = float(getattr(config, "clearance_loss_weight", clearance_loss_weight))
        self.clearance_safe_margin_m = max(
            1e-3, float(getattr(config, "clearance_safe_margin_m", clearance_safe_margin_m))
        )
        self.clearance_voxel_size_m = max(
            1e-3, float(getattr(config, "clearance_voxel_size_m", clearance_voxel_size_m))
        )
        self.clearance_local_range_m = max(
            1e-3, float(getattr(config, "clearance_local_range_m", clearance_local_range_m))
        )
        self.clearance_depth_pool_size = max(
            1, int(getattr(config, "clearance_depth_pool_size", clearance_depth_pool_size))
        )
        self.clearance_max_voxels = max(1, int(getattr(config, "clearance_max_voxels", clearance_max_voxels)))
        self.clearance_path_samples_per_segment = max(
            1, int(getattr(config, "clearance_path_samples_per_segment", clearance_path_samples_per_segment))
        )
        self.clearance_temperature = max(1e-3, float(getattr(config, "clearance_temperature", clearance_temperature)))
        self.clearance_depth_max_m = max(
            1e-3, float(getattr(config, "clearance_depth_max_m", clearance_depth_max_m))
        )
        self.use_da3_joint_supervision = bool(
            getattr(config, "use_da3_joint_supervision", use_da3_joint_supervision)
        )
        self.da3_joint_weight = float(getattr(config, "da3_joint_weight", da3_joint_weight))
        self.da3_joint_teacher_dim = max(1, int(getattr(config, "da3_joint_teacher_dim", da3_joint_teacher_dim)))
        self.trajectory_token_embedding = nn.Embedding(1, hidden_size)
        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_size, reduced_size),
            nn.ReLU(),
            nn.Linear(reduced_size, reduced_size),
            nn.ReLU(),
            nn.Linear(reduced_size, 64),
        )
        self.trajectory_output = nn.Linear(64, self.traj_horizon * 3)
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_size, reduced_size),
            nn.ReLU(),
            nn.Linear(reduced_size, 1),
        )
        if self.use_da3_joint_supervision:
            self.da3_joint_projector = DA3JointProjector(input_dim=hidden_size, teacher_dim=self.da3_joint_teacher_dim)

        self.trajectory_loss_scale = 1.0
        self.special_token_dict = {}
        self._last_trajectory_loss = None
        self._last_stop_loss = None
        self._latest_loss_components = {}
        self._warned_nonfinite_loss = False
        if not hasattr(self, "rope_deltas"):
            self.rope_deltas = None
        
        self.config.stop_loss_weight = self.stop_loss_weight
        self.config.stop_rank_loss_weight = self.stop_rank_loss_weight
        self.config.stop_rank_margin = self.stop_rank_margin
        self.config.stop_rank_min_gap = self.stop_rank_min_gap
        self.config.traj_horizon = self.traj_horizon
        self.config.traj_execute_points = self.traj_execute_points
        self.config.use_clearance_supervision = self.use_clearance_supervision
        self.config.project = "GeoRisk"
        self.config.georisk_schema_version = 1
        self.config.trajectory_token = DEFAULT_TRAJ_TOKEN
        self.config.depth_reconstruction = False
        self.config.dpt_present = False
        self.config.geometry_teacher = "da3_large"
        self.config.clearance_loss_weight = self.clearance_loss_weight
        self.config.clearance_safe_margin_m = self.clearance_safe_margin_m
        self.config.clearance_voxel_size_m = self.clearance_voxel_size_m
        self.config.clearance_local_range_m = self.clearance_local_range_m
        self.config.clearance_depth_sampling = "local_min_argmin_v1"
        self.config.clearance_depth_pool_size = self.clearance_depth_pool_size
        self.config.clearance_depth_zero_is_valid = True
        self.config.clearance_max_voxels = self.clearance_max_voxels
        self.config.clearance_path_samples_per_segment = self.clearance_path_samples_per_segment
        self.config.clearance_temperature = self.clearance_temperature
        self.config.clearance_depth_max_m = self.clearance_depth_max_m
        self.config.use_da3_joint_supervision = self.use_da3_joint_supervision
        self.config.da3_joint_weight = self.da3_joint_weight
        self.config.da3_joint_teacher_dim = self.da3_joint_teacher_dim


    @staticmethod
    def _pack_visual_tokens_by_view(visual_tokens, token_counts, batch_size, view_count, name):
        chunks = []
        offset = 0
        max_tokens = max(token_counts) if token_counts else 0
        for count in token_counts:
            chunk = visual_tokens[offset:offset + count]
            offset += count
            if count < max_tokens:
                chunk = F.pad(chunk, (0, 0, 0, max_tokens - count))
            chunks.append(chunk)
        if len(chunks) != batch_size * view_count or offset != visual_tokens.shape[0]:
            raise ValueError(f"{name}: visual tokens do not match the front/down batch")
        stacked = torch.stack(chunks)
        return stacked.view(batch_size, view_count, max_tokens, stacked.shape[-1])

    def _extract_llm_image_tokens(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor],
        batch_size: int,
        view_count: int,
    ) -> torch.Tensor:
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required for DA3 joint feature supervision.")
        if input_ids is None:
            raise ValueError("input_ids is required for DA3 joint feature supervision.")
        if int(view_count) != 2:
            raise ValueError(
                f"DA3 joint feature supervision requires dual-view main inputs [front, down]; got view_count={view_count}."
            )
        image_grid_thw = image_grid_thw.to(device=hidden_states.device)
        merge_size = int(getattr(self.visual, "spatial_merge_size", 2))
        token_counts = [
            int(image_grid_thw[idx].prod().item() // max(1, merge_size * merge_size))
            for idx in range(int(image_grid_thw.shape[0]))
        ]
        expected_views = int(batch_size) * int(view_count)
        if len(token_counts) != expected_views:
            raise ValueError(
                f"DA3 joint feature supervision expects B*2 image grids, got {len(token_counts)} grids for "
                f"batch_size={batch_size}, view_count={view_count}."
            )
        image_token_id = int(getattr(self.config, "image_token_id"))
        image_mask = input_ids.to(device=hidden_states.device) == image_token_id
        expected_tokens = int(sum(token_counts))
        actual_tokens = int(image_mask.sum().item())
        if actual_tokens != expected_tokens:
            raise ValueError(
                f"LLM image-token count mismatch for DA3 joint supervision: input_ids tokens={actual_tokens}, "
                f"image_grid_thw tokens={expected_tokens}."
            )
        image_hidden = hidden_states[image_mask]
        return self._pack_visual_tokens_by_view(
            image_hidden,
            token_counts,
            batch_size=batch_size,
            view_count=view_count,
            name="da3_llm_image",
        )

    def _compute_da3_joint_loss(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor],
        batch_size: int,
        view_count: int,
        da3_joint_feat: torch.Tensor,
        da3_joint_valid_mask: torch.Tensor,
        da3_joint_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        llm_image_tokens = self._extract_llm_image_tokens(
            hidden_states=hidden_states,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            batch_size=batch_size,
            view_count=view_count,
        )
        student = self.da3_joint_projector(llm_image_tokens)
        teacher = da3_joint_feat.to(device=student.device, dtype=student.dtype)
        if teacher.ndim == 3:
            teacher = teacher.unsqueeze(1)
        if teacher.ndim != 4:
            raise ValueError(f"DA3 token teacher must have shape [B,V,L,D], got {tuple(teacher.shape)}")
        if teacher.shape[-1] != student.shape[-1]:
            raise ValueError(
                f"DA3 teacher dim mismatch: teacher={teacher.shape[-1]}, "
                f"projector={student.shape[-1]}. Set --da3_joint_teacher_dim to match cache."
            )
        if teacher.shape[1] != student.shape[1]:
            raise ValueError(
                f"DA3 teacher view count mismatch: teacher={teacher.shape[1]}, student={student.shape[1]}."
            )
        if da3_joint_token_mask is None:
            token_mask = torch.ones(
                teacher.shape[:-1],
                device=student.device,
                dtype=student.dtype,
            )
        else:
            token_mask = da3_joint_token_mask.to(device=student.device, dtype=student.dtype)
            if token_mask.ndim == 2:
                token_mask = token_mask.unsqueeze(1).expand(-1, teacher.shape[1], -1)
            if token_mask.shape[:3] != teacher.shape[:3]:
                raise ValueError(
                    f"DA3 token mask shape mismatch: mask={tuple(token_mask.shape)}, teacher={tuple(teacher.shape)}"
                )
        teacher, token_mask = self._pool_da3_teacher_to_student_grid(
            teacher_tokens=teacher,
            token_mask=token_mask,
            target_token_count=int(student.shape[2]),
        )
        teacher = F.normalize(teacher.float(), dim=-1, eps=1e-6).to(dtype=student.dtype)
        valid = da3_joint_valid_mask.to(device=student.device, dtype=student.dtype).view(-1, 1, 1)
        token_mask = token_mask.to(device=student.device, dtype=student.dtype) * valid
        per_token = ((student.float() - teacher.float().detach()) ** 2).sum(dim=-1)
        denom = token_mask.float().sum().clamp_min(1.0)
        return (per_token * token_mask.float()).sum() / denom

    @staticmethod
    def _infer_token_grid(token_count: int) -> Tuple[int, int]:
        token_count = max(1, int(token_count))
        root = int(round(math.sqrt(token_count)))
        if root * root == token_count:
            return root, root
        best_h = 1
        best_w = token_count
        best_gap = token_count
        for h in range(1, int(math.sqrt(token_count)) + 1):
            if token_count % h == 0:
                w = token_count // h
                gap = abs(w - h)
                if gap < best_gap:
                    best_h, best_w, best_gap = h, w, gap
        return int(best_h), int(best_w)

    def _pool_da3_teacher_to_student_grid(
        self,
        teacher_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        target_token_count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, views, src_tokens, channels = teacher_tokens.shape
        src_h, src_w = self._infer_token_grid(int(src_tokens))
        tgt_h, tgt_w = self._infer_token_grid(int(target_token_count))
        src_total = src_h * src_w
        if src_total != int(src_tokens):
            pad = src_total - int(src_tokens)
            teacher_tokens = F.pad(teacher_tokens, (0, 0, 0, max(0, pad)))
            token_mask = F.pad(token_mask, (0, max(0, pad)))
        teacher_grid = teacher_tokens.reshape(bsz * views, src_h, src_w, channels).permute(0, 3, 1, 2)
        mask_grid = token_mask.reshape(bsz * views, 1, src_h, src_w)
        weighted = teacher_grid * mask_grid
        pooled_mask = F.adaptive_avg_pool2d(mask_grid.float(), (tgt_h, tgt_w)).clamp_min(1e-6)
        pooled = F.adaptive_avg_pool2d(weighted.float(), (tgt_h, tgt_w)) / pooled_mask
        pooled_valid = (pooled_mask > 1e-5).to(dtype=teacher_tokens.dtype)
        pooled = pooled.permute(0, 2, 3, 1).reshape(bsz, views, tgt_h * tgt_w, channels)
        pooled_valid = pooled_valid.reshape(bsz, views, tgt_h * tgt_w)
        if pooled.shape[2] > int(target_token_count):
            pooled = pooled[:, :, : int(target_token_count), :]
            pooled_valid = pooled_valid[:, :, : int(target_token_count)]
        return pooled.to(dtype=teacher_tokens.dtype), pooled_valid.to(dtype=teacher_tokens.dtype)

    @staticmethod
    def _local_min_depth_samples(
        raw_depth: torch.Tensor,
        pool_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if raw_depth.ndim != 2:
            raise ValueError(f"Expected one depth view [H,W], got {tuple(raw_depth.shape)}")
        height, width = int(raw_depth.shape[0]), int(raw_depth.shape[1])
        pool_size = max(1, int(pool_size))
        if height % pool_size != 0 or width % pool_size != 0:
            raise ValueError(
                f"Depth size {(height, width)} must be divisible by clearance pool size {pool_size}."
            )

        out_h, out_w = height // pool_size, width // pool_size
        blocks = raw_depth.reshape(out_h, pool_size, out_w, pool_size).permute(0, 2, 1, 3)
        flat_blocks = blocks.reshape(out_h, out_w, pool_size * pool_size)
        min_depth, argmin = flat_blocks.min(dim=-1)
        block_v = torch.arange(out_h, device=raw_depth.device).view(out_h, 1) * pool_size
        block_u = torch.arange(out_w, device=raw_depth.device).view(1, out_w) * pool_size
        v_coords = block_v + torch.div(argmin, pool_size, rounding_mode="floor")
        u_coords = block_u + torch.remainder(argmin, pool_size)
        return min_depth, v_coords, u_coords

    def _unproject_depth_view_to_current(
        self,
        raw_depth: torch.Tensor,
        valid: torch.Tensor,
        view_to_current_pose: torch.Tensor,
        camera_slot: int,
    ) -> torch.Tensor:
        device = raw_depth.device
        dtype = torch.float32
        if float(valid.detach().float().item()) <= 0.5:
            return raw_depth.new_zeros((0, 3), dtype=dtype)

        depth_raw, vv, uu = self._local_min_depth_samples(
            raw_depth.to(device=device, dtype=dtype),
            self.clearance_depth_pool_size,
        )
        depth_m = depth_raw / 255.0 * float(self.clearance_depth_max_m)
        h_full, w_full = int(raw_depth.shape[-2]), int(raw_depth.shape[-1])
        vv = vv.to(device=device, dtype=dtype)
        uu = uu.to(device=device, dtype=dtype)

        valid_depth = torch.isfinite(depth_m) & (depth_m >= 0.0) & (depth_m <= float(self.clearance_local_range_m))
        if not bool(valid_depth.any().item()):
            return raw_depth.new_zeros((0, 3), dtype=dtype)

        f = torch.tensor(max((w_full - 1) / 2.0, 1.0), device=device, dtype=dtype)
        cx = torch.tensor((w_full - 1) / 2.0, device=device, dtype=dtype)
        cy = torch.tensor((h_full - 1) / 2.0, device=device, dtype=dtype)
        x = depth_m
        y = ((uu - cx) / f) * x
        z = ((vv - cy) / f) * x
        pts_cam = torch.stack([x, y, z], dim=-1)[valid_depth]

        if int(camera_slot) == 1:
            r_cam_to_body = torch.tensor(
                [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                device=device,
                dtype=dtype,
            )
            t_cam_to_body = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
        else:
            r_cam_to_body = torch.eye(3, device=device, dtype=dtype)
            t_cam_to_body = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        pts_body = (r_cam_to_body @ pts_cam.t()).t() + t_cam_to_body.view(1, 3)

        pose = view_to_current_pose.to(device=device, dtype=dtype).view(4, 4)
        ones = torch.ones((pts_body.shape[0], 1), device=device, dtype=dtype)
        pts_h = torch.cat([pts_body, ones], dim=-1)
        pts_current = (pose @ pts_h.t()).t()[:, :3]
        local_range = float(self.clearance_local_range_m)
        in_range = (
            torch.isfinite(pts_current).all(dim=-1)
            & (pts_current.abs() <= local_range).all(dim=-1)
            & (torch.linalg.norm(pts_current, dim=-1) <= local_range)
        )
        return pts_current[in_range]

    def _voxelize_obstacle_points(self, points: torch.Tensor) -> torch.Tensor:
        if points.numel() == 0:
            return points.reshape(0, 3)
        voxel_size = float(self.clearance_voxel_size_m)
        vox = torch.round(points / voxel_size).to(torch.int64)
        unique_vox = torch.unique(vox, dim=0)
        max_voxels = int(self.clearance_max_voxels)
        if unique_vox.shape[0] > max_voxels:
            keep = torch.linspace(
                0,
                unique_vox.shape[0] - 1,
                max_voxels,
                device=unique_vox.device,
            ).round().long()
            unique_vox = unique_vox[keep]
        return unique_vox.to(dtype=torch.float32) * voxel_size

    def _sample_predicted_path_current(
        self,
        predicted_trajectory: torch.Tensor,
        target_to_current_rot: torch.Tensor,
        traj_valid_mask: Optional[torch.Tensor],
        phase2_valid: torch.Tensor,
    ) -> torch.Tensor:
        device = predicted_trajectory.device
        pred_current = torch.matmul(
            predicted_trajectory.float(),
            target_to_current_rot.to(device=device, dtype=torch.float32),
        )
        n_exec = min(int(self.traj_execute_points), int(pred_current.shape[0]))
        if n_exec <= 0 or float(phase2_valid.detach().float().item()) <= 0.5:
            return predicted_trajectory.sum().view(1, 1) * 0.0
        if traj_valid_mask is None:
            valid = torch.ones((n_exec,), device=device, dtype=torch.bool)
        else:
            valid = traj_valid_mask[:n_exec].to(device=device).float() > 0.5
        if not bool(valid.any().item()):
            return predicted_trajectory.sum().view(1, 1) * 0.0

        samples: List[torch.Tensor] = []
        prev = torch.zeros((3,), device=device, dtype=torch.float32)
        alphas = torch.linspace(
            1.0 / float(self.clearance_path_samples_per_segment),
            1.0,
            int(self.clearance_path_samples_per_segment),
            device=device,
            dtype=torch.float32,
        )
        for idx in range(n_exec):
            cur = pred_current[idx]
            if bool(valid[idx].item()):
                seg = prev.view(1, 3) + (cur - prev).view(1, 3) * alphas.view(-1, 1)
                samples.append(seg)
            prev = cur
        if not samples:
            return predicted_trajectory.sum().view(1, 1) * 0.0
        return torch.cat(samples, dim=0)

    def _compute_clearance_loss(
        self,
        predicted_trajectory: torch.Tensor,
        clearance_depth_gt: torch.Tensor,
        clearance_depth_valid_mask: torch.Tensor,
        clearance_geom_valid_mask: torch.Tensor,
        clearance_view_to_current_poses: torch.Tensor,
        clearance_target_to_current_rot: torch.Tensor,
        traj_valid_mask: Optional[torch.Tensor],
        phase2_valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz = int(predicted_trajectory.shape[0])
        losses: List[torch.Tensor] = []
        for row in range(bsz):
            row_geom = float(clearance_geom_valid_mask[row].detach().float().item()) > 0.5
            if not row_geom:
                losses.append(predicted_trajectory[row].sum() * 0.0)
                continue

            view_points: List[torch.Tensor] = []
            view_count = int(clearance_depth_gt.shape[1])
            for view_idx in range(view_count):
                pts = self._unproject_depth_view_to_current(
                    raw_depth=clearance_depth_gt[row, view_idx],
                    valid=clearance_depth_valid_mask[row, view_idx],
                    view_to_current_pose=clearance_view_to_current_poses[row, view_idx],
                    camera_slot=view_idx % 2,
                )
                if pts.numel() > 0:
                    view_points.append(pts)
            if not view_points:
                losses.append(predicted_trajectory[row].sum() * 0.0)
                continue
            centers = self._voxelize_obstacle_points(torch.cat(view_points, dim=0))
            if centers.numel() == 0:
                losses.append(predicted_trajectory[row].sum() * 0.0)
                continue

            phase_valid = (
                phase2_valid_mask[row]
                if phase2_valid_mask is not None
                else torch.ones((), device=predicted_trajectory.device, dtype=predicted_trajectory.dtype)
            )
            row_traj_mask = traj_valid_mask[row] if traj_valid_mask is not None else None
            path_samples = self._sample_predicted_path_current(
                predicted_trajectory=predicted_trajectory[row],
                target_to_current_rot=clearance_target_to_current_rot[row],
                traj_valid_mask=row_traj_mask,
                phase2_valid=phase_valid,
            )
            if path_samples.ndim != 2 or path_samples.shape[-1] != 3 or path_samples.shape[0] == 0:
                losses.append(predicted_trajectory[row].sum() * 0.0)
                continue
            centers = centers.to(device=path_samples.device, dtype=torch.float32)
            nearest = torch.cdist(path_samples.float().unsqueeze(0), centers.unsqueeze(0)).squeeze(0).min(dim=1).values
            clearance = nearest - 0.5 * float(self.clearance_voxel_size_m)
            margin_gap = (float(self.clearance_safe_margin_m) - clearance) / float(self.clearance_temperature)
            penalty = F.softplus(margin_gap) * float(self.clearance_temperature)
            losses.append(penalty.mean() / float(self.clearance_safe_margin_m))
        if not losses:
            return predicted_trajectory.sum() * 0.0
        return torch.stack(losses).mean()

    @staticmethod
    def _zero_module_loss(*modules: nn.Module) -> torch.Tensor:
        total = None
        for module in modules:
            for param in module.parameters():
                term = param.sum() * 0.0
                total = term if total is None else total + term
        if total is None:
            return torch.tensor(0.0)
        return total

    def get_special_token_id(self, special_token_dict):
        if set(special_token_dict) != {DEFAULT_TRAJ_TOKEN, DEFAULT_STOP_TOKEN}:
            raise ValueError("GeoRisk requires exactly <traj> and <stop> navigation token IDs")
        self.special_token_dict = dict(special_token_dict)
        self.config.navigation_token_ids = dict(special_token_dict)

    def _extract_trajectory_states(self, hidden_states, input_ids, attention_mask):
        traj_token_id = self.special_token_dict.get(DEFAULT_TRAJ_TOKEN, None)
        if traj_token_id is None:
            raise ValueError("Special token id for <traj> is not set. Call get_special_token_id first.")

        trajectory_states = []
        for row in range(input_ids.shape[0]):
            matches = torch.nonzero(input_ids[row] == traj_token_id, as_tuple=False).flatten()
            if len(matches) != 1:
                raise ValueError(f"Sample {row} must have exactly one <traj> token")
            idx = int(matches[0].item())
            trajectory_states.append(hidden_states[row, idx, :])
        return torch.stack(trajectory_states, dim=0)

    def _extract_stop_states(self, hidden_states, input_ids):
        stop_token_id = self.special_token_dict.get(DEFAULT_STOP_TOKEN, None)
        if stop_token_id is None:
            raise ValueError("Special token id for <stop> is not set. Call get_special_token_id first.")

        stop_states = []
        for row in range(input_ids.shape[0]):
            matches = torch.nonzero(input_ids[row] == stop_token_id, as_tuple=False).flatten()
            if len(matches) != 1:
                raise ValueError(
                    f"Sample {row} has {len(matches)} <stop> tokens, expected exactly 1."
                )
            idx = int(matches[-1].item())
            stop_states.append(hidden_states[row, idx, :])
        return torch.stack(stop_states, dim=0)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        trajectory: Optional[torch.FloatTensor] = None,
        stop_labels: Optional[torch.FloatTensor] = None,
        stop_valid_mask: Optional[torch.FloatTensor] = None,
        stop_distance_to_goal: Optional[torch.FloatTensor] = None,
        phase2_valid_mask: Optional[torch.FloatTensor] = None,
        traj_valid_mask: Optional[torch.FloatTensor] = None,
        clearance_depth_gt: Optional[torch.Tensor] = None,
        clearance_depth_valid_mask: Optional[torch.Tensor] = None,
        clearance_geom_valid_mask: Optional[torch.Tensor] = None,
        clearance_view_to_current_poses: Optional[torch.Tensor] = None,
        clearance_target_to_current_rot: Optional[torch.Tensor] = None,
        da3_joint_feat: Optional[torch.Tensor] = None,
        da3_joint_valid_mask: Optional[torch.Tensor] = None,
        da3_joint_token_mask: Optional[torch.Tensor] = None,
        return_trajectory: Optional[bool] = False,
        return_stop_prob: Optional[bool] = False,
        return_stop_only: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple, GeoRiskOutput]:
        use_uav_heads = bool(
            return_trajectory
            or return_stop_prob
            or return_stop_only
            or trajectory is not None
            or stop_labels is not None
            or stop_valid_mask is not None
            or stop_distance_to_goal is not None
            or phase2_valid_mask is not None
            or traj_valid_mask is not None
            or clearance_depth_gt is not None
            or da3_joint_feat is not None
        )


        if not use_uav_heads:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                return_dict=return_dict,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                rope_deltas=rope_deltas,
                cache_position=cache_position,
                second_per_grid_ts=second_per_grid_ts,
                **kwargs,
            )

        if "return_dict" in kwargs:
            kwargs.pop("return_dict")
        self._last_trajectory_loss = None
        self._last_stop_loss = None
        self._latest_loss_components = {}
        
        if input_ids is None:
            raise ValueError("input_ids is required for trajectory_point extraction.")


        if rope_deltas is not None and hasattr(self, "model") and hasattr(self.model, "rope_deltas"):
            self.model.rope_deltas = rope_deltas

        if inputs_embeds is not None and input_ids is not None:
            model_input_ids = None
        else:
            model_input_ids = input_ids

        outputs = self.model(
            input_ids=model_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            use_cache=False if use_cache is None else use_cache,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
            **kwargs,
        )

        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = getattr(outputs, "last_hidden_state", None)
            if hidden_states is None:
                hidden_states = outputs[0]
            if hasattr(outputs, "rope_deltas") and outputs.rope_deltas is not None:
                self.rope_deltas = outputs.rope_deltas
        hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e4, neginf=-1e4)

        if return_stop_only:
            stop_states = self._extract_stop_states(hidden_states, input_ids).to(dtype=hidden_states.dtype)
            predicted_stop_logits = self.stop_head(stop_states).view(-1)
            predicted_stop_logits = torch.nan_to_num(predicted_stop_logits, nan=0.0, posinf=20.0, neginf=-20.0)
            predicted_stop_probs = torch.sigmoid(predicted_stop_logits)
            predicted_stop_probs = torch.nan_to_num(predicted_stop_probs, nan=0.0, posinf=1.0, neginf=0.0)
            return predicted_stop_probs

        trajectory_states = self._extract_trajectory_states(hidden_states, input_ids, attention_mask)
        trajectory_states = trajectory_states + self.trajectory_token_embedding.weight[0].to(dtype=trajectory_states.dtype)
        trajectory_states = torch.nan_to_num(trajectory_states, nan=0.0, posinf=1e4, neginf=-1e4)

        predicted_trajectory = self.trajectory_output(self.trajectory_head(trajectory_states))
        predicted_trajectory = predicted_trajectory.view(-1, self.traj_horizon, 3)
        predicted_trajectory = torch.nan_to_num(predicted_trajectory, nan=0.0, posinf=1e4, neginf=-1e4)
        stop_states = self._extract_stop_states(hidden_states, input_ids).to(dtype=predicted_trajectory.dtype)
        predicted_stop_logits = self.stop_head(stop_states).view(-1)
        predicted_stop_logits = torch.nan_to_num(predicted_stop_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        predicted_stop_probs = torch.sigmoid(predicted_stop_logits)
        predicted_stop_probs = torch.nan_to_num(predicted_stop_probs, nan=0.0, posinf=1.0, neginf=0.0)

        if trajectory is None:
            if return_trajectory:
                if return_stop_prob:
                    return predicted_trajectory, predicted_stop_probs
                return predicted_trajectory
            if return_stop_prob:
                return predicted_stop_probs
            return GeoRiskOutput(
                loss=None,
                trajectory_loss=None,
                stop_loss=None,
                stop_rank_loss=None,
                clearance_loss=None,
                da3_joint_loss=None,
                predicted_trajectory=predicted_trajectory,
                predicted_stop_probs=predicted_stop_probs,
                hidden_states=None,
            )

        trajectory = trajectory.to(device=predicted_trajectory.device, dtype=predicted_trajectory.dtype)
        if trajectory.ndim == 2 and trajectory.shape[-1] == self.traj_horizon * 3:
            trajectory = trajectory.view(-1, self.traj_horizon, 3)
        if trajectory.ndim != 3 or trajectory.shape[1] != self.traj_horizon or trajectory.shape[2] != 3:
            raise ValueError(
                f"Expected trajectory trajectory shape [B,{self.traj_horizon},3], got {tuple(trajectory.shape)}"
            )
        trajectory = torch.nan_to_num(trajectory, nan=0.0, posinf=1e4, neginf=-1e4)
        predicted_trajectory_f = predicted_trajectory.float()
        trajectory_points_f = trajectory.float()
        if phase2_valid_mask is None:
            phase2_mask_f = torch.ones(
                predicted_trajectory_f.shape[0],
                device=predicted_trajectory_f.device,
                dtype=predicted_trajectory_f.dtype,
            )
        else:
            phase2_mask_f = phase2_valid_mask.to(
                device=predicted_trajectory_f.device,
                dtype=predicted_trajectory_f.dtype,
            ).view(-1)
            phase2_mask_f = torch.clamp(phase2_mask_f, min=0.0, max=1.0)
        if traj_valid_mask is None:
            traj_mask_f = torch.ones(
                predicted_trajectory_f.shape[:2],
                device=predicted_trajectory_f.device,
                dtype=predicted_trajectory_f.dtype,
            )
        else:
            traj_mask_f = traj_valid_mask.to(
                device=predicted_trajectory_f.device,
                dtype=predicted_trajectory_f.dtype,
            )
            if traj_mask_f.ndim == 1:
                traj_mask_f = traj_mask_f.view(-1, self.traj_horizon)
            traj_mask_f = torch.clamp(traj_mask_f, min=0.0, max=1.0)
        loss_raw = F.smooth_l1_loss(
            predicted_trajectory_f,
            trajectory_points_f,
            reduction="none",
        )
        loss_mask = traj_mask_f.unsqueeze(-1) * phase2_mask_f.view(-1, 1, 1)
        loss_denom = loss_mask.float().sum() * 3.0
        if float(loss_denom.item()) > 0.0:
            trajectory_loss = self.trajectory_loss_scale * (loss_raw * loss_mask.float()).sum() / loss_denom
        else:
            trajectory_loss = predicted_trajectory_f.sum() * 0.0

        loss = trajectory_loss
        if not torch.isfinite(loss):
            if not self._warned_nonfinite_loss:
                logger.warning("Non-finite trajectory_point loss detected; applying nan_to_num safeguard.")
                self._warned_nonfinite_loss = True
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
        self._last_trajectory_loss = float(trajectory_loss.detach().item())

        stop_loss = None
        stop_rank_loss = None
        if stop_labels is not None:
            stop_target = stop_labels.to(
                device=predicted_stop_logits.device,
                dtype=predicted_stop_logits.dtype,
            ).view(-1)
            stop_target = torch.clamp(stop_target, min=0.0, max=1.0)
            if stop_valid_mask is None:
                valid_mask = torch.ones_like(stop_target)
            else:
                valid_mask = stop_valid_mask.to(
                    device=predicted_stop_logits.device,
                    dtype=predicted_stop_logits.dtype,
                ).view(-1)
                valid_mask = torch.clamp(valid_mask, min=0.0, max=1.0)
            stop_loss_per = F.binary_cross_entropy_with_logits(
                predicted_stop_logits.float(),
                stop_target.float(),
                reduction="none",
            )
            valid_denom = valid_mask.float().sum()
            if float(valid_denom.item()) > 0.0:
                stop_loss = (stop_loss_per * valid_mask.float()).sum() / valid_denom
            else:
                stop_loss = predicted_stop_logits.sum() * 0.0
            if (
                self.stop_rank_loss_weight > 0.0
                and stop_distance_to_goal is not None
            ):
                dist = stop_distance_to_goal.to(
                    device=predicted_stop_probs.device,
                    dtype=predicted_stop_probs.dtype,
                ).view(-1)
                valid_bool = valid_mask > 0.5
                if int(valid_bool.sum().item()) > 1:
                    
                    dist_delta = dist.unsqueeze(0) - dist.unsqueeze(1)  
                    pair_mask = valid_bool.unsqueeze(1) & valid_bool.unsqueeze(0)
                    pair_mask = pair_mask & (dist_delta >= float(self.stop_rank_min_gap))
                    if bool(pair_mask.any().item()):
                        p_row = predicted_stop_probs.unsqueeze(1)
                        p_col = predicted_stop_probs.unsqueeze(0)
                        rank_margin = float(self.stop_rank_margin) - (p_row - p_col)
                        rank_loss_matrix = F.relu(rank_margin)
                        stop_rank_loss = rank_loss_matrix[pair_mask].mean()
                    else:
                        stop_rank_loss = predicted_stop_logits.sum() * 0.0
                else:
                    stop_rank_loss = predicted_stop_logits.sum() * 0.0
            else:
                stop_rank_loss = predicted_stop_logits.sum() * 0.0
            if not torch.isfinite(stop_loss):
                if not self._warned_nonfinite_loss:
                    logger.warning("Non-finite stop loss detected; applying nan_to_num safeguard.")
                    self._warned_nonfinite_loss = True
                stop_loss = torch.nan_to_num(stop_loss, nan=0.0, posinf=1e4, neginf=-1e4)
            if stop_rank_loss is not None and (not torch.isfinite(stop_rank_loss)):
                if not self._warned_nonfinite_loss:
                    logger.warning("Non-finite stop rank loss detected; applying nan_to_num safeguard.")
                    self._warned_nonfinite_loss = True
                stop_rank_loss = torch.nan_to_num(stop_rank_loss, nan=0.0, posinf=1e4, neginf=-1e4)
            stop_total_loss = stop_loss
            if stop_rank_loss is not None:
                stop_total_loss = stop_total_loss + float(self.stop_rank_loss_weight) * stop_rank_loss
            loss = loss + self.stop_loss_weight * stop_total_loss
            self._last_stop_loss = float(stop_loss.detach().item())
        else:
            self._last_stop_loss = 0.0

        clearance_loss = None
        da3_joint_loss = None
        clearance_enabled = (
            self.use_clearance_supervision
            and self.training
            and clearance_depth_gt is not None
            and clearance_depth_valid_mask is not None
            and clearance_geom_valid_mask is not None
            and clearance_view_to_current_poses is not None
            and clearance_target_to_current_rot is not None
        )
        if clearance_enabled:
            clearance_loss = self._compute_clearance_loss(
                predicted_trajectory=predicted_trajectory_f,
                clearance_depth_gt=clearance_depth_gt.to(device=predicted_trajectory_f.device),
                clearance_depth_valid_mask=clearance_depth_valid_mask.to(device=predicted_trajectory_f.device),
                clearance_geom_valid_mask=clearance_geom_valid_mask.to(device=predicted_trajectory_f.device),
                clearance_view_to_current_poses=clearance_view_to_current_poses.to(device=predicted_trajectory_f.device),
                clearance_target_to_current_rot=clearance_target_to_current_rot.to(device=predicted_trajectory_f.device),
                traj_valid_mask=traj_mask_f,
                phase2_valid_mask=phase2_mask_f,
            )
            clearance_total = float(self.clearance_loss_weight) * clearance_loss
            if torch.isfinite(clearance_total):
                loss = loss + clearance_total.to(device=loss.device, dtype=loss.dtype)
            else:
                loss = loss + torch.nan_to_num(clearance_total, nan=0.0, posinf=1e4, neginf=-1e4).to(
                    device=loss.device,
                    dtype=loss.dtype,
                )

        da3_joint_enabled = (
            self.use_da3_joint_supervision
            and self.training
            and da3_joint_feat is not None
            and da3_joint_valid_mask is not None
            and image_grid_thw is not None
            and input_ids is not None
        )
        if da3_joint_enabled:
            da3_joint_loss = self._compute_da3_joint_loss(
                hidden_states=hidden_states,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                batch_size=int(input_ids.shape[0]),
                view_count=2,
                da3_joint_feat=da3_joint_feat,
                da3_joint_valid_mask=da3_joint_valid_mask,
                da3_joint_token_mask=da3_joint_token_mask,
            )
            da3_total = float(self.da3_joint_weight) * da3_joint_loss
            if torch.isfinite(da3_total):
                loss = loss + da3_total.to(device=loss.device, dtype=loss.dtype)
            else:
                loss = loss + torch.nan_to_num(da3_total, nan=0.0, posinf=1e4, neginf=-1e4).to(
                    device=loss.device,
                    dtype=loss.dtype,
                )
        elif self.use_da3_joint_supervision and self.training:
            loss = loss + self._zero_module_loss(self.da3_joint_projector).to(
                device=loss.device,
                dtype=loss.dtype,
            )

        if not torch.isfinite(loss):
            if not self._warned_nonfinite_loss:
                logger.warning("Non-finite total loss detected; applying nan_to_num safeguard.")
                self._warned_nonfinite_loss = True
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)

        self._latest_loss_components = {
            "trajectory_loss": float(trajectory_loss.detach().item()) if trajectory_loss is not None else None,
            "stop_loss": float(stop_loss.detach().item()) if stop_loss is not None else None,
            "stop_rank_loss": float(stop_rank_loss.detach().item()) if stop_rank_loss is not None else None,
            "clearance_loss": float(clearance_loss.detach().item()) if clearance_loss is not None else None,
            "da3_joint_loss": float(da3_joint_loss.detach().item()) if da3_joint_loss is not None else None,
        }

        if return_trajectory:
            if return_stop_prob:
                return loss, predicted_trajectory, predicted_stop_probs
            return loss, predicted_trajectory

        return GeoRiskOutput(
            loss=loss,
            trajectory_loss=trajectory_loss,
            stop_loss=stop_loss,
            stop_rank_loss=stop_rank_loss,
            clearance_loss=clearance_loss,
            da3_joint_loss=da3_joint_loss,
            predicted_trajectory=predicted_trajectory,
            predicted_stop_probs=predicted_stop_probs,
            hidden_states=None,
        )
