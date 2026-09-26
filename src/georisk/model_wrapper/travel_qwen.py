import numpy as np
import torch
import os
from typing import Dict, List, Sequence
from ..llm.constants import DEFAULT_TRAJ_TOKEN

from .base_model import BaseModelWrapper
from .utils.travel_qwen_util import (
    load_model,
    prepare_data_to_inputs,
)


class GeoRiskNavigator(BaseModelWrapper):
    def __init__(self, model_args, data_args=None):
        self.tokenizer, self.model, self.image_processor = load_model(model_args)
        self.model_args = model_args
        self.data_args = data_args
        self.last_pred_stop_probs = None
        self._last_assist_notices = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        core_model = self._core_model()
        core_config = getattr(core_model, "config", None)
        self.traj_horizon = int(getattr(core_model, "traj_horizon", getattr(core_config, "traj_horizon", 10)))
        self.traj_execute_points = int(
            getattr(core_model, "traj_execute_points", getattr(core_config, "traj_execute_points", 5))
        )

    def _core_model(self):
        base_model = getattr(self.model, "base_model", None)
        if base_model is not None and hasattr(base_model, "model"):
            return base_model.model
        return self.model

    def _assert_instance_traj_token(self, instance: Dict) -> None:
        phase2_input_ids = instance.get("phase2_input_ids")
        if not torch.is_tensor(phase2_input_ids):
            return
        traj_token_id = int(self.tokenizer.convert_tokens_to_ids(DEFAULT_TRAJ_TOKEN))
        traj_count = int((phase2_input_ids == traj_token_id).sum().item())
        if traj_count < 1:
            raise ValueError("Phase2 eval template is missing required <traj> token.")

    def prepare_inputs(self, episodes, target_positions, assist_notices=None):
        instances = []
        rot_to_targets = []
        view_mode = getattr(self.data_args, "view_mode", "dual") if self.data_args is not None else "dual"
        for i in range(len(episodes)):
            instance, rot_to_target = prepare_data_to_inputs(
                episodes=episodes[i],
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                target_point=target_positions[i],
                assist_notice=assist_notices[i] if assist_notices is not None else None,
                view_mode=view_mode,
            )
            self._assert_instance_traj_token(instance)
            instances.append(instance)
            rot_to_targets.append(rot_to_target)

        self._last_assist_notices = list(assist_notices) if assist_notices is not None else None
        inputs = {"instances": instances}
        return inputs, rot_to_targets

    def _build_phase_batch(self, instances: List[Dict], phase: str, indices: Sequence[int]) -> Dict[str, torch.Tensor]:
        selected = [instances[int(i)] for i in indices]
        if len(selected) == 0:
            raise ValueError("No selected instances for phase batch.")

        key_input_ids = f"{phase}_input_ids"
        key_position_ids = f"{phase}_position_ids"
        key_attention_mask = f"{phase}_attention_mask"
        input_ids = [x[key_input_ids] for x in selected]
        position_ids = [x[key_position_ids] for x in selected]
        attention_masks = [x[key_attention_mask] for x in selected]

        
        max_len = max(x.shape[0] for x in input_ids)
        bs = len(input_ids)
        padded_input_ids = torch.full(
            (bs, max_len),
            int(self.tokenizer.pad_token_id),
            dtype=input_ids[0].dtype,
        )
        padded_attention = torch.zeros((bs, max_len), dtype=attention_masks[0].dtype)
        padded_positions = []
        for i, (ids, attn, pos) in enumerate(zip(input_ids, attention_masks, position_ids)):
            cur_len = int(ids.shape[0])
            padded_input_ids[i, max_len - cur_len :] = ids
            padded_attention[i, max_len - cur_len :] = attn
            
            padded_positions.append(torch.nn.functional.pad(pos, (max_len - pos.shape[2], 0), "constant", 1))
        input_ids = padded_input_ids[:, -self.tokenizer.model_max_length :]
        attention_mask = padded_attention[:, -self.tokenizer.model_max_length :]
        position_ids = torch.cat(padded_positions, dim=1)
        position_ids = position_ids[:, :, -self.tokenizer.model_max_length :]

        pixel_values = torch.cat([x["pixel_values"] for x in selected], dim=0)
        image_grid_thw = torch.cat([x["image_grid_thw"] for x in selected], dim=0)
        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "position_ids": position_ids.to(self.device),
            "pixel_values": pixel_values.to(self.device),
            "image_grid_thw": image_grid_thw.to(self.device),
        }

    def _phase1_trigger_mask(self, pred_stop_probs: np.ndarray, episodes) -> List[bool]:
        bs = int(len(pred_stop_probs))
        threshold = float(getattr(self, "stop_prob_threshold", 0.5))
        consecutive = max(1, int(getattr(self, "stop_prob_consecutive", 2)))
        streaks = getattr(self, "_stop_prob_streaks", None)
        if not isinstance(streaks, list) or len(streaks) != bs:
            streaks = [0 for _ in range(bs)]
        triggered = []
        for i in range(bs):
            prev = 0 if len(episodes[i]) <= 1 else int(streaks[i])
            p = float(pred_stop_probs[i])
            cond = bool((p == p) and np.isfinite(p) and p >= threshold)
            nxt = prev + 1 if cond else 0
            triggered.append(bool(nxt >= consecutive))
        return triggered

    def _phase2_generate_and_decode(self, phase2_inputs: Dict[str, torch.Tensor]) -> np.ndarray:
        decode_out = self.model(
            input_ids=phase2_inputs["input_ids"],
            attention_mask=phase2_inputs["attention_mask"],
            position_ids=phase2_inputs["position_ids"],
            pixel_values=phase2_inputs["pixel_values"],
            image_grid_thw=phase2_inputs["image_grid_thw"],
            return_trajectory=True,
            return_stop_prob=False,
            use_cache=False,
        )
        pred_trajectory_points = decode_out
        if not torch.is_tensor(pred_trajectory_points):
            raise TypeError(f"Unexpected LLM output type for trajectory_points: {type(pred_trajectory_points)}")
        if pred_trajectory_points.ndim != 3 or pred_trajectory_points.shape[1] != self.traj_horizon or pred_trajectory_points.shape[-1] != 3:
            raise ValueError(
                f"Unexpected trajectory output shape from LLM: {tuple(pred_trajectory_points.shape)}. "
                f"Expect [B, {self.traj_horizon}, 3]."
            )
        return pred_trajectory_points.detach().cpu().to(dtype=torch.float32).numpy()

    def run_llm_model(self, inputs, episodes):
        instances = list(inputs["instances"])
        all_indices = list(range(len(instances)))
        phase1_inputs = self._build_phase_batch(instances, phase="phase1", indices=all_indices)
        phase1_out = self.model(
            input_ids=phase1_inputs["input_ids"],
            attention_mask=phase1_inputs["attention_mask"],
            position_ids=phase1_inputs["position_ids"],
            pixel_values=phase1_inputs["pixel_values"],
            image_grid_thw=phase1_inputs["image_grid_thw"],
            return_trajectory=False,
            return_stop_prob=True,
            return_stop_only=True,
            use_cache=False,
        )
        if not torch.is_tensor(phase1_out):
            raise TypeError(f"Unexpected stop-only output type: {type(phase1_out)}")
        pred_stop_probs = phase1_out.view(-1).detach().cpu().to(dtype=torch.float32).numpy()
        self.last_pred_stop_probs = pred_stop_probs
        stop_triggered = self._phase1_trigger_mask(pred_stop_probs=pred_stop_probs, episodes=episodes)

        active_indices = [idx for idx in all_indices if not stop_triggered[idx]]
        all_trajectory_points_llm = np.zeros((len(instances), self.traj_horizon, 3), dtype=np.float32)
        if len(active_indices) > 0:
            phase2_inputs = self._build_phase_batch(instances, phase="phase2", indices=active_indices)
            active_trajectory_points = self._phase2_generate_and_decode(phase2_inputs)
            for j, idx in enumerate(active_indices):
                all_trajectory_points_llm[idx] = active_trajectory_points[j]

        debug_trajectory_point = os.environ.get("GEORISK_DEBUG_TRAJECTORY", "0").strip().lower() in {"1", "true", "yes", "on"}
        llm_sign = self._parse_axis_sign_from_env("GEORISK_LLM_SIGN", default=(1.0, 1.0, 1.0))
        trajectory_points_llm_new = all_trajectory_points_llm * llm_sign.reshape(1, 1, 3)
        if debug_trajectory_point and len(all_trajectory_points_llm) > 0:
            first_raw = all_trajectory_points_llm[0, 0]
            first_vec = trajectory_points_llm_new[0, 0]
            print(
                "[GEORISK_DEBUG_TRAJECTORY] raw_first=",
                np.round(first_raw, 6).tolist(),
                "vec_first=",
                np.round(first_vec, 6).tolist(),
                "llm_sign=",
                np.round(llm_sign, 4).tolist(),
                "pred_stop_prob_first=",
                None if self.last_pred_stop_probs is None else round(float(self.last_pred_stop_probs[0]), 6),
            )
        return np.asarray(trajectory_points_llm_new, dtype=np.float32), stop_triggered

    @staticmethod
    def _parse_axis_sign_from_env(env_name: str, default=(1.0, 1.0, 1.0)):
        raw = os.environ.get(env_name, "").strip()
        if not raw:
            return np.asarray(default, dtype=np.float32)
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 3:
            raise ValueError(f"{env_name} must be three comma-separated values, e.g. '1,1,1' or '-1,-1,-1'.")
        vals = []
        for p in parts:
            v = float(p)
            if abs(v) < 1e-8:
                raise ValueError(f"{env_name} contains zero value '{p}', each axis scale should be non-zero.")
            vals.append(v)
        return np.asarray(vals, dtype=np.float32)

    @staticmethod
    def _convert_target_frame_to_current_frame(episodes, trajectory_points_target_frame, rot_to_targets):
        targets_current_frame = []
        for i in range(len(episodes)):
            info = episodes[i]
            target = np.asarray(trajectory_points_target_frame[i], dtype=np.float32)
            if target.ndim == 1:
                target = target.reshape(1, 3)
            rot_to_target = None
            if rot_to_targets is not None and rot_to_targets[i] is not None:
                rot_to_target = np.asarray(rot_to_targets[i], dtype=np.float32)
            rot_0 = np.asarray(info[0]["sensors"]["imu"]["rotation"], dtype=np.float32)
            rot = np.asarray(info[-1]["sensors"]["imu"]["rotation"], dtype=np.float32)
            if rot_to_target is not None:
                target_current = target @ (rot_to_target.T @ rot_0.T @ rot)
            else:
                target_current = target @ (rot_0.T @ rot)
            targets_current_frame.append(target_current)
        return np.asarray(targets_current_frame, dtype=np.float32)

    def _build_world_path_from_current_vectors(self, episodes, vecs_current_frame, num_points: int = 5):
        paths = []
        for i in range(len(episodes)):
            ep = episodes[i]
            pos = np.asarray(ep[-1]["sensors"]["state"]["position"], dtype=np.float32)
            rot = np.asarray(ep[-1]["sensors"]["imu"]["rotation"], dtype=np.float32)
            world_step = rot @ np.asarray(vecs_current_frame[i], dtype=np.float32)
            path = np.stack([pos + world_step * ((k + 1) / float(num_points)) for k in range(num_points)], axis=0)
            paths.append(path)
        return paths

    def _build_world_paths_from_target_traj(self, episodes, traj_target_frame, rot_to_targets):
        traj_current = self._convert_target_frame_to_current_frame(episodes, traj_target_frame, rot_to_targets)
        execute_n = max(1, min(self.traj_execute_points, traj_current.shape[1]))
        paths = []
        for i in range(len(episodes)):
            ep = episodes[i]
            pos = np.asarray(ep[-1]["sensors"]["state"]["position"], dtype=np.float32)
            rot = np.asarray(ep[-1]["sensors"]["imu"]["rotation"], dtype=np.float32)
            cur_path = []
            for j in range(execute_n):
                cur_vec = np.asarray(traj_current[i, j], dtype=np.float32)
                cur_path.append(pos + rot @ cur_vec)
            paths.append(np.asarray(cur_path, dtype=np.float32))
        return paths

    def eval(self):
        self.model.eval()

    def run(self, inputs, episodes, rot_to_targets, target_positions=None):
        trajectory_points_llm_new, stop_triggered = self.run_llm_model(inputs, episodes)
        refined_trajectory_points = self._build_world_paths_from_target_traj(episodes, trajectory_points_llm_new, rot_to_targets)
        
        stop_indices = [i for i, v in enumerate(stop_triggered) if bool(v)]
        if len(stop_indices) > 0:
            zero_vecs = np.zeros((len(stop_indices), 3), dtype=np.float32)
            selected_eps = [episodes[i] for i in stop_indices]
            stationary = self._build_world_path_from_current_vectors(selected_eps, zero_vecs, num_points=5)
            for local_i, global_i in enumerate(stop_indices):
                refined_trajectory_points[global_i] = stationary[local_i]
        return refined_trajectory_points
