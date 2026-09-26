import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer
from .online_rgb import airsim_bgr_to_model_rgb
from ...llm.checkpoint import read_checkpoint_config, validate_tokenizer, load_non_lora_weights, load_special_embeddings

from ...llm.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_STOP_TOKEN, DEFAULT_TRAJ_TOKEN
from ...llm.geometry import (
    rotation_matrix_from_vector,
    transform_point,
)
from ...llm.preprocess_qwen import preprocess_qwen_visual
from ...llm.qwen_uav_model import GeoRiskForNavigation
from ...llm.rope2d import get_rope_index_3
from ...llm.prompts import observation_prompt, trajectory_prompt, STOP_REPLY, TRAJECTORY_REPLY


RGB_CAMERA_ORDER = ["frontcamera", "leftcamera", "rightcamera", "rearcamera", "downcamera"]
VIEW_MODE_TO_CAMERAS = {
    "dual": ["frontcamera", "downcamera"],
}
CAMERA_NAME_TO_INDEX = {name: idx for idx, name in enumerate(RGB_CAMERA_ORDER)}
_EVAL_TEXT_INSTANCE_CACHE: "OrderedDict[tuple, Dict[str, torch.Tensor]]" = OrderedDict()
_EVAL_TEXT_INSTANCE_CACHE_SIZE = max(0, min(1024, int(os.environ.get("GEORISK_EVAL_TEXT_CACHE_SIZE", "512"))))
_EPISODE_INSTRUCTION_CACHE: "OrderedDict[int, str]" = OrderedDict()
_EPISODE_INSTRUCTION_CACHE_SIZE = max(0, min(1024, int(os.environ.get("GEORISK_EPISODE_INSTR_CACHE_SIZE", "1024"))))


def _resolve_view_mode_indices(view_mode: str) -> List[int]:
    mode = (view_mode or "dual").strip().lower()
    if mode not in VIEW_MODE_TO_CAMERAS:
        raise ValueError(
            f"Unsupported view_mode '{mode}'. Use one of: {sorted(VIEW_MODE_TO_CAMERAS.keys())}"
        )
    return [CAMERA_NAME_TO_INDEX[name] for name in VIEW_MODE_TO_CAMERAS[mode]]


def load_model(args):
    model_path = os.path.expanduser(args.model_path)
    if not args.model_base:
        raise ValueError("--model_base must point to Qwen3-VL-2B-Instruct")
    model_base = os.path.expanduser(args.model_base)
    custom_cfg = read_checkpoint_config(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side="right")
    token_ids = validate_tokenizer(tokenizer, custom_cfg)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor = AutoProcessor.from_pretrained(model_path)
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    attn_impl = os.environ.get("GEORISK_ATTN_IMPLEMENTATION", "flash_attention_2" if torch.cuda.is_available() else "sdpa")
    model = GeoRiskForNavigation.from_pretrained(
        model_base,
        dtype=torch_dtype,
        attn_implementation=attn_impl,
        traj_horizon=custom_cfg["traj_horizon"],
        traj_execute_points=custom_cfg["traj_execute_points"],
        use_clearance_supervision=False,
        use_da3_joint_supervision=False,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.get_special_token_id(token_ids)
    load_special_embeddings(model, model_path)
    model = PeftModel.from_pretrained(model, model_path)
    load_non_lora_weights(model, model_path, inference=True)
    model.eval()
    return tokenizer, model, processor.image_processor


def _extract_instruction(episode_steps: Sequence[Dict]) -> str:
    
    for step in reversed(list(episode_steps)):
        instruction = step.get("instruction", None)
        if isinstance(instruction, str) and instruction.strip():
            return instruction
    
    for step in reversed(list(episode_steps)):
        raw_info = step.get("raw_trajectory_info", None)
        if not isinstance(raw_info, dict):
            continue
        instruction = raw_info.get("instruction", None)
        if isinstance(instruction, str) and instruction.strip():
            return instruction
    raise KeyError("No valid `instruction` text found in episode steps.")


def _extract_latest_two_rgb_steps(episode_steps: Sequence[Dict]):
    latest = None
    prev = None
    for step in reversed(episode_steps):
        if "rgb" not in step:
            continue
        if latest is None:
            latest = step
        else:
            prev = step
            break
    if latest is None:
        raise ValueError("No rgb frame found in episode.")
    return latest, prev


def _extract_instruction_cached(episode_steps: Sequence[Dict]) -> str:
    cache_key = int(id(episode_steps))
    cached = _EPISODE_INSTRUCTION_CACHE.get(cache_key, None)

    tail_instruction = None
    if len(episode_steps) > 0:
        tail = episode_steps[-1]
        candidate = tail.get("instruction", None)
        if isinstance(candidate, str) and candidate.strip():
            tail_instruction = candidate.strip()
        elif isinstance(tail.get("raw_trajectory_info", None), dict):
            candidate = tail["raw_trajectory_info"].get("instruction", None)
            if isinstance(candidate, str) and candidate.strip():
                tail_instruction = candidate.strip()

    if tail_instruction is not None:
        if (cached is None) or (tail_instruction != cached):
            _EPISODE_INSTRUCTION_CACHE[cache_key] = tail_instruction
            _EPISODE_INSTRUCTION_CACHE.move_to_end(cache_key)
            while len(_EPISODE_INSTRUCTION_CACHE) > _EPISODE_INSTRUCTION_CACHE_SIZE:
                _EPISODE_INSTRUCTION_CACHE.popitem(last=False)
            return tail_instruction
        _EPISODE_INSTRUCTION_CACHE.move_to_end(cache_key)
        return cached

    if cached is not None:
        _EPISODE_INSTRUCTION_CACHE.move_to_end(cache_key)
        return cached

    instruction = _extract_instruction(episode_steps).strip()
    _EPISODE_INSTRUCTION_CACHE[cache_key] = instruction
    _EPISODE_INSTRUCTION_CACHE.move_to_end(cache_key)
    while len(_EPISODE_INSTRUCTION_CACHE) > _EPISODE_INSTRUCTION_CACHE_SIZE:
        _EPISODE_INSTRUCTION_CACHE.popitem(last=False)
    return instruction


def _mask_eval_labels(
    data_dict: Dict[str, torch.Tensor],
    stop_token_id: int,
    traj_token_id: int,
):
    
    if stop_token_id is not None and int(stop_token_id) >= 0:
        stop_mask = data_dict["input_ids"] == int(stop_token_id)
        data_dict["labels"][stop_mask] = -100
    
    if stop_token_id is not None and int(stop_token_id) >= 0 and traj_token_id is not None and int(traj_token_id) >= 0:
        input_ids = data_dict["input_ids"]
        labels = data_dict["labels"]
        for row in range(input_ids.shape[0]):
            stop_matches = torch.nonzero(input_ids[row] == int(stop_token_id), as_tuple=False).flatten()
            wp_matches = torch.nonzero(input_ids[row] == int(traj_token_id), as_tuple=False).flatten()
            if len(stop_matches) == 0 or len(wp_matches) == 0:
                continue
            start = int(stop_matches[0].item()) + 1
            end = int(wp_matches[-1].item())
            if start < end:
                labels[row, start:end] = -100


def _build_eval_text_instance(
    source: List[Dict[str, str]],
    tokenizer,
    image_grid_thw: torch.Tensor,
    merged_grid_tokens: Sequence[int],
    merge_size: int,
    stop_token_id: int,
    traj_token_id: int,
) -> Dict[str, torch.Tensor]:
    source_key = tuple((str(x.get("from", x.get("role", ""))), str(x.get("value", x.get("content", "")))) for x in source)
    cache_key = (
        int(id(tokenizer)),
        tuple(int(x) for x in merged_grid_tokens),
        int(merge_size),
        int(stop_token_id) if stop_token_id is not None else -1,
        int(traj_token_id) if traj_token_id is not None else -1,
        source_key,
    )
    cached = _EVAL_TEXT_INSTANCE_CACHE.get(cache_key, None)
    if cached is not None:
        _EVAL_TEXT_INSTANCE_CACHE.move_to_end(cache_key)
        return {k: v.clone() for k, v in cached.items()}

    data_dict = preprocess_qwen_visual([source], tokenizer, grid_thw_image=merged_grid_tokens)
    _mask_eval_labels(
        data_dict,
        stop_token_id=stop_token_id,
        traj_token_id=traj_token_id,
    )
    attention_mask = torch.ones_like(data_dict["input_ids"])
    position_ids, _ = get_rope_index_3(
        spatial_merge_size=merge_size,
        input_ids=data_dict["input_ids"],
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    instance = {
        "input_ids": data_dict["input_ids"][0],
        "labels": data_dict["labels"][0],
        "position_ids": position_ids[:, 0:1, :],
        "attention_mask": attention_mask[0],
    }
    _EVAL_TEXT_INSTANCE_CACHE[cache_key] = {k: v.clone() for k, v in instance.items()}
    _EVAL_TEXT_INSTANCE_CACHE.move_to_end(cache_key)
    while len(_EVAL_TEXT_INSTANCE_CACHE) > _EVAL_TEXT_INSTANCE_CACHE_SIZE:
        _EVAL_TEXT_INSTANCE_CACHE.popitem(last=False)
    return instance


def prepare_data_to_inputs(
    episodes,
    tokenizer,
    image_processor,
    target_point,
    assist_notice=None,
    view_mode="dual",
):
    ori_sources = episodes
    latest_rgb_source, prev_rgb_source = _extract_latest_two_rgb_steps(ori_sources)
    images = []
    selected_camera_indices = _resolve_view_mode_indices(view_mode)
    rgb_views = latest_rgb_source["rgb"]
    for cam_idx in selected_camera_indices:
        if cam_idx >= len(rgb_views):
            raise ValueError(
                f"Episode rgb views has {len(rgb_views)} items, cannot access camera index {cam_idx} "
                f"for view_mode={view_mode}."
            )
        images.append(airsim_bgr_to_model_rgb(rgb_views[cam_idx]))

    rot = np.asarray(ori_sources[0]["sensors"]["imu"]["rotation"])
    pos = np.asarray(ori_sources[0]["sensors"]["state"]["position"])
    target_point = np.asarray(rot.T @ (target_point - pos), dtype=np.float32)
    rotation_to_target = rotation_matrix_from_vector(float(target_point[0]), float(target_point[1]))

    latest_pos = np.asarray(latest_rgb_source["sensors"]["state"]["position"], dtype=np.float32)
    if prev_rgb_source is not None:
        prev_pos = np.asarray(prev_rgb_source["sensors"]["state"]["position"])
        delta = np.asarray(rot.T @ (latest_pos - prev_pos), dtype=np.float32)
        delta = transform_point(delta, rotation_to_target)
    else:
        delta = np.asarray([0.0, 0.0, -4.5], dtype=np.float32)
    delta = delta / (np.linalg.norm(delta) + 1e-8)
    delta_str = ",".join([str(round(float(x), 1)) for x in delta])
    cur_pos = np.asarray(rot.T @ (latest_pos - pos), dtype=np.float32)
    cur_pos = transform_point(cur_pos, rotation_to_target)
    cur_pos_str = ",".join([str(round(float(x), 1)) for x in cur_pos])

    stage = assist_notice if assist_notice is not None else ("cruise" if len(ori_sources) > 20 else "take off")
    stage = str(stage).strip() if stage is not None else "cruise"
    if stage == "":
        stage = "cruise"
    instruction = _extract_instruction_cached(ori_sources).replace(DEFAULT_IMAGE_TOKEN, "").strip()
    round1_user = observation_prompt(instruction=instruction, image_num=len(images))
    round2_user = trajectory_prompt(stage=stage, delta=delta_str, cur_pos=cur_pos_str)
    phase1_source = [
        {"from": "human", "value": round1_user},
        {"from": "gpt", "value": STOP_REPLY},
    ]
    phase2_source = [
        {"from": "human", "value": round1_user},
        {"from": "gpt", "value": f"Stop: {DEFAULT_STOP_TOKEN}"},
        {"from": "human", "value": round2_user},
        {"from": "gpt", "value": TRAJECTORY_REPLY},
    ]

    merge_size = image_processor.merge_size
    safe_images: List[np.ndarray] = []
    for image in images:
        safe_images.append(np.array(image, copy=True, order="C"))
    processed = image_processor.preprocess(safe_images, return_tensors="pt")
    pixel_values = processed["pixel_values"]
    image_grid_thw = processed["image_grid_thw"]
    if isinstance(pixel_values, list):
        tensor_list = []
        for value in pixel_values:
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            tensor_list.append(value if value.ndim == 2 else value.reshape(-1, value.shape[-1]))
        pixel_values = torch.cat(tensor_list, dim=0)
    elif not isinstance(pixel_values, torch.Tensor):
        pixel_values = torch.as_tensor(pixel_values)
    if isinstance(image_grid_thw, list):
        image_grid_thw = torch.as_tensor(image_grid_thw)
    elif not isinstance(image_grid_thw, torch.Tensor):
        image_grid_thw = torch.as_tensor(image_grid_thw)
    if image_grid_thw.ndim == 1:
        image_grid_thw = image_grid_thw.unsqueeze(0)
    merged_grid_tokens = [
        int(image_grid_thw[idx].prod().item() // (merge_size**2))
        for idx in range(image_grid_thw.shape[0])
    ]

    stop_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_STOP_TOKEN)
    traj_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_TRAJ_TOKEN)
    phase1_instance = _build_eval_text_instance(
        source=phase1_source,
        tokenizer=tokenizer,
        image_grid_thw=image_grid_thw,
        merged_grid_tokens=merged_grid_tokens,
        merge_size=merge_size,
        stop_token_id=stop_token_id,
        traj_token_id=traj_token_id,
    )
    phase2_instance = _build_eval_text_instance(
        source=phase2_source,
        tokenizer=tokenizer,
        image_grid_thw=image_grid_thw,
        merged_grid_tokens=merged_grid_tokens,
        merge_size=merge_size,
        stop_token_id=stop_token_id,
        traj_token_id=traj_token_id,
    )

    return {
        "phase1_input_ids": phase1_instance["input_ids"],
        "phase1_labels": phase1_instance["labels"],
        "phase1_position_ids": phase1_instance["position_ids"],
        "phase1_attention_mask": phase1_instance["attention_mask"],
        "phase2_input_ids": phase2_instance["input_ids"],
        "phase2_labels": phase2_instance["labels"],
        "phase2_position_ids": phase2_instance["position_ids"],
        "phase2_attention_mask": phase2_instance["attention_mask"],
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "prompt": round1_user,
        "stage": stage,
        "assistant_phase2_prefix": TRAJECTORY_REPLY,
        "assistant_expected_suffix": f"{DEFAULT_TRAJ_TOKEN}",
    }, rotation_to_target
