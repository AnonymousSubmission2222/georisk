import logging
import json
import os
import pathlib
import inspect
import time
import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import AutoProcessor, AutoTokenizer, Trainer, TrainerCallback
from transformers.trainer_callback import PrinterCallback

from .collator import GeoRiskCollator
from .constants import DEFAULT_STOP_TOKEN, DEFAULT_TRAJ_TOKEN
from .dataset_uav import GeoRiskDataset
from .qwen_uav_model import GeoRiskForNavigation, resolve_uav_hidden_size
from .checkpoint import read_checkpoint_config, save_special_embeddings, load_special_embeddings, load_non_lora_weights
from ..paths import ASSETS_ROOT


logger = logging.getLogger(__name__)


def _format_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _resolve_core_model(model):
    while model is not None and hasattr(model, "module"):
        model = model.module
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "model"):
        return base_model.model
    return model


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen3-VL-2B-Instruct")
    attn_implementation: Optional[str] = field(default="flash_attention_2")
    
    use_qlora: bool = field(default=False)
    load_in_4bit: bool = field(default=False)
    bnb_4bit_quant_type: str = field(default="nf4")  
    bnb_4bit_compute_dtype: str = field(default="bfloat16")  
    bnb_4bit_use_double_quant: bool = field(default=True)
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=True)
    tune_mm_mlp_with_lora: bool = field(default=False)
    tune_mm_vision: bool = field(default=True)
    tune_trajectory_head: bool = field(default=True)
    stop_loss_weight: float = field(default=0.1)
    stop_rank_loss_weight: float = field(default=0.0)
    stop_rank_margin: float = field(default=0.1)
    stop_rank_min_gap: float = field(default=2.0)
    traj_horizon: int = field(default=10)
    traj_execute_points: int = field(default=5)
    use_clearance_supervision: bool = field(default=True)
    clearance_loss_weight: float = field(default=10.0)
    clearance_safe_margin_m: float = field(default=2.0)
    clearance_depth_max_m: float = field(default=100.0)
    clearance_voxel_size_m: float = field(default=100.0 / 255.0)
    clearance_local_range_m: float = field(default=30.0)
    clearance_depth_pool_size: int = field(default=8)
    clearance_max_voxels: int = field(default=2048)
    clearance_path_samples_per_segment: int = field(default=4)
    clearance_temperature: float = field(default=0.25)
    use_da3_joint_supervision: bool = field(default=True)
    da3_joint_weight: float = field(default=0.5)
    da3_joint_teacher_dim: int = field(default=1024)
    visual_lora_block_indices: str = field(default="5,11,17,23")
    lora_enable: bool = field(default=True)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    merger_lora_r: int = field(default=8)
    merger_lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")
    lora_target_modules: str = field(default="auto")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to TravelUAV uav_dataset train json."})
    dataset_path: str = field(default=None, metadata={"help": "Path to decompressed TravelUAV dataset root."})
    dense_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to TravelUAV_original_decompressed_merged_all containing merged_data_all.json."},
    )
    max_samples: Optional[int] = field(default=None)
    view_mode: str = field(
        default="dual",
        metadata={"help": "GeoRisk uses dual front/down views."},
    )
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    stop_terminal_repeat: int = field(default=1)
    stop_soft_r: float = field(default=20.0)
    stop_soft_tau: float = field(default=5.0)
    stop_label_clip_eps: float = field(default=1e-4)
    stop_phase2_disable_dist: Optional[float] = field(default=None)
    stop_near_resample_rules: str = field(default="0,5,2;5,10,2;10,20,2")
    depth_dataset_path: Optional[str] = field(
        default=str(ASSETS_ROOT / "TravelUAV_depth_trainset"),
        metadata={"help": "Sidecar root containing map/uuid/depth_imgs_uint8.npz for clearance supervision."},
    )
    clearance_bad_depth_maps: str = field(default="BrushifyCountryRoads,NordicHarbour")
    da3_joint_teacher_path: Optional[str] = field(
        default=str(ASSETS_ROOT / "TravelUAV_da3_large_joint_teacher"),
        metadata={"help": "Offline DA3-LARGE joint feature cache root: map/uuid/da3_joint_feats.npz."},
    )
    strict_da3_joint_cache: bool = field(
        default=True,
        metadata={"help": "Raise when a requested DA3 joint teacher feature is missing instead of masking it out."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch_fused")
    remove_unused_columns: bool = field(default=False)
    non_lora_trainable_learning_rate: Optional[float] = field(
        default=None,
        metadata={
            "help": "Optional LR for trainable non-LoRA params (e.g., projector/trajectory_point heads). "
            "If unset, all trainable params use --learning_rate."
        },
    )
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length."},
    )
    head_warmup_enable: bool = field(
        default=True,
        metadata={"help": "Run a pre-training stage that freezes LLM and trains UAV heads only."},
    )
    head_warmup_steps: int = field(
        default=5,
        metadata={"help": "Number of optimizer steps for pre-training head-only warmup."},
    )
    head_warmup_learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate used in pre-training head-only warmup stage."},
    )


class UAVTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._grad_health_reported = False
        self._nonfinite_warned = False
        self._aux_log_sums = {}
        self._aux_log_counts = {}
        
        self._stop_fn_count_sum = 0.0
        self._stop_fn_pos_count_sum = 0.0

    def _is_main_process(self) -> bool:
        if self.args.local_rank in (-1, 0):
            return True
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return False

    def _report_lora_grad_health_once(self):
        if self._grad_health_reported:
            return
        if not self._is_main_process():
            self._grad_health_reported = True
            return

        has_trainable_lora = False
        has_grad = False
        for name, p in self.model.named_parameters():
            if "lora_" not in name:
                continue
            if not p.requires_grad:
                continue
            has_trainable_lora = True
            has_grad = has_grad or p.grad is not None

        if has_trainable_lora and not has_grad:
            logger.warning(
                "Trainable adapters have no gradients. "
                "Please verify gradient checkpointing / requires_grad settings."
            )
        self._grad_health_reported = True

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        non_lora_lr = getattr(self.args, "non_lora_trainable_learning_rate", None)
        if non_lora_lr is None:
            return super().create_optimizer()

        lora_lr = float(self.args.learning_rate)
        non_lora_lr = float(non_lora_lr)

        lora_params = []
        non_lora_params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_" in name:
                lora_params.append(p)
            else:
                non_lora_params.append(p)

        if len(lora_params) == 0 or len(non_lora_params) == 0:
            if self.is_world_process_zero():
                logger.warning(
                    "Differential LR requested but parameter split is degenerate. "
                    "Falling back to Trainer default optimizer."
                )
            return super().create_optimizer()

        try:
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)
        except TypeError:
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        optimizer_kwargs = dict(optimizer_kwargs)
        optimizer_kwargs.pop("lr", None)

        optim_groups = [
            {"params": lora_params, "lr": lora_lr, "weight_decay": float(self.args.weight_decay)},
            {"params": non_lora_params, "lr": non_lora_lr, "weight_decay": float(self.args.weight_decay)},
        ]
        self.optimizer = optimizer_cls(optim_groups, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model_inputs = dict(inputs)
        outputs = model(**model_inputs)
        if isinstance(outputs, dict):
            loss = outputs.get("loss", None)
        else:
            loss = getattr(outputs, "loss", None)
        if loss is None:
            if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                loss = outputs[0]
            else:
                raise ValueError("Model did not return a loss.")

        aux_logs = {}

        trajectory_loss = outputs.get("trajectory_loss", None) if isinstance(outputs, dict) else getattr(outputs, "trajectory_loss", None)
        stop_loss = outputs.get("stop_loss", None) if isinstance(outputs, dict) else getattr(outputs, "stop_loss", None)
        stop_rank_loss = (
            outputs.get("stop_rank_loss", None) if isinstance(outputs, dict) else getattr(outputs, "stop_rank_loss", None)
        )
        clearance_loss = (
            outputs.get("clearance_loss", None)
            if isinstance(outputs, dict)
            else getattr(outputs, "clearance_loss", None)
        )
        da3_joint_loss = (
            outputs.get("da3_joint_loss", None) if isinstance(outputs, dict) else getattr(outputs, "da3_joint_loss", None)
        )
        predicted_stop_probs = (
            outputs.get("predicted_stop_probs", None)
            if isinstance(outputs, dict)
            else getattr(outputs, "predicted_stop_probs", None)
        )

        if trajectory_loss is not None:
            try:
                aux_logs["trajectory_loss"] = float(trajectory_loss.detach().mean().cpu().item())
            except Exception:
                pass
        if stop_loss is not None:
            try:
                aux_logs["stop_loss"] = float(stop_loss.detach().mean().cpu().item())
            except Exception:
                pass
        if stop_rank_loss is not None:
            try:
                aux_logs["stop_rank_loss"] = float(stop_rank_loss.detach().mean().cpu().item())
            except Exception:
                pass
        for key, value in [
            ("clearance_loss", clearance_loss),
            ("da3_joint_raw_loss", da3_joint_loss),
        ]:
            if value is not None:
                try:
                    aux_logs[key] = float(value.detach().mean().cpu().item())
                except Exception:
                    pass
        if da3_joint_loss is not None:
            try:
                core_model = _resolve_core_model(model)
                weight = float(getattr(core_model, "da3_joint_weight", 0.5))
                aux_logs["da3_joint_weighted_loss"] = float(
                    da3_joint_loss.detach().mean().cpu().item() * weight
                )
            except Exception:
                pass
        da3_valid = model_inputs.get("da3_joint_valid_mask", None)
        if torch.is_tensor(da3_valid):
            aux_logs["da3_joint_valid_rate"] = float(da3_valid.detach().float().mean().cpu().item())
        
        
        try:
            stop_labels = model_inputs.get("stop_labels", None)
            stop_valid_mask = model_inputs.get("stop_valid_mask", None)
            stop_dists = model_inputs.get("stop_distance_to_goal", None)
            phase2_valid_mask = model_inputs.get("phase2_valid_mask", None)
            if torch.is_tensor(stop_labels):
                lbl = stop_labels.detach().float()
                if torch.is_tensor(stop_valid_mask):
                    v = stop_valid_mask.detach().float().clamp(min=0.0, max=1.0)
                    denom = float(v.sum().item())
                    if denom > 0.0:
                        aux_logs["stop_label_mean"] = float((lbl * v).sum().item() / denom)
                    else:
                        aux_logs["stop_label_mean"] = float(lbl.mean().item())
                else:
                    aux_logs["stop_label_mean"] = float(lbl.mean().item())
            if torch.is_tensor(stop_dists):
                dist = stop_dists.detach().float()
                if torch.is_tensor(stop_valid_mask):
                    v = stop_valid_mask.detach().float().clamp(min=0.0, max=1.0)
                    denom = float(v.sum().item())
                    if denom > 0.0:
                        aux_logs["stop_dist_mean"] = float((dist * v).sum().item() / denom)
                    else:
                        aux_logs["stop_dist_mean"] = float(dist.mean().item())
                else:
                    aux_logs["stop_dist_mean"] = float(dist.mean().item())
            if torch.is_tensor(phase2_valid_mask):
                aux_logs["phase2_valid_ratio"] = float(phase2_valid_mask.detach().float().mean().item())
            if (
                torch.is_tensor(predicted_stop_probs)
                and torch.is_tensor(stop_labels)
            ):
                probs = predicted_stop_probs.detach().view(-1).float()
                labels = stop_labels.detach().to(device=probs.device).view(-1).float()
                if torch.is_tensor(stop_valid_mask):
                    valid = stop_valid_mask.detach().to(device=probs.device).view(-1).float() > 0.5
                else:
                    valid = torch.ones_like(labels, dtype=torch.bool)
                pos_mask = valid & (labels > 0.5)
                pos_count = int(pos_mask.sum().item())
                if pos_count > 0:
                    fn_count = int(((probs < 0.5) & pos_mask).sum().item())
                    aux_logs["stop_fn_rate"] = float(fn_count / max(pos_count, 1))
                    self._stop_fn_count_sum += float(fn_count)
                    self._stop_fn_pos_count_sum += float(pos_count)
        except Exception:
            pass
        if aux_logs:
            for key, value in aux_logs.items():
                self._aux_log_sums[key] = float(self._aux_log_sums.get(key, 0.0) + float(value))
                self._aux_log_counts[key] = int(self._aux_log_counts.get(key, 0) + 1)

        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs, *args, **kwargs):
        if isinstance(logs, dict) and self._aux_log_sums:
            avg_aux_logs = {}
            for key, sum_value in self._aux_log_sums.items():
                cnt = int(self._aux_log_counts.get(key, 0))
                if cnt <= 0:
                    continue
                avg_aux_logs[key] = float(sum_value / cnt)
            if self._stop_fn_pos_count_sum > 0.0:
                avg_aux_logs["stop_fn_rate"] = float(self._stop_fn_count_sum / self._stop_fn_pos_count_sum)
            for key, value in avg_aux_logs.items():
                logs.setdefault(key, value)
            self._aux_log_sums.clear()
            self._aux_log_counts.clear()
            self._stop_fn_count_sum = 0.0
            self._stop_fn_pos_count_sum = 0.0
        return super().log(logs, *args, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        self._sanitize_nonfinite_gradients_and_params()
        if not torch.isfinite(loss):
            if self._is_main_process() and not self._nonfinite_warned:
                logger.warning("Non-finite training_step loss detected; replacing reported loss with 0.0.")
                self._nonfinite_warned = True
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        self._report_lora_grad_health_once()
        return loss

    def _sanitize_nonfinite_gradients_and_params(self):
        nonfinite_grad_tensors = 0
        nonfinite_param_tensors = 0
        nonfinite_grad_names = []
        nonfinite_param_names = []
        for name, p in self.model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
                nonfinite_grad_tensors += 1
                if len(nonfinite_grad_names) < 6:
                    nonfinite_grad_names.append(name)
            if p.requires_grad and not torch.isfinite(p.data).all():
                with torch.no_grad():
                    p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1e4, neginf=-1e4)
                nonfinite_param_tensors += 1
                if len(nonfinite_param_names) < 6:
                    nonfinite_param_names.append(name)

        if (nonfinite_grad_tensors > 0 or nonfinite_param_tensors > 0) and self._is_main_process():
            logger.warning(
                "Sanitized non-finite tensors: grad_tensors=%d, param_tensors=%d, grad_examples=%s, param_examples=%s",
                nonfinite_grad_tensors,
                nonfinite_param_tensors,
                ",".join(nonfinite_grad_names) if nonfinite_grad_names else "none",
                ",".join(nonfinite_param_names) if nonfinite_param_names else "none",
            )


class ProgressLoggingCallback(TrainerCallback):
    

    def __init__(self):
        self._start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        if self._start_time is None:
            return
        if not isinstance(logs, dict):
            return

        step = int(state.global_step or 0)
        total_steps = int(state.max_steps or 0)
        remaining_steps = max(total_steps - step, 0) if total_steps > 0 else 0
        elapsed_sec = time.time() - self._start_time
        avg_step_sec = (elapsed_sec / step) if step > 0 else 0.0
        eta_sec = remaining_steps * avg_step_sec

        metric_parts = []
        if "loss" in logs:
            metric_parts.append(f"loss={float(logs['loss']):.4f}")
        if "learning_rate" in logs:
            metric_parts.append(f"lr={float(logs['learning_rate']):.6g}")
        if "grad_norm" in logs:
            metric_parts.append(f"grad_norm={float(logs['grad_norm']):.4f}")
        if "epoch" in logs:
            metric_parts.append(f"epoch={float(logs['epoch']):.4f}")
        if "trajectory_loss" in logs:
            metric_parts.append(f"trajectory_loss={float(logs['trajectory_loss']):.4f}")
        if "stop_loss" in logs:
            metric_parts.append(f"stop_loss={float(logs['stop_loss']):.4f}")
        for key in ["clearance_loss", "da3_joint_raw_loss", "da3_joint_weighted_loss", "da3_joint_valid_rate"]:
            if key in logs:
                metric_parts.append(f"{key}={float(logs[key]):.4f}")
        if "stop_fn_rate" in logs:
            metric_parts.append(f"stop_fn_rate={float(logs['stop_fn_rate']):.4f}")

        core_model = _resolve_core_model(kwargs.get("model", None))
        if core_model is not None:
            latest_loss_components = getattr(core_model, "_latest_loss_components", None)
            if isinstance(latest_loss_components, dict):
                trajectory_loss = latest_loss_components.get("trajectory_loss", None)
                stop_loss = latest_loss_components.get("stop_loss", None)
                if trajectory_loss is not None and "trajectory_loss" not in logs:
                    metric_parts.append(f"trajectory_loss={float(trajectory_loss):.4f}")
                if stop_loss is not None and "stop_loss" not in logs:
                    metric_parts.append(f"stop_loss={float(stop_loss):.4f}")
                for key in ["clearance_loss", "da3_joint_raw_loss", "da3_joint_weighted_loss", "da3_joint_valid_rate"]:
                    value = latest_loss_components.get(key, None)
                    if value is not None and key not in logs:
                        metric_parts.append(f"{key}={float(value):.4f}")
            else:
                trajectory_loss = getattr(core_model, "_last_trajectory_loss", None)
                stop_loss = getattr(core_model, "_last_stop_loss", None)
                if trajectory_loss is not None and "trajectory_loss" not in logs:
                    metric_parts.append(f"trajectory_loss={float(trajectory_loss):.4f}")
                if stop_loss is not None and "stop_loss" not in logs:
                    metric_parts.append(f"stop_loss={float(stop_loss):.4f}")
        metrics_text = " ".join(metric_parts)

        if total_steps > 0:
            prefix = f"step={step}/{total_steps}"
        else:
            prefix = f"step={step}"
        logger.info(
            "%s time=%s<%s %s",
            prefix,
            _format_hms(elapsed_sec),
            _format_hms(eta_sec),
            metrics_text,
        )


class PeriodicArtifactSaveCallback(TrainerCallback):
    

    def __init__(self, tokenizer, processor, lora_enable: bool):
        self.tokenizer = tokenizer
        self.processor = processor
        self.lora_enable = bool(lora_enable)

    def on_save(self, args, state, control, **kwargs):
        if not bool(getattr(args, "should_save", True)):
            return control
        if state.global_step is None:
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{int(state.global_step)}")
        os.makedirs(ckpt_dir, exist_ok=True)

        model = kwargs.get("model", None)
        if self.lora_enable and model is not None:
            non_lora_state_dict = collect_non_lora_state_dict(model.named_parameters())
            torch.save(non_lora_state_dict, os.path.join(ckpt_dir, "non_lora_trainables.bin"))

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(ckpt_dir)
        if self.processor is not None:
            self.processor.save_pretrained(ckpt_dir)
        if model is not None:
            save_uav_config(model, ckpt_dir)
        return control


def is_main_process(training_args: TrainingArguments) -> bool:
    if training_args.local_rank in (-1, 0):
        return True
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return False


def resolve_torch_dtype(dtype_name: str, fallback: torch.dtype) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = (dtype_name or "").strip().lower()
    if key == "":
        return fallback
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Use one of: {sorted(mapping.keys())}")
    return mapping[key]


def normalize_4bit_quant_type(quant_type: str) -> str:
    raw = (quant_type or "").strip().lower()
    if raw in ("fp4", "nf4"):
        return raw
    raise ValueError("Unsupported bnb_4bit_quant_type. Use nf4 or fp4")


def unwrap_uav_model(model):
    while hasattr(model, "module"):
        model = model.module
    
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "model"):
        return base_model.model
    return model


def _set_trainable(module, trainable: bool):
    if module is None:
        return
    for p in module.parameters():
        if trainable:
            
            if p.is_floating_point() or p.is_complex():
                p.requires_grad = True
        else:
            p.requires_grad = False


def _set_parameter_trainable(parameter, trainable: bool):
    if parameter is None:
        return
    if trainable:
        if parameter.is_floating_point() or parameter.is_complex():
            parameter.requires_grad = True
    else:
        parameter.requires_grad = False


def _is_bnb_4bit_linear(module) -> bool:
    cls_name = module.__class__.__name__.lower()
    mod_name = module.__class__.__module__.lower()
    return "bitsandbytes" in mod_name and "4bit" in cls_name and "linear" in cls_name


def _is_lora_linear_candidate(module) -> bool:
    return isinstance(module, nn.Linear) or _is_bnb_4bit_linear(module)


def get_uav_language_model(uav_model):
    return uav_model.model.language_model


def find_lora_target_modules(model) -> list:
    
    exclude_keywords = (
        "visual",
        "trajectory_token_embedding",
        "trajectory_head",
        "trajectory_output",
        "stop_head",
        "lm_head",
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if not _is_lora_linear_candidate(module):
            continue
        
        
        if not (
            name.startswith("model.language_model.layers.")
        ):
            continue
        if any(keyword in name for keyword in exclude_keywords):
            continue
        
        
        lora_module_names.add(name)
    if len(lora_module_names) == 0:
        raise RuntimeError("No LoRA target modules were found. Please verify model structure or override --lora_target_modules.")
    return sorted(lora_module_names)


def find_visual_merger_lora_targets(model) -> list:
    
    patterns = (
        "visual.merger.linear_fc1",   
        "visual.merger.linear_fc2",   
    )
    targets = set()
    for name, module in model.named_modules():
        if not _is_lora_linear_candidate(module):
            continue
        if any(name.endswith(p) for p in patterns):
            targets.add(name)
    return sorted(targets)


def _is_visual_encoder_lora_name(name: str) -> bool:
    
    parts = (name or "").split(".")
    if "visual" not in parts:
        return False
    visual_idx = parts.index("visual")
    tail = parts[visual_idx + 1 :]
    if not tail:
        return False
    low_tail = ".".join(tail).lower()
    
    
    excluded = (
        "merger",
        "projector",
        "patch_embed",
        "patchifier",
        "rotary",
        "rope",
        "pos_embed",
        "position",
        "embeddings",
    )
    return not any(x in low_tail for x in excluded)


def _extract_visual_encoder_block_id(name: str):
    parts = name.split(".")
    if "visual" not in parts:
        return None
    tail = parts[parts.index("visual") + 1:]
    if len(tail) >= 2 and tail[0] == "blocks" and tail[1].isdigit():
        return int(tail[1])
    return None


def parse_visual_lora_block_indices(spec: str) -> tuple:
    text = str(spec or "").strip()
    if not text:
        return ()
    indices = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            raise ValueError(f"Invalid empty visual LoRA block in {spec!r}.")
        try:
            block_id = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid visual LoRA block index {token!r}.") from exc
        if block_id < 0 or block_id in indices:
            raise ValueError(f"Invalid or duplicate visual LoRA block index {block_id}.")
        indices.append(block_id)
    return tuple(indices)


def find_visual_encoder_block_lora_targets(model, block_indices) -> list:
    requested = {int(value) for value in block_indices}
    candidates = []
    available = set()
    for name, module in model.named_modules():
        if not _is_lora_linear_candidate(module) or not _is_visual_encoder_lora_name(name):
            continue
        block_id = _extract_visual_encoder_block_id(name)
        if block_id is None:
            continue
        available.add(block_id)
        if block_id in requested:
            candidates.append((block_id, name))
    missing = requested - available
    resolved = {block_id for block_id, _ in candidates}
    if missing or resolved != requested:
        raise RuntimeError(
            f"Visual LoRA target mismatch: requested={sorted(requested)}, "
            f"available={sorted(available)}, resolved={sorted(resolved)}"
        )
    targets = sorted({name for _, name in candidates})
    if not targets:
        raise RuntimeError(f"No linear LoRA targets found in visual blocks {sorted(requested)}.")
    return targets


def _filter_lora_targets_keep_visual_projector_only(target_modules: list, allow_visual_encoder: bool = False) -> list:
    filtered = []
    for name in target_modules:
        n = (name or "").strip()
        low = n.lower()
        if "visual" in low and "visual.merger" not in low and not allow_visual_encoder:
            continue
        filtered.append(n)
    return filtered


def _module_contains_bnb_4bit_linear(module) -> bool:
    for sub_m in module.modules():
        if _is_bnb_4bit_linear(sub_m):
            return True
    return False


def rebuild_uav_heads_for_qlora(model, target_dtype: torch.dtype):
    
    uav_model = unwrap_uav_model(model)
    hidden_size = resolve_uav_hidden_size(uav_model.config)
    reduced_size = max(64, hidden_size // 2)

    ref_param = next(uav_model.parameters(), None)
    device = ref_param.device if ref_param is not None else torch.device("cpu")

    if _module_contains_bnb_4bit_linear(uav_model.trajectory_head):
        uav_model.trajectory_head = nn.Sequential(
            nn.Linear(hidden_size, reduced_size, device=device, dtype=target_dtype),
            nn.ReLU(),
            nn.Linear(reduced_size, reduced_size, device=device, dtype=target_dtype),
            nn.ReLU(),
            nn.Linear(reduced_size, 64, device=device, dtype=target_dtype),
        )
    if _module_contains_bnb_4bit_linear(uav_model.trajectory_output):
        uav_model.trajectory_output = nn.Linear(64, uav_model.traj_horizon * 3, device=device, dtype=target_dtype)
    if _module_contains_bnb_4bit_linear(uav_model.stop_head):
        uav_model.stop_head = nn.Sequential(
            nn.Linear(hidden_size, reduced_size, device=device, dtype=target_dtype),
            nn.ReLU(),
            nn.Linear(reduced_size, 1, device=device, dtype=target_dtype),
        )
    if hasattr(uav_model, "da3_joint_projector") and _module_contains_bnb_4bit_linear(uav_model.da3_joint_projector):
        from .qwen_uav_model import DA3JointProjector

        uav_model.da3_joint_projector = DA3JointProjector(
            input_dim=hidden_size,
            teacher_dim=getattr(uav_model, "da3_joint_teacher_dim", 1024),
        ).to(device=device, dtype=target_dtype)


def configure_trainable_params(model, model_args: ModelArguments):
    uav_model = unwrap_uav_model(model)

    _set_trainable(uav_model, False)

    if model_args.tune_mm_vision and not model_args.lora_enable:
        _set_trainable(uav_model.visual, True)

    if (
        model_args.tune_mm_mlp
        and hasattr(uav_model.visual, "merger")
        and not (model_args.lora_enable and model_args.tune_mm_mlp_with_lora)
    ):
        _set_trainable(uav_model.visual.merger, True)

    if model_args.tune_mm_llm:
        _set_trainable(get_uav_language_model(uav_model), True)
        _set_trainable(uav_model.lm_head, True)

    if model_args.tune_trajectory_head:
        module_names = [
            "trajectory_token_embedding",
            "trajectory_head",
            "trajectory_output",
            "stop_head",
        ]
        if model_args.use_da3_joint_supervision:
            module_names.append("da3_joint_projector")
        for module_name in module_names:
            module = getattr(uav_model, module_name, None)
            if module is not None:
                _set_trainable(module, True)


def restore_non_lora_trainables_after_lora(model, model_args: ModelArguments):
    
    uav_model = unwrap_uav_model(model)

    if model_args.tune_mm_vision and hasattr(uav_model, "visual") and not model_args.lora_enable:
        _set_trainable(uav_model.visual, True)

    if (
        model_args.tune_mm_mlp
        and hasattr(uav_model, "visual")
        and hasattr(uav_model.visual, "merger")
        and not model_args.tune_mm_mlp_with_lora
    ):
        _set_trainable(uav_model.visual.merger, True)

    if model_args.tune_trajectory_head:
        module_names = [
            "trajectory_token_embedding",
            "trajectory_head",
            "trajectory_output",
            "stop_head",
        ]
        if model_args.use_da3_joint_supervision:
            module_names.append("da3_joint_projector")
        for module_name in module_names:
            module = getattr(uav_model, module_name, None)
            if module is not None:
                _set_trainable(module, True)


def capture_requires_grad_flags(model) -> dict:
    return {name: bool(p.requires_grad) for name, p in model.named_parameters()}


def restore_requires_grad_flags(model, flags: dict):
    if not isinstance(flags, dict):
        return
    for name, p in model.named_parameters():
        if name in flags:
            p.requires_grad = bool(flags[name])


def configure_head_only_warmup_trainable_params(model):
    uav_model = unwrap_uav_model(model)
    _set_trainable(uav_model, False)
    for module_name in [
        "trajectory_token_embedding",
        "trajectory_head",
        "trajectory_output",
        "stop_head",
        "da3_joint_projector",
    ]:
        module = getattr(uav_model, module_name, None)
        if module is not None:
            _set_trainable(module, True)


def build_head_warmup_args(training_args: TrainingArguments) -> TrainingArguments:
    warmup_args = copy.deepcopy(training_args)
    warmup_steps = max(1, int(getattr(training_args, "head_warmup_steps", 5) or 5))
    warmup_lr = float(getattr(training_args, "head_warmup_learning_rate", 1e-4) or 1e-4)

    warmup_args.max_steps = warmup_steps
    warmup_args.num_train_epochs = 1
    warmup_args.learning_rate = warmup_lr
    warmup_args.non_lora_trainable_learning_rate = None
    warmup_args.warmup_ratio = 0.0
    warmup_args.warmup_steps = 0
    warmup_args.lr_scheduler_type = "constant"
    warmup_args.save_strategy = "no"
    warmup_args.save_steps = 0
    warmup_args.save_total_limit = 1
    return warmup_args


def ensure_input_require_grads_for_checkpointing(model, training_args: TrainingArguments):
    if not training_args.gradient_checkpointing:
        return
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return

    input_embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if input_embeddings is None:
        return

    def _make_inputs_require_grad(_, __, output):
        if isinstance(output, torch.Tensor):
            output.requires_grad_(True)

    input_embeddings.register_forward_hook(_make_inputs_require_grad)


def resolve_optimizer_with_fallback(training_args: TrainingArguments):
    optim_value = training_args.optim.value if hasattr(training_args.optim, "value") else str(training_args.optim)
    optim_value = str(optim_value).strip().lower()
    if optim_value != "adamw_torch_fused":
        return

    if not torch.cuda.is_available():
        logger.warning("Requested optim=adamw_torch_fused but CUDA is unavailable. Fallback to adamw_torch.")
        training_args.optim = "adamw_torch"
        return

    try:
        if "fused" not in inspect.signature(torch.optim.AdamW).parameters:
            logger.warning(
                "Current torch.optim.AdamW has no fused argument. Fallback optim from adamw_torch_fused to adamw_torch."
            )
            training_args.optim = "adamw_torch"
    except Exception:
        logger.warning("Unable to verify fused AdamW support. Fallback optim from adamw_torch_fused to adamw_torch.")
        training_args.optim = "adamw_torch"


def collect_non_lora_state_dict(named_parameters):
    return {k: v.detach().cpu() for k, v in named_parameters if "lora_" not in k and v.requires_grad}


def save_uav_config(model, output_dir: str):
    uav_model = unwrap_uav_model(model)
    config = getattr(uav_model, "config", None)
    if config is None:
        return
    if hasattr(config, "save_pretrained"):
        config.save_pretrained(output_dir)
    elif hasattr(config, "to_json_file"):
        config.to_json_file(os.path.join(output_dir, "config.json"))
    save_special_embeddings(uav_model, output_dir)


def validate_resume_checkpoint_projects(checkpoints, expected_project: str):
    for checkpoint in checkpoints:
        read_checkpoint_config(checkpoint)
        config_path = pathlib.Path(checkpoint) / "config.json"
        if not config_path.is_file():
            raise RuntimeError(f"Cannot resume from checkpoint without config.json: {checkpoint}")
        with config_path.open("r", encoding="utf-8") as f:
            checkpoint_config = json.load(f)
        checkpoint_project = checkpoint_config.get("project")
        if checkpoint_project != expected_project:
            raise RuntimeError(
                f"Checkpoint project={checkpoint_project!r}; expected {expected_project!r}: {checkpoint}"
            )
        if checkpoint_config.get("visual_lora_block_indices") != [5, 11, 17, 23]:
            raise RuntimeError(f"Checkpoint has incompatible visual LoRA blocks: {checkpoint}")
        expected = {
            "geometry_teacher": "da3_large",
            "depth_reconstruction": False,
            "dpt_present": False,
            "clearance_loss_weight": 10.0,
            "clearance_safe_margin_m": 2.0,
            "clearance_depth_sampling": "local_min_argmin_v1",
            "clearance_depth_pool_size": 8,
            "clearance_depth_zero_is_valid": True,
            "da3_joint_weight": 0.5,
            "da3_joint_teacher_dim": 1024,
        }
        mismatches = [key for key, value in expected.items() if checkpoint_config.get(key) != value]
        if mismatches:
            raise RuntimeError(f"Checkpoint has incompatible metadata {mismatches}: {checkpoint}")


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    requested_visual_blocks = parse_visual_lora_block_indices(model_args.visual_lora_block_indices)
    expected_visual_blocks = (5, 11, 17, 23)
    if requested_visual_blocks != expected_visual_blocks:
        raise ValueError(
            "GeoRisk requires visual LoRA blocks "
            f"{expected_visual_blocks}, got {requested_visual_blocks}."
        )
    if float(model_args.clearance_loss_weight) != 10.0 or float(model_args.clearance_safe_margin_m) != 2.0:
        raise ValueError("This project requires clearance weight 10.0 and margin 2.0 m.")
    if int(model_args.clearance_depth_pool_size) != 8:
        raise ValueError(
            "GeoRisk requires "
            f"clearance_depth_pool_size=8, got {model_args.clearance_depth_pool_size}."
        )
    if float(model_args.da3_joint_weight) != 0.5 or int(model_args.da3_joint_teacher_dim) != 1024:
        raise ValueError("This project requires DA3 weight 0.5 and teacher dimension 1024.")
    if not data_args.strict_da3_joint_cache:
        raise ValueError("This project requires --strict_da3_joint_cache True.")
    if not model_args.lora_enable or model_args.tune_mm_mlp_with_lora or model_args.tune_mm_llm:
        raise ValueError("GeoRisk uses language/visual LoRA and a fully trained visual merger.")
    if not all((model_args.tune_mm_mlp, model_args.tune_mm_vision, model_args.tune_trajectory_head,
                model_args.use_clearance_supervision, model_args.use_da3_joint_supervision)):
        raise ValueError("GeoRisk requires both supervision branches and the navigation heads.")
    if (model_args.traj_horizon, model_args.traj_execute_points) != (10, 5):
        raise ValueError("GeoRisk predicts 10 trajectory points and executes 5.")
    expected_lora_alpha = int(model_args.lora_r) * 2
    model_args.lora_alpha = expected_lora_alpha
    expected_merger_lora_alpha = int(model_args.merger_lora_r) * 2
    model_args.merger_lora_alpha = expected_merger_lora_alpha

    
    training_args.remove_unused_columns = False
    training_args.disable_tqdm = True
    if training_args.gradient_checkpointing:
        if training_args.gradient_checkpointing_kwargs is None:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        elif "use_reentrant" not in training_args.gradient_checkpointing_kwargs:
            training_args.gradient_checkpointing_kwargs = dict(training_args.gradient_checkpointing_kwargs)
            training_args.gradient_checkpointing_kwargs["use_reentrant"] = False

    resolve_optimizer_with_fallback(training_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info("Data args: %s", data_args)
    logger.info("Training args: %s", training_args)
    data_args.view_mode = (data_args.view_mode or "dual").strip().lower()
    if data_args.view_mode != "dual":
        raise ValueError(
            f"Unsupported view_mode '{data_args.view_mode}'. GeoRisk requires dual front/down views."
        )

    os.makedirs(training_args.output_dir, exist_ok=True)
    torch_dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32)
    requested_4bit = model_args.use_qlora or model_args.load_in_4bit
    if model_args.use_qlora and not model_args.lora_enable:
        logger.warning("use_qlora=True requires LoRA adapters. Forcing lora_enable=True.")
        model_args.lora_enable = True

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    image_processor = processor.image_processor
    image_processor.max_pixels = data_args.max_pixels
    image_processor.min_pixels = data_args.min_pixels
    image_processor.size["longest_edge"] = data_args.max_pixels
    image_processor.size["shortest_edge"] = data_args.min_pixels

    model_load_kwargs = dict(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=model_args.attn_implementation,
        stop_loss_weight=model_args.stop_loss_weight,
        stop_rank_loss_weight=model_args.stop_rank_loss_weight,
        stop_rank_margin=model_args.stop_rank_margin,
        stop_rank_min_gap=model_args.stop_rank_min_gap,
        traj_horizon=model_args.traj_horizon,
        traj_execute_points=model_args.traj_execute_points,
        use_clearance_supervision=model_args.use_clearance_supervision,
        clearance_loss_weight=model_args.clearance_loss_weight,
        clearance_safe_margin_m=model_args.clearance_safe_margin_m,
        clearance_depth_max_m=model_args.clearance_depth_max_m,
        clearance_voxel_size_m=model_args.clearance_voxel_size_m,
        clearance_local_range_m=model_args.clearance_local_range_m,
        clearance_depth_pool_size=model_args.clearance_depth_pool_size,
        clearance_max_voxels=model_args.clearance_max_voxels,
        clearance_path_samples_per_segment=model_args.clearance_path_samples_per_segment,
        clearance_temperature=model_args.clearance_temperature,
        use_da3_joint_supervision=model_args.use_da3_joint_supervision,
        da3_joint_weight=model_args.da3_joint_weight,
        da3_joint_teacher_dim=model_args.da3_joint_teacher_dim,
    )
    if requested_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError(
                "4-bit quantization was requested, but BitsAndBytesConfig is unavailable. "
                "Please install a bitsandbytes-enabled environment."
            ) from exc
        try:
            import bitsandbytes as bnb
        except Exception as exc:
            raise RuntimeError(
                "4-bit quantization was requested, but `bitsandbytes` is not installed or not loadable. "
                "Please install bitsandbytes (or use WSL/Linux with CUDA) and retry."
            ) from exc

        bnb_compute_dtype = resolve_torch_dtype(model_args.bnb_4bit_compute_dtype, fallback=torch_dtype)
        bnb_quant_type = normalize_4bit_quant_type(model_args.bnb_4bit_quant_type)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=bnb_quant_type,
            bnb_4bit_compute_dtype=bnb_compute_dtype,
            bnb_4bit_use_double_quant=model_args.bnb_4bit_use_double_quant,
        )
        model_load_kwargs["quantization_config"] = quantization_config
        model_load_kwargs["dtype"] = bnb_compute_dtype
        logger.info(
            "Enable 4-bit quantization: requested=%s, resolved=%s, compute_dtype=%s, double_quant=%s",
            model_args.bnb_4bit_quant_type,
            bnb_quant_type,
            model_args.bnb_4bit_compute_dtype,
            model_args.bnb_4bit_use_double_quant,
        )
        logger.info("bitsandbytes version: %s", getattr(bnb, "__version__", "unknown"))
    else:
        model_load_kwargs["dtype"] = torch_dtype

    model = GeoRiskForNavigation.from_pretrained(**model_load_kwargs)
    model.config.use_cache = False
    model.config.project = "GeoRisk"
    model.config.visual_lora_block_indices = list(requested_visual_blocks)
    model.config.geometry_teacher = "da3_large"
    model.config.depth_reconstruction = False
    model.config.dpt_present = False
    model.config.clearance_loss_weight = 10.0
    model.config.clearance_safe_margin_m = 2.0
    model.config.clearance_depth_sampling = "local_min_argmin_v1"
    model.config.clearance_depth_pool_size = 8
    model.config.clearance_depth_zero_is_valid = True

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_new_tokens = tokenizer.add_tokens(
        [DEFAULT_TRAJ_TOKEN, DEFAULT_STOP_TOKEN],
        special_tokens=True,
    )
    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    model.get_special_token_id(
        {
            DEFAULT_TRAJ_TOKEN: tokenizer.convert_tokens_to_ids(DEFAULT_TRAJ_TOKEN),
            DEFAULT_STOP_TOKEN: tokenizer.convert_tokens_to_ids(DEFAULT_STOP_TOKEN),
        }
    )

    if requested_4bit:
        rebuild_uav_heads_for_qlora(model, target_dtype=torch_dtype)
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
        )

    configure_trainable_params(model, model_args)

    if model_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        target_modules_spec = (model_args.lora_target_modules or "").strip().lower()
        if target_modules_spec in ("", "auto"):
            target_modules = find_lora_target_modules(model)
        else:
            target_modules = [x.strip() for x in model_args.lora_target_modules.split(",") if x.strip()]
        
        if model_args.tune_mm_mlp and model_args.tune_mm_mlp_with_lora:
            merger_targets = find_visual_merger_lora_targets(model)
            if len(merger_targets) == 0:
                logger.warning(
                    "No visual merger linear modules found for LoRA; projector LoRA will be skipped."
                )
            else:
                target_modules.extend(merger_targets)
        if model_args.tune_mm_vision and model_args.lora_enable:
            visual_targets = find_visual_encoder_block_lora_targets(model, requested_visual_blocks)
            if len(visual_targets) == 0:
                raise RuntimeError("No visual encoder LoRA targets found for blocks 5/11/17/23.")
            else:
                target_modules.extend(visual_targets)
        
        target_modules = list(dict.fromkeys(target_modules))
        target_modules = _filter_lora_targets_keep_visual_projector_only(
            target_modules,
            allow_visual_encoder=bool(model_args.tune_mm_vision and model_args.lora_enable),
        )
        if len(target_modules) == 0:
            raise RuntimeError("All LoRA targets were filtered out. Please check lora_target_modules / tune_mm_mlp settings.")

        lora_config_kwargs = dict(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        if model_args.tune_mm_mlp and model_args.tune_mm_mlp_with_lora:
            merger_targets = find_visual_merger_lora_targets(model)
            if len(merger_targets) > 0:
                lora_cfg_sig = inspect.signature(LoraConfig.__init__)
                if "rank_pattern" not in lora_cfg_sig.parameters or "alpha_pattern" not in lora_cfg_sig.parameters:
                    raise RuntimeError(
                        "Current peft.LoraConfig does not support rank_pattern/alpha_pattern, "
                        "but merger_lora_r/merger_lora_alpha were requested. Please upgrade peft."
                    )
                lora_config_kwargs["rank_pattern"] = {
                    name: int(model_args.merger_lora_r) for name in merger_targets
                }
                lora_config_kwargs["alpha_pattern"] = {
                    name: int(model_args.merger_lora_alpha) for name in merger_targets
                }

        lora_config = LoraConfig(**lora_config_kwargs)
        model = get_peft_model(model, lora_config)
        restore_non_lora_trainables_after_lora(model, model_args)

    ensure_input_require_grads_for_checkpointing(model, training_args)

    train_dataset = GeoRiskDataset(
        data_path=data_args.data_path,
        dataset_path=data_args.dataset_path,
        dense_dataset_path=data_args.dense_dataset_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_samples=data_args.max_samples,
        view_mode=data_args.view_mode,
        traj_horizon=model_args.traj_horizon,
        stop_terminal_repeat=data_args.stop_terminal_repeat,
        stop_soft_r=data_args.stop_soft_r,
        stop_soft_tau=data_args.stop_soft_tau,
        stop_label_clip_eps=data_args.stop_label_clip_eps,
        stop_phase2_disable_dist=data_args.stop_phase2_disable_dist,
        stop_near_resample_rules=data_args.stop_near_resample_rules,
        depth_dataset_path=data_args.depth_dataset_path,
        use_clearance_supervision=model_args.use_clearance_supervision,
        clearance_bad_depth_maps=data_args.clearance_bad_depth_maps,
        da3_joint_teacher_path=data_args.da3_joint_teacher_path,
        use_da3_joint_supervision=model_args.use_da3_joint_supervision,
        strict_da3_joint_cache=data_args.strict_da3_joint_cache,
        da3_joint_teacher_dim=model_args.da3_joint_teacher_dim,
    )
    data_collator = GeoRiskCollator(tokenizer=tokenizer)

    existing_checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    validate_resume_checkpoint_projects(
        existing_checkpoints,
        "GeoRisk",
    )
    should_run_head_warmup = (
        bool(getattr(training_args, "head_warmup_enable", True))
        and int(getattr(training_args, "head_warmup_steps", 0) or 0) > 0
        and len(existing_checkpoints) == 0
    )
    if should_run_head_warmup:
        full_trainable_flags = capture_requires_grad_flags(model)
        configure_head_only_warmup_trainable_params(model)
        warmup_args = build_head_warmup_args(training_args)
        logger.info(
            "Starting head-only warmup before main training: steps=%d lr=%g (independent of main epochs/steps).",
            int(warmup_args.max_steps),
            float(warmup_args.learning_rate),
        )
        warmup_trainer = UAVTrainer(
            model=model,
            processing_class=tokenizer,
            args=warmup_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
            callbacks=[ProgressLoggingCallback()],
        )
        warmup_trainer.remove_callback(PrinterCallback)
        warmup_trainer.train()
        restore_requires_grad_flags(model, full_trainable_flags)
        del warmup_trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Head-only warmup finished; entering normal training stage.")
    elif len(existing_checkpoints) > 0:
        logger.info("Skip head-only warmup because checkpoint exists; training will resume directly.")
    else:
        logger.info("Head-only warmup disabled or steps<=0; entering normal training directly.")

    trainer = UAVTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        callbacks=[
            ProgressLoggingCallback(),
            PeriodicArtifactSaveCallback(
                tokenizer=tokenizer,
                processor=processor,
                lora_enable=model_args.lora_enable,
            ),
        ],
    )
    trainer.remove_callback(PrinterCallback)

    if len(existing_checkpoints) > 0:
        checkpoint = max(existing_checkpoints, key=lambda path: int(path.name.split("-")[-1]))
        load_special_embeddings(unwrap_uav_model(model), checkpoint)
        load_non_lora_weights(model, checkpoint)
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if model_args.lora_enable:
        if is_main_process(training_args):
            model.save_pretrained(training_args.output_dir)
            non_lora_state_dict = collect_non_lora_state_dict(model.named_parameters())
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
            tokenizer.save_pretrained(training_args.output_dir)
            processor.save_pretrained(training_args.output_dir)
            save_uav_config(model, training_args.output_dir)
    else:
        trainer.save_model(training_args.output_dir)
        if is_main_process(training_args):
            tokenizer.save_pretrained(training_args.output_dir)
            processor.save_pretrained(training_args.output_dir)
            save_uav_config(model, training_args.output_dir)


if __name__ == "__main__":
    train()
