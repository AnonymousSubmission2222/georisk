from pathlib import Path
import json

import torch

from .constants import DEFAULT_STOP_TOKEN, DEFAULT_TRAJ_TOKEN


PROJECT = "GeoRisk"
SCHEMA_VERSION = 1
METADATA = {
    "project": PROJECT,
    "georisk_schema_version": SCHEMA_VERSION,
    "trajectory_token": DEFAULT_TRAJ_TOKEN,
    "visual_lora_block_indices": [20, 21, 22, 23],
    "traj_horizon": 10,
    "traj_execute_points": 5,
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


def validate_checkpoint_config(config):
    errors = {key: (config.get(key), value) for key, value in METADATA.items()
              if config.get(key) != value}
    if errors:
        raise ValueError(f"Not a GeoRisk checkpoint: {errors}")


def read_checkpoint_config(directory):
    path = Path(directory)
    with (path / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    validate_checkpoint_config(config)
    for filename in ("adapter_config.json", "non_lora_trainables.bin", "special_token_embeddings.bin"):
        if not (path / filename).is_file():
            raise FileNotFoundError(path / filename)
    return config


def validate_tokenizer(tokenizer, config):
    ids = {}
    for token in (DEFAULT_TRAJ_TOKEN, DEFAULT_STOP_TOKEN):
        if token not in tokenizer.get_added_vocab():
            raise ValueError(f"Checkpoint tokenizer is missing {token}")
        ids[token] = int(tokenizer.convert_tokens_to_ids(token))
    if config.get("navigation_token_ids") != ids:
        raise ValueError("Tokenizer IDs do not match checkpoint navigation_token_ids")
    return ids


def save_special_embeddings(model, directory):
    ids = model.config.navigation_token_ids
    rows = [ids[DEFAULT_TRAJ_TOKEN], ids[DEFAULT_STOP_TOKEN]]
    torch.save({
        "token_ids": ids,
        "input": model.get_input_embeddings().weight[rows].detach().cpu().clone(),
        "output": model.get_output_embeddings().weight[rows].detach().cpu().clone(),
    }, Path(directory) / "special_token_embeddings.bin")


def load_special_embeddings(model, directory):
    state = torch.load(Path(directory) / "special_token_embeddings.bin", map_location="cpu", weights_only=True)
    ids = model.config.navigation_token_ids
    if state.get("token_ids") != ids:
        raise ValueError("Special-token embedding IDs do not match the model")
    rows = [ids[DEFAULT_TRAJ_TOKEN], ids[DEFAULT_STOP_TOKEN]]
    for name, embedding in (("input", model.get_input_embeddings()), ("output", model.get_output_embeddings())):
        values = state[name]
        if tuple(values.shape) != (2, embedding.weight.shape[1]) or not torch.isfinite(values).all():
            raise ValueError(f"Invalid {name} special-token embeddings")
        with torch.no_grad():
            embedding.weight[rows] = values.to(embedding.weight)


def non_lora_parameter_names(model):
    prefixes = ("trajectory_token_embedding.", "trajectory_head.", "trajectory_output.",
                "stop_head.", "da3_joint_projector.", "model.visual.merger.")
    names = set()
    for name, _ in model.named_parameters():
        key = name.removeprefix("base_model.model.")
        if key.startswith(prefixes) and "lora_" not in key:
            names.add(name)
    return names


def load_non_lora_weights(model, directory, *, inference=False):
    weights = torch.load(Path(directory) / "non_lora_trainables.bin", map_location="cpu", weights_only=True)
    if inference:
        
        weights = {key: value for key, value in weights.items()
                   if not key.removeprefix("base_model.model.").startswith("da3_joint_projector.")}
    expected = non_lora_parameter_names(model)
    if set(weights) != expected:
        raise ValueError(f"Non-LoRA keys differ: missing={sorted(expected - weights.keys())}, "
                         f"unexpected={sorted(weights.keys() - expected)}")
    state = model.state_dict()
    for key, value in weights.items():
        if value.shape != state[key].shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid checkpoint tensor: {key}")
    model.load_state_dict(weights, strict=False)
