#!/usr/bin/env python
import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import traceback
import multiprocessing as mp
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


WORKSPACE_ROOT = Path(os.environ.get("GEORISK_ASSETS_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
DEFAULT_DA3_REPO = WORKSPACE_ROOT / "Depth-Anything-3-main"
DEFAULT_DA3_CHECKPOINT = WORKSPACE_ROOT / "Depth-Anything-3-Checkpoints" / "DA3-LARGE-1.1"


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_export_layer(ckpt_path: Path, requested: int) -> int:
    if int(requested) > 0:
        return int(requested)
    cfg_path = ckpt_path / "config.json"
    if not cfg_path.is_file():
        return 23
    cfg = load_json(cfg_path)
    out_layers = cfg.get("config", {}).get("net", {}).get("out_layers", None)
    if isinstance(out_layers, list) and len(out_layers) > 0:
        return int(out_layers[-1])
    return 23


def resolve_da3_model_name(ckpt_path: Path) -> str:
    cfg_path = ckpt_path / "config.json"
    if not cfg_path.is_file():
        return "da3-base"
    try:
        cfg = load_json(cfg_path)
        name = str(cfg.get("model_name", "")).strip()
        if name:
            return name
    except Exception:
        pass
    name = ckpt_path.name.lower()
    if "large" in name:
        return "da3-large"
    if "small" in name:
        return "da3-small"
    if "giant" in name:
        return "da3-giant"
    return "da3-base"


def infer_da3_token_dim(ckpt_path: Path) -> int:
    cfg_path = ckpt_path / "config.json"
    if cfg_path.is_file():
        try:
            cfg = load_json(cfg_path)
            dim = cfg.get("config", {}).get("cam_enc", {}).get("dim_out", None)
            if dim is not None:
                return int(dim)
        except Exception:
            pass
    name = ckpt_path.name.lower()
    if "small" in name:
        return 384
    if "base" in name:
        return 768
    if "large" in name:
        return 1024
    if "giant" in name:
        return 1536
    return 768


def load_da3_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    safetensors_path = ckpt_path / "model.safetensors"
    if safetensors_path.is_file():
        from safetensors.torch import load_file as safe_load_file

        return safe_load_file(str(safetensors_path), device="cpu")
    for name in ("pytorch_model.bin", "model.bin", "model.pt", "checkpoint.pt"):
        path = ckpt_path / name
        if not path.is_file():
            continue
        payload = torch.load(str(path), map_location="cpu")
        if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
            payload = payload["state_dict"]
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unsupported checkpoint payload: {path}")
        return payload
    raise FileNotFoundError(f"No DA3 checkpoint weights found under {ckpt_path}")


def parse_sample_key_from_json_rel(rel: str) -> Optional[Tuple[str, str]]:
    rel = str(rel).replace("\\", "/").strip("/")
    parts = rel.split("/")
    if len(parts) >= 3:
        return parts[-3], parts[-2]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None


def collect_requested_sparse_frames(data_json: Optional[Path]) -> Dict[Tuple[str, str], List[int]]:
    if data_json is None or not data_json.is_file():
        return {}
    data = load_json(data_json)
    if isinstance(data, dict):
        entries = data.get("samples", data.get("data", []))
    else:
        entries = data
    out: "OrderedDict[Tuple[str, str], set[int]]" = OrderedDict()
    for item in entries:
        if not isinstance(item, dict):
            continue
        key = parse_sample_key_from_json_rel(str(item.get("json", "")))
        if key is None:
            continue
        out.setdefault(key, set())
        try:
            frame = int(item.get("frame", 0) or 0)
        except Exception:
            frame = 0
        if frame > 0:
            out[key].add(frame)
    return {key: sorted(frames) for key, frames in out.items()}


def iter_trajectories(input_root: Path, data_json: Path = None) -> Iterable[Tuple[str, str]]:
    if data_json is not None and data_json.is_file():
        for key in collect_requested_sparse_frames(data_json).keys():
            yield key
        return

    for map_dir in sorted(input_root.iterdir()):
        if not map_dir.is_dir():
            continue
        for traj_dir in sorted(map_dir.iterdir()):
            if traj_dir.is_dir() and (traj_dir / "merged_data.json").is_file():
                yield map_dir.name, traj_dir.name


def scan_webdataset_members(index_path: Path) -> Dict[str, Dict[str, str]]:
    index_data = load_json(index_path)
    shards = index_data.get("shards", [])
    sample_map: Dict[str, Dict[str, str]] = {}
    for shard in shards:
        shard_path = Path(str(shard))
        if not shard_path.is_absolute():
            shard_path = index_path.parent / shard_path
        shard_path = shard_path.resolve()
        with tarfile.open(shard_path, "r") as tar:
            for member in tar.getmembers():
                if not member.isfile() or not (member.name.endswith(".json") or member.name.endswith(".npz")):
                    continue
                key = member.name.rsplit(".", 1)[0]
                entry = sample_map.setdefault(
                    key,
                    {"shard_path": str(shard_path), "json_member": "", "npz_member": ""},
                )
                if member.name.endswith(".json"):
                    entry["json_member"] = member.name
                elif member.name.endswith(".npz"):
                    entry["npz_member"] = member.name
    return {k: v for k, v in sample_map.items() if v.get("json_member") and v.get("npz_member")}


def iter_webdataset_keys(sample_map: Dict[str, Dict[str, str]]) -> Iterable[Tuple[str, str]]:
    for key in sorted(sample_map.keys()):
        if "__" not in key:
            continue
        map_name, uuid = key.split("__", 1)
        yield map_name, uuid


def load_webdataset_sample(entry: Dict[str, str]) -> Tuple[Dict[str, Any], bytes]:
    with tarfile.open(entry["shard_path"], "r") as tar:
        jf = tar.extractfile(entry["json_member"])
        nf = tar.extractfile(entry["npz_member"])
        if jf is None or nf is None:
            raise FileNotFoundError(f"Missing json/npz member in {entry['shard_path']}")
        payload = json.loads(jf.read().decode("utf-8"))
        npz_bytes = nf.read()
    if isinstance(payload, dict) and "merged_data" in payload:
        merged_data = payload["merged_data"]
    else:
        merged_data = payload
    return merged_data, npz_bytes


class WebDatasetSampleReader:
    def __init__(self, max_open: int = 4):
        self.max_open = max(1, int(max_open))
        self._handles: "OrderedDict[str, tarfile.TarFile]" = OrderedDict()

    def _get_tar(self, shard_path: str) -> tarfile.TarFile:
        shard_path = str(shard_path)
        handle = self._handles.get(shard_path)
        if handle is not None:
            self._handles.move_to_end(shard_path)
            return handle
        handle = tarfile.open(shard_path, "r")
        
        handle.getmembers()
        self._handles[shard_path] = handle
        self._handles.move_to_end(shard_path)
        while len(self._handles) > self.max_open:
            _, old = self._handles.popitem(last=False)
            old.close()
        return handle

    def load(self, entry: Dict[str, str]) -> Tuple[Dict[str, Any], bytes]:
        tar = self._get_tar(entry["shard_path"])
        jf = tar.extractfile(entry["json_member"])
        nf = tar.extractfile(entry["npz_member"])
        if jf is None or nf is None:
            raise FileNotFoundError(f"Missing json/npz member in {entry['shard_path']}")
        payload = json.loads(jf.read().decode("utf-8"))
        npz_bytes = nf.read()
        if isinstance(payload, dict) and "merged_data" in payload:
            merged_data = payload["merged_data"]
        else:
            merged_data = payload
        return merged_data, npz_bytes

    def close(self) -> None:
        while self._handles:
            _, handle = self._handles.popitem(last=False)
            handle.close()


def normalize_camera_name(name: Any) -> Optional[str]:
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="ignore")
    name = str(name).strip().lower()
    if "front" in name:
        return "frontcamera"
    if "left" in name:
        return "leftcamera"
    if "right" in name:
        return "rightcamera"
    if "rear" in name or "back" in name:
        return "rearcamera"
    if "down" in name or "bottom" in name:
        return "downcamera"
    return None


def camera_map_from_npz(data, cam_count: int) -> Dict[str, int]:
    camera_index_to_name = {
        0: "frontcamera",
        1: "leftcamera",
        2: "rightcamera",
        3: "rearcamera",
        4: "downcamera",
    }
    camera_pos_map: Dict[str, int] = {}
    if "camera_names" in data:
        names = data["camera_names"].tolist()
        if not isinstance(names, list):
            names = [names]
        for pos, raw_name in enumerate(names):
            name = normalize_camera_name(raw_name)
            if name is not None:
                camera_pos_map.setdefault(name, pos)
    if not camera_pos_map and "camera_indices" in data:
        indices = data["camera_indices"].tolist()
        if not isinstance(indices, list):
            indices = [indices]
        for pos, raw_idx in enumerate(indices):
            name = camera_index_to_name.get(int(raw_idx), None)
            if name is not None:
                camera_pos_map.setdefault(name, pos)
    if not camera_pos_map:
        if cam_count == 2:
            camera_pos_map = {"frontcamera": 0, "downcamera": 1}
        else:
            camera_pos_map = {name: idx for idx, name in camera_index_to_name.items() if idx < cam_count}
    return camera_pos_map


def sampled_sparse_frame_ids(merged_data: Dict, sample_frames: int) -> np.ndarray:
    indices = merged_data.get("index", None)
    if isinstance(indices, list) and len(indices) > 0:
        raw_ids = np.asarray([int(x) for x in indices], dtype=np.int32)
    else:
        length = int(merged_data.get("length", len(merged_data.get("trajectory", [])) or 0))
        raw_ids = np.arange(max(0, length), dtype=np.int32)
    if raw_ids.size == 0:
        return raw_ids
    if raw_ids.size <= sample_frames:
        return raw_ids
    pos = np.linspace(0, raw_ids.size - 1, int(sample_frames)).round().astype(np.int64)
    return raw_ids[pos]


def sparse_to_raw_frame_id(merged_data: Dict, sparse_frame: int) -> int:
    indices = merged_data.get("index", None)
    if isinstance(indices, list) and len(indices) > 0:
        pos = max(0, min(int(sparse_frame) - 1, len(indices) - 1))
        return int(indices[pos])
    return max(0, int(sparse_frame) - 1)


def load_images(traj_dir: Path, frame_ids: np.ndarray, cameras: List[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for frame_id in frame_ids.tolist():
        for camera in cameras:
            path = traj_dir / camera / f"{int(frame_id):06d}.png"
            if not path.is_file():
                continue
            images.append(Image.open(path).convert("RGB"))
    return images


def load_images_from_webdataset_npz(npz_bytes: bytes, frame_ids: np.ndarray, cameras: List[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as data:
        rgb_key = "imgs" if "imgs" in data else ("rgb" if "rgb" in data else None)
        if rgb_key is None:
            raise KeyError(f"WebDataset NPZ has no imgs/rgb key. Existing keys: {list(data.files)}")
        rgb = np.asarray(data[rgb_key], dtype=np.uint8)
        npz_frame_ids = data["frame_ids"].tolist() if "frame_ids" in data else None
        camera_pos_map = camera_map_from_npz(data, int(rgb.shape[1]))
    frame_map = {int(fid): idx for idx, fid in enumerate(npz_frame_ids or [])}
    for frame_id in frame_ids.tolist():
        if frame_map:
            frame_pos = frame_map.get(int(frame_id), None)
            if frame_pos is None:
                continue
        else:
            frame_pos = max(0, min(int(frame_id), len(rgb) - 1))
        for camera in cameras:
            cam_pos = camera_pos_map.get(camera, None)
            if cam_pos is None or cam_pos >= rgb.shape[1]:
                continue
            images.append(Image.fromarray(np.array(rgb[frame_pos, cam_pos], copy=True)).convert("RGB"))
    return images


def load_rgb_pack_from_webdataset_npz(npz_bytes: bytes) -> Tuple[np.ndarray, Optional[List[int]], Dict[str, int]]:
    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as data:
        rgb_key = "imgs" if "imgs" in data else ("rgb" if "rgb" in data else None)
        if rgb_key is None:
            raise KeyError(f"WebDataset NPZ has no imgs/rgb key. Existing keys: {list(data.files)}")
        rgb = np.asarray(data[rgb_key], dtype=np.uint8)
        frame_ids = data["frame_ids"].tolist() if "frame_ids" in data else None
        camera_pos_map = camera_map_from_npz(data, int(rgb.shape[1]))
    return rgb, frame_ids, camera_pos_map


def extract_images_from_rgb_pack(
    rgb: np.ndarray,
    frame_ids: Optional[List[int]],
    camera_pos_map: Dict[str, int],
    raw_frame_id: int,
    cameras: List[str],
) -> List[Image.Image]:
    if frame_ids is not None:
        frame_map = {int(fid): idx for idx, fid in enumerate(frame_ids)}
        frame_pos = frame_map.get(int(raw_frame_id), None)
        if frame_pos is None:
            return []
    else:
        frame_pos = max(0, min(int(raw_frame_id), len(rgb) - 1))
    images: List[Image.Image] = []
    for camera in cameras:
        cam_pos = camera_pos_map.get(camera, None)
        if cam_pos is None or cam_pos >= rgb.shape[1]:
            continue
        images.append(Image.fromarray(np.array(rgb[frame_pos, cam_pos], copy=True)).convert("RGB"))
    return images


def load_frame_images_from_filesystem(traj_dir: Path, raw_frame_id: int, cameras: List[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for camera in cameras:
        path = traj_dir / camera / f"{int(raw_frame_id):06d}.png"
        if path.is_file():
            images.append(Image.open(path).convert("RGB"))
    return images


def extract_da3_features(
    model,
    images: List[Image.Image],
    process_res: int,
    export_layer: int,
    batch_size: int,
    process_res_method: str,
    ref_view_strategy: str,
    output_grid_hw: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sliced_parts: List[np.ndarray] = []
    batch_size = max(1, int(batch_size))
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            pred = model.inference(
                batch,
                process_res=int(process_res),
                process_res_method=str(process_res_method),
                ref_view_strategy=str(ref_view_strategy),
                export_feat_layers=[int(export_layer)],
            )
            aux = getattr(pred, "aux", {}) or {}
            feat = aux.get(f"feat_layer_{int(export_layer)}", None)
            if feat is None:
                raise RuntimeError(f"DA3 output does not contain aux feature feat_layer_{export_layer}.")
            feat = np.asarray(feat, dtype=np.float32)
            if feat.ndim == 5:
                feat = feat.reshape(-1, feat.shape[-3], feat.shape[-2], feat.shape[-1])
            if feat.ndim != 4:
                raise RuntimeError(f"Unexpected DA3 feature shape: {feat.shape}")
            output_grid_hw = int(output_grid_hw)
            if output_grid_hw > 0 and feat.shape[1:3] != (output_grid_hw, output_grid_hw):
                feat_tensor = torch.from_numpy(feat).permute(0, 3, 1, 2)
                feat = (
                    F.adaptive_avg_pool2d(feat_tensor, (output_grid_hw, output_grid_hw))
                    .permute(0, 2, 3, 1)
                    .numpy()
                )
            sliced_parts.append(feat.reshape(feat.shape[0], -1, feat.shape[-1]).astype(np.float32))
    sliced = np.concatenate(sliced_parts, axis=0) if len(sliced_parts) > 1 else sliced_parts[0]
    global_feat = sliced.mean(axis=(0, 1))
    return sliced.astype(np.float32), global_feat.astype(np.float32)


def load_da3_model(da3_repo: Path, da3_checkpoint: Path, device: str):
    repo_src = da3_repo / "src"
    sys.path.insert(0, str(repo_src if repo_src.is_dir() else da3_repo))
    try:
        import depth_anything_3.utils.logger as da3_logging

        da3_logging.logger.level = da3_logging.LOG_LEVELS["ERROR"]
    except Exception:
        pass
    from depth_anything_3.api import DepthAnything3

    try:
        model = DepthAnything3(model_name=resolve_da3_model_name(da3_checkpoint))
        state_dict = load_da3_state_dict(da3_checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[DA3][WARN] missing checkpoint keys: {len(missing)}")
        if unexpected:
            print(f"[DA3][WARN] unexpected checkpoint keys: {len(unexpected)}")
    except Exception as exc:
        print(f"[DA3][WARN] direct checkpoint load failed, falling back to from_pretrained: {exc}")
        model = DepthAnything3.from_pretrained(str(da3_checkpoint))
    if str(device).strip().lower() != "cpu" and torch.cuda.is_available():
        model = model.to(device)
    model.eval()
    return model


def existing_frame_cache_matches(
    frame_dir: Path,
    expected_feat_dim: int,
    expected_token_count: int,
) -> bool:
    sliced_path = frame_dir / "da3_sliced.npy"
    if not sliced_path.is_file():
        return False
    if int(expected_feat_dim) <= 0:
        return True
    try:
        sliced_arr = np.load(sliced_path, mmap_mode="r")
        if sliced_arr.shape[-1] != int(expected_feat_dim):
            return False
        if int(expected_token_count) > 0 and sliced_arr.shape[-2] != int(expected_token_count):
            return False
    except Exception:
        return False
    return True


def process_one_task(
    task: Dict[str, Any],
    cfg: Dict[str, Any],
    model,
    wds_reader: Optional[WebDatasetSampleReader] = None,
) -> Tuple[str, str]:
    map_name = task["map_name"]
    uuid = task["uuid"]
    input_root = Path(cfg["input_root"])
    output_root = Path(cfg["output_root"])
    cameras = list(cfg["cameras"])
    out_dir = output_root / map_name / uuid

    npz_bytes = None
    rgb_pack: Optional[Tuple[np.ndarray, Optional[List[int]], Dict[str, int]]] = None
    if task.get("wds_entry") is not None:
        if wds_reader is not None:
            merged_data, npz_bytes = wds_reader.load(task["wds_entry"])
        else:
            merged_data, npz_bytes = load_webdataset_sample(task["wds_entry"])
        rgb_pack = load_rgb_pack_from_webdataset_npz(npz_bytes)
    else:
        traj_dir = input_root / map_name / uuid
        merged_path = traj_dir / "merged_data.json"
        if not merged_path.is_file():
            return "skip", "missing merged_data.json"
        merged_data = load_json(merged_path)

    requested_sparse = task.get("sparse_frames", None)
    if requested_sparse:
        raw_frame_ids = np.asarray(
            [sparse_to_raw_frame_id(merged_data, int(frame)) for frame in requested_sparse],
            dtype=np.int32,
        )
    else:
        raw_frame_ids = sampled_sparse_frame_ids(merged_data, sample_frames=int(cfg["sample_frames"]))
    raw_frame_ids = np.unique(raw_frame_ids.astype(np.int32))
    if raw_frame_ids.size == 0:
        return "skip", "no frames"

    cached = 0
    skipped = 0
    missing = 0
    feature_dim: Optional[int] = None
    expected_feat_dim = int(cfg.get("expected_feat_dim", 0) or 0)
    output_grid_hw = int(cfg.get("output_grid_hw", 0) or 0)
    expected_token_count = output_grid_hw * output_grid_hw if output_grid_hw > 0 else 0
    traj_dir = input_root / map_name / uuid
    for raw_frame_id in raw_frame_ids.tolist():
        frame_dir = out_dir / "frames" / f"{int(raw_frame_id):06d}"
        done = frame_dir / "da3_sliced.npy"
        if (
            done.is_file()
            and not bool(cfg["overwrite"])
            and existing_frame_cache_matches(
                frame_dir=frame_dir,
                expected_feat_dim=expected_feat_dim,
                expected_token_count=expected_token_count,
            )
        ):
            skipped += 1
            continue
        if rgb_pack is not None:
            rgb, npz_frame_ids, camera_pos_map = rgb_pack
            images = extract_images_from_rgb_pack(
                rgb=rgb,
                frame_ids=npz_frame_ids,
                camera_pos_map=camera_pos_map,
                raw_frame_id=int(raw_frame_id),
                cameras=cameras,
            )
        else:
            images = load_frame_images_from_filesystem(traj_dir, int(raw_frame_id), cameras=cameras)
        if len(images) == 0:
            missing += 1
            continue

        sliced, global_feat = extract_da3_features(
            model,
            images=images,
            process_res=int(cfg["process_res"]),
            export_layer=int(cfg["export_layer"]),
            batch_size=int(cfg["batch_size"]),
            process_res_method=str(cfg["process_res_method"]),
            ref_view_strategy=str(cfg["ref_view_strategy"]),
            output_grid_hw=output_grid_hw,
        )
        actual_feat_dim = int(sliced.shape[-1])
        if expected_feat_dim > 0 and actual_feat_dim != expected_feat_dim:
            raise RuntimeError(
                f"DA3 feature dim mismatch for {map_name}/{uuid}/frame={int(raw_frame_id):06d}: "
                f"actual={actual_feat_dim}, expected={expected_feat_dim}. "
                "For DA3-BASE expected dim is 768; for DA3-LARGE expected dim is 1024."
            )
        frame_dir.mkdir(parents=True, exist_ok=True)
        np.save(frame_dir / "da3_sliced.npy", sliced.astype(np.float16, copy=False))
        if bool(cfg.get("save_global", False)):
            np.save(frame_dir / "da3_global_feat.npy", global_feat)
        with (frame_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "map": map_name,
                    "uuid": uuid,
                    "raw_frame_id": int(raw_frame_id),
                    "image_count": int(len(images)),
                    "cameras": cameras,
                    "cache_granularity": "frame",
                    "export_feat_layers": [int(cfg["export_layer"])],
                    "process_res": int(cfg["process_res"]),
                    "process_res_method": str(cfg["process_res_method"]),
                    "ref_view_strategy": str(cfg["ref_view_strategy"]),
                    "feature_dim": int(global_feat.shape[0]),
                    "expected_feature_dim": int(expected_feat_dim),
                    "output_grid_hw": int(output_grid_hw),
                    "tokens_per_view": int(sliced.shape[1]),
                    "saved_sliced": True,
                    "saved_global": bool(cfg.get("save_global", False)),
                },
                f,
                indent=2,
            )
        cached += 1
        feature_dim = int(global_feat.shape[0])

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "frame_teacher_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "map": map_name,
                "uuid": uuid,
                "cache_granularity": "frame",
                "requested_raw_frame_count": int(raw_frame_ids.size),
                "cached": int(cached),
                "skipped": int(skipped),
                "missing": int(missing),
                "cameras": cameras,
                "feature_dim": feature_dim,
                "expected_feature_dim": int(expected_feat_dim),
            },
            f,
            indent=2,
        )
    if cached == 0 and skipped == 0:
        return "skip", f"no images for frames={raw_frame_ids.size}"
    return "ok", f"cached={cached} skipped={skipped} missing={missing} frames={raw_frame_ids.size} dim={feature_dim}"


def worker_main(worker_id: int, gpu_id: int, tasks: List[Dict[str, Any]], cfg: Dict[str, Any], queue: mp.Queue):
    fatal = None
    wds_reader: Optional[WebDatasetSampleReader] = None
    try:
        torch.set_num_threads(max(1, int(cfg.get("torch_num_threads", 1))))
        if gpu_id >= 0 and torch.cuda.is_available():
            device = f"cuda:{gpu_id}"
            torch.cuda.set_device(gpu_id)
        else:
            device = "cpu"
        print(f"[DA3][worker {worker_id}] gpu_id={gpu_id} device={device}", flush=True)
        model = load_da3_model(Path(cfg["da3_repo"]), Path(cfg["da3_checkpoint"]), device=device)
        wds_reader = WebDatasetSampleReader(max_open=int(cfg.get("wds_tar_cache", 4)))
        for task in tasks:
            try:
                status, detail = process_one_task(task, cfg, model, wds_reader=wds_reader)
                queue.put({"type": "result", "status": status, "detail": detail, "task": f"{task['map_name']}/{task['uuid']}", "worker": worker_id})
            except Exception as exc:
                queue.put({"type": "result", "status": "fail", "detail": f"{type(exc).__name__}: {exc}", "task": f"{task['map_name']}/{task['uuid']}", "worker": worker_id})
                if bool(cfg["strict"]):
                    fatal = traceback.format_exc()
                    break
    except Exception:
        fatal = traceback.format_exc()
    finally:
        if wds_reader is not None:
            wds_reader.close()
        queue.put({"type": "done", "worker": worker_id, "fatal": fatal})


def worker_file_main(worker_id: int, gpu_id: int, tasks: List[Dict[str, Any]], cfg: Dict[str, Any], result_file: Path) -> int:
    result_file.parent.mkdir(parents=True, exist_ok=True)
    wds_reader: Optional[WebDatasetSampleReader] = None

    def write_event(event: Dict[str, Any]) -> None:
        with result_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            f.flush()

    try:
        torch.set_num_threads(max(1, int(cfg.get("torch_num_threads", 1))))
        if gpu_id >= 0:
            gpu_binding = str(cfg.get("gpu_binding", "set_device"))
            if gpu_binding == "visible":
                
                
                device = "cuda:0"
            else:
                
                device = f"cuda:{gpu_id}"
        else:
            device = "cpu"
        model = load_da3_model(Path(cfg["da3_repo"]), Path(cfg["da3_checkpoint"]), device=device)
        wds_reader = WebDatasetSampleReader(max_open=int(cfg.get("wds_tar_cache", 4)))
        for task in tasks:
            task_name = f"{task['map_name']}/{task['uuid']}"
            try:
                status, detail = process_one_task(task, cfg, model, wds_reader=wds_reader)
                write_event({"type": "result", "status": status, "detail": detail, "task": task_name, "worker": worker_id})
            except Exception as exc:
                write_event({"type": "result", "status": "fail", "detail": f"{type(exc).__name__}: {exc}", "task": task_name, "worker": worker_id})
                if bool(cfg["strict"]):
                    write_event({"type": "fatal", "worker": worker_id, "fatal": traceback.format_exc()})
                    return 2
        write_event({"type": "done", "worker": worker_id})
        return 0
    except Exception:
        write_event({"type": "fatal", "worker": worker_id, "fatal": traceback.format_exc()})
        return 1
    finally:
        if wds_reader is not None:
            wds_reader.close()


def run_subprocess_workers(
    worker_specs: List[Tuple[int, int]],
    buckets: List[List[Dict[str, Any]]],
    cfg: Dict[str, Any],
    output_root: Path,
    strict: bool,
) -> None:
    tmp_root = output_root / f".da3_worker_tmp_{os.getpid()}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    cfg_path = tmp_root / "cfg.json"
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False)

    procs: List[Tuple[int, int, subprocess.Popen, Path]] = []
    for idx, (worker_id, gpu_id) in enumerate(worker_specs):
        chunk = buckets[idx]
        if not chunk:
            continue
        task_path = tmp_root / f"tasks_{worker_id:04d}.json"
        result_path = tmp_root / f"results_{worker_id:04d}.jsonl"
        with task_path.open("w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if gpu_id >= 0 and str(cfg.get("gpu_binding", "set_device")) == "visible":
            
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker_task_file",
            str(task_path),
            "--worker_config_file",
            str(cfg_path),
            "--worker_result_file",
            str(result_path),
            "--worker_id",
            str(worker_id),
            "--worker_gpu",
            str(gpu_id),
        ]
        proc = subprocess.Popen(cmd, env=env)
        procs.append((worker_id, gpu_id, proc, result_path))
        stagger = float(cfg.get("launch_stagger_seconds", 0.0) or 0.0)
        if stagger > 0:
            time.sleep(stagger)

    ok = 0
    skipped = 0
    failed = 0
    fatal: Dict[int, str] = {}
    offsets: Dict[Path, int] = {result_path: 0 for _, _, _, result_path in procs}
    seen_done: set[int] = set()
    pbar = tqdm(total=sum(len(bucket) for bucket in buckets), desc="DA3 cache", dynamic_ncols=True) if tqdm is not None else None

    def consume_result_file(result_path: Path) -> None:
        nonlocal ok, skipped, failed
        if not result_path.is_file():
            return
        with result_path.open("r", encoding="utf-8") as f:
            f.seek(offsets.get(result_path, 0))
            lines = f.readlines()
            offsets[result_path] = f.tell()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                continue
            msg_type = msg.get("type")
            if msg_type == "done":
                seen_done.add(int(msg.get("worker", -1)))
            elif msg_type == "fatal":
                fatal[int(msg.get("worker", -1))] = str(msg.get("fatal", ""))
            elif msg_type == "result":
                status = msg.get("status")
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skipped += 1
                else:
                    failed += 1
                    print(f"[DA3][FAIL][w{msg.get('worker')}] {msg.get('task')}: {msg.get('detail')}")
                if pbar is not None:
                    pbar.update(1)

    while True:
        for _, _, _, result_path in procs:
            consume_result_file(result_path)
        if all(proc.poll() is not None for _, _, proc, _ in procs):
            break
        time.sleep(0.5)

    for _, _, _, result_path in procs:
        consume_result_file(result_path)
    if pbar is not None:
        pbar.close()
    bad_codes = []
    for worker_id, gpu_id, proc, _ in procs:
        code = proc.wait()
        if code not in (0, None):
            bad_codes.append((worker_id, gpu_id, code))
    print(f"[DA3] done ok={ok} skipped={skipped} failed={failed} total={sum(len(bucket) for bucket in buckets)}")
    if fatal:
        first = sorted(fatal.keys())[0]
        raise RuntimeError(f"DA3 worker {first} fatal:\n{fatal[first]}")
    if bad_codes:
        raise RuntimeError(f"DA3 workers exited with non-zero codes: {bad_codes}")
    if bool(strict) and failed:
        raise RuntimeError(f"Strict mode failed={failed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", default=None)
    parser.add_argument("--output_root", default=str(WORKSPACE_ROOT / "TravelUAV_da3_large_joint_teacher"))
    parser.add_argument("--da3_repo", default=str(DEFAULT_DA3_REPO))
    parser.add_argument(
        "--da3_checkpoint",
        default=str(DEFAULT_DA3_CHECKPOINT),
    )
    parser.add_argument("--expected_feat_dim", type=int, default=0, help="0=auto from DA3 checkpoint config. DA3-BASE=768, DA3-LARGE=1024.")
    parser.add_argument("--data_json", default=None)
    parser.add_argument("--sample_frames", type=int, default=32)
    parser.add_argument("--cameras", default="frontcamera,downcamera")
    parser.add_argument("--process_res", type=int, default=252)
    parser.add_argument(
        "--output_grid_hw",
        type=int,
        default=8,
        help="Pool DA3 features to this square token grid before caching; 0 keeps the native grid.",
    )
    parser.add_argument("--process_res_method", default="upper_bound_resize")
    parser.add_argument("--ref_view_strategy", default="saddle_balanced")
    parser.add_argument("--export_layer", type=int, default=0, help="0=auto last export layer from checkpoint config")
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--save_global", action=argparse.BooleanOptionalAction, default=False, help="Optionally also save pooled da3_global_feat.npy for debugging.")
    parser.add_argument("--wds_tar_cache", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpus", default="", help="Comma GPU ids, e.g. 0,1,2,3. Overrides --device.")
    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=int(os.environ.get("DA3_WORKERS_PER_GPU", "4")),
        help="Parallel DA3 worker processes per GPU. Default 4; reduce if CUDA initialization or VRAM contention is unstable.",
    )
    parser.add_argument("--torch_num_threads", type=int, default=1)
    parser.add_argument("--limit_trajs", type=int, default=0)
    parser.add_argument("--task_shard_id", type=int, default=0, help="Shard id for external multi-process launch.")
    parser.add_argument("--task_shard_count", type=int, default=1, help="Total shard count for external multi-process launch.")
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--launcher", choices=["subprocess", "mp"], default="mp", help="mp uses spawn and a work queue; subprocess launches independent Python workers.")
    parser.add_argument("--gpu_binding", choices=["set_device", "visible"], default="set_device", help="set_device: keep inherited CUDA_VISIBLE_DEVICES and use cuda:<gpu_id>; visible: launch each worker with CUDA_VISIBLE_DEVICES=<gpu_id> and use cuda:0.")
    parser.add_argument("--launch_stagger_seconds", type=float, default=1.0, help="Sleep between worker launches to avoid simultaneous CUDA initialization.")
    parser.add_argument("--worker_task_file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_config_file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_result_file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_id", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--worker_gpu", type=int, default=-1, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_task_file:
        with Path(args.worker_task_file).open("r", encoding="utf-8") as f:
            worker_tasks = json.load(f)
        with Path(args.worker_config_file).open("r", encoding="utf-8") as f:
            worker_cfg = json.load(f)
        raise SystemExit(
            worker_file_main(
                worker_id=int(args.worker_id),
                gpu_id=int(args.worker_gpu),
                tasks=worker_tasks,
                cfg=worker_cfg,
                result_file=Path(args.worker_result_file),
            )
        )

    if not args.input_root:
        raise RuntimeError("--input_root is required.")

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    data_json = Path(args.data_json).expanduser().resolve() if args.data_json else None
    da3_repo = Path(args.da3_repo).expanduser().resolve()
    da3_checkpoint = Path(args.da3_checkpoint).expanduser().resolve()
    cameras = [x.strip() for x in str(args.cameras).split(",") if x.strip()]
    wds_index = input_root / "dataset_index.json"
    wds_samples = scan_webdataset_members(wds_index) if wds_index.is_file() else None
    export_layer = resolve_export_layer(da3_checkpoint, int(args.export_layer))
    expected_feat_dim = int(args.expected_feat_dim) if int(args.expected_feat_dim) > 0 else infer_da3_token_dim(da3_checkpoint)
    requested_sparse_frames = collect_requested_sparse_frames(data_json)

    output_root.mkdir(parents=True, exist_ok=True)
    iterator = requested_sparse_frames.keys() if requested_sparse_frames else iter_trajectories(input_root, data_json=data_json)
    if wds_samples is not None and data_json is None:
        iterator = iter_webdataset_keys(wds_samples)
    tasks: List[Dict[str, Any]] = []
    for map_name, uuid in iterator:
        task = {
            "map_name": map_name,
            "uuid": uuid,
            "wds_entry": None,
            "sparse_frames": requested_sparse_frames.get((map_name, uuid), []),
        }
        if wds_samples is not None:
            entry = wds_samples.get(f"{map_name}__{uuid}")
            if entry is None:
                continue
            task["wds_entry"] = entry
        tasks.append(task)

    shard_count = max(1, int(args.task_shard_count))
    shard_id = int(args.task_shard_id)
    if shard_id < 0 or shard_id >= shard_count:
        raise RuntimeError(f"--task_shard_id must be in [0, {shard_count}), got {shard_id}.")
    if shard_count > 1:
        tasks = [task for idx, task in enumerate(tasks) if idx % shard_count == shard_id]
    if int(args.limit_trajs) > 0:
        tasks = tasks[: int(args.limit_trajs)]

    if not tasks:
        raise RuntimeError("No DA3 cache tasks found.")

    gpu_ids: List[int] = []
    for part in str(args.gpus).split(","):
        part = part.strip()
        if part:
            gpu_ids.append(int(part))
    if not gpu_ids:
        if str(args.device).startswith("cuda"):
            if ":" in str(args.device):
                gpu_ids = [int(str(args.device).split(":", 1)[1])]
            else:
                gpu_ids = [0]
        else:
            gpu_ids = [-1]

    worker_specs: List[Tuple[int, int]] = []
    wid = 0
    for gpu_id in gpu_ids:
        for _ in range(max(1, int(args.workers_per_gpu))):
            worker_specs.append((wid, gpu_id))
            wid += 1

    buckets: List[List[Dict[str, Any]]] = [[] for _ in worker_specs]
    for idx, task in enumerate(tasks):
        buckets[idx % len(buckets)].append(task)

    cfg = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "da3_repo": str(da3_repo),
        "da3_checkpoint": str(da3_checkpoint),
        "sample_frames": int(args.sample_frames),
        "cameras": cameras,
        "process_res": int(args.process_res),
        "output_grid_hw": int(args.output_grid_hw),
        "process_res_method": str(args.process_res_method),
        "ref_view_strategy": str(args.ref_view_strategy),
        "export_layer": int(export_layer),
        "expected_feat_dim": int(expected_feat_dim),
        "batch_size": max(1, int(args.batch_size)),
        "save_global": bool(args.save_global),
        "wds_tar_cache": max(1, int(args.wds_tar_cache)),
        "overwrite": bool(args.overwrite),
        "strict": bool(args.strict),
        "torch_num_threads": max(1, int(args.torch_num_threads)),
        "gpu_binding": str(args.gpu_binding),
        "launch_stagger_seconds": float(args.launch_stagger_seconds),
    }
    print(f"[DA3] checkpoint={da3_checkpoint} expected_token_dim={expected_feat_dim}")
    print(f"[DA3] tasks={len(tasks)} shard={shard_id}/{shard_count} gpus={gpu_ids} workers={len(worker_specs)} workers/gpu={max(1, int(args.workers_per_gpu))} launcher={args.launcher} gpu_binding={args.gpu_binding} stagger={args.launch_stagger_seconds}s batch={cfg['batch_size']} process_res={cfg['process_res']} ref={cfg['ref_view_strategy']} layer={export_layer}")

    if len(worker_specs) == 1:
        device = f"cuda:{worker_specs[0][1]}" if worker_specs[0][1] >= 0 else "cpu"
        model = load_da3_model(da3_repo, da3_checkpoint, device=device)
        wds_reader = WebDatasetSampleReader(max_open=cfg["wds_tar_cache"]) if wds_samples is not None else None
        iterator_tasks = tasks if tqdm is None else tqdm(tasks, desc="DA3 cache", dynamic_ncols=True)
        for task in iterator_tasks:
            status, detail = process_one_task(task, cfg, model, wds_reader=wds_reader)
            if status == "ok":
                print(f"[DA3] cached {task['map_name']}/{task['uuid']}: {detail}")
        if wds_reader is not None:
            wds_reader.close()
        return

    if str(args.launcher) == "subprocess":
        run_subprocess_workers(
            worker_specs=worker_specs,
            buckets=buckets,
            cfg=cfg,
            output_root=output_root,
            strict=bool(args.strict),
        )
        return

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue(maxsize=4096)
    procs: List[mp.Process] = []
    for idx, (worker_id, gpu_id) in enumerate(worker_specs):
        chunk = buckets[idx]
        if not chunk:
            continue
        proc = ctx.Process(target=worker_main, args=(worker_id, gpu_id, chunk, cfg, queue), daemon=False)
        proc.start()
        procs.append(proc)

    done = 0
    ok = 0
    skipped = 0
    failed = 0
    fatal = {}
    pbar = tqdm(total=len(tasks), desc="DA3 cache", dynamic_ncols=True) if tqdm is not None else None
    while done < len(procs):
        msg = queue.get()
        if msg.get("type") == "done":
            done += 1
            if msg.get("fatal"):
                fatal[msg.get("worker")] = msg.get("fatal")
            continue
        if msg.get("type") == "result":
            status = msg.get("status")
            if status == "ok":
                ok += 1
            elif status == "skip":
                skipped += 1
            else:
                failed += 1
                print(f"[DA3][FAIL][w{msg.get('worker')}] {msg.get('task')}: {msg.get('detail')}")
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()
    for proc in procs:
        proc.join()
    bad_codes = [proc.exitcode for proc in procs if proc.exitcode not in (0, None)]
    print(f"[DA3] done ok={ok} skipped={skipped} failed={failed} total={len(tasks)}")
    if bad_codes:
        raise RuntimeError(f"DA3 workers exited with non-zero codes: {bad_codes}")
    if fatal:
        first = sorted(fatal.keys())[0]
        raise RuntimeError(f"DA3 worker {first} fatal:\n{fatal[first]}")
    if bool(args.strict) and failed:
        raise RuntimeError(f"Strict mode failed={failed}")


if __name__ == "__main__":
    main()
