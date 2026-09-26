import copy
import io
import json
import logging
import math
import os
import tarfile
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import transformers

from .constants import DEFAULT_IMAGE_TOKEN, DEFAULT_STOP_TOKEN, DEFAULT_TRAJ_TOKEN, IGNORE_INDEX
from .geometry import rotation_matrix_from_vector, transform_point
from .preprocess_qwen import preprocess_qwen_visual
from .rope2d import get_rope_index_3
from .prompts import observation_prompt, trajectory_prompt, STOP_REPLY, TRAJECTORY_REPLY
from ..paths import ASSETS_ROOT


logger = logging.getLogger(__name__)


class GeoRiskDataset(Dataset):
    DEFAULT_TRAJ_HORIZON = 10
    RGB_FOLDERS = ["frontcamera", "leftcamera", "rightcamera", "rearcamera", "downcamera"]
    CAMERA_INDEX_TO_NAME = {
        0: "frontcamera",
        1: "leftcamera",
        2: "rightcamera",
        3: "rearcamera",
        4: "downcamera",
    }
    VIEW_MODE_TO_CAMERAS = {
        "dual": ["frontcamera", "downcamera"],
    }
    DEFAULT_LITE_CAMERA_ORDER = ["frontcamera", "downcamera"]
    NPZ_RGB_KEY_CANDIDATES = ("imgs", "rgb")
    NPZ_CANDIDATE_FILENAMES = ("rgb_imgs_uint8_lite.npz", "rgb_imgs_uint8.npz")
    CAMERA_EXTRINSICS = {
        "frontcamera": {
            "t_cam_to_body": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "r_cam_to_body": np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
        },
        "downcamera": {
            "t_cam_to_body": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "r_cam_to_body": np.array(
                [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                dtype=np.float64,
            ),
        },
    }

    def __init__(
        self,
        data_path: str,
        dataset_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        image_processor,
        max_samples: int = None,
        view_mode: str = "dual",
        stop_terminal_repeat: int = 3,
        stop_soft_r: float = 20.0,
        stop_soft_tau: float = 5.0,
        stop_label_clip_eps: float = 1e-4,
        stop_phase2_disable_dist: Optional[float] = None,
        stop_near_resample_rules: str = "0,5,3;5,10,2;10,20,2",
        dense_dataset_path: Optional[str] = None,
        traj_horizon: int = DEFAULT_TRAJ_HORIZON,
        depth_dataset_path: Optional[str] = None,
        use_clearance_supervision: bool = False,
        clearance_history_frames: int = 3,
        clearance_bad_depth_maps: str = "BrushifyCountryRoads,NordicHarbour",
        da3_joint_teacher_path: Optional[str] = None,
        use_da3_joint_supervision: bool = False,
        strict_da3_joint_cache: bool = True,
        da3_joint_teacher_dim: int = 1024,
    ):
        self.dataset_path = os.path.abspath(os.path.expanduser(dataset_path))
        self.data_path = os.path.abspath(os.path.expanduser(data_path))
        if dense_dataset_path is None:
            dataset_root = self.dataset_path
            if os.path.isfile(dataset_root):
                dataset_root = os.path.dirname(dataset_root)
            dense_dataset_path = os.path.join(
                os.path.dirname(os.path.abspath(dataset_root)),
                "TravelUAV_original_decompressed_merged_all",
            )
        self.dense_dataset_path = os.path.abspath(os.path.expanduser(dense_dataset_path))
        self.traj_horizon = max(1, int(traj_horizon))
        if depth_dataset_path is None:
            depth_dataset_path = str(ASSETS_ROOT / "TravelUAV_depth_trainset")
        self.depth_dataset_path = os.path.abspath(os.path.expanduser(depth_dataset_path))
        self.use_clearance_supervision = bool(use_clearance_supervision)
        self.clearance_depth_cameras = ["frontcamera", "downcamera"]
        self.clearance_history_frames = max(1, int(clearance_history_frames))
        self.clearance_frame_offsets = list(range(-(self.clearance_history_frames - 1), 1))
        self.clearance_bad_depth_maps = {
            part.strip()
            for part in str(clearance_bad_depth_maps or "").split(",")
            if part.strip()
        }
        self._depth_sidecar_cache: OrderedDict[str, Optional[Dict[str, Any]]] = OrderedDict()
        
        
        self._depth_cache_size = self._env_cache_size("GEORISK_DEPTH_CACHE_SIZE", default=32)
        if da3_joint_teacher_path is None:
            da3_joint_teacher_path = str(ASSETS_ROOT / "TravelUAV_da3_large_joint_teacher")
        self.da3_joint_teacher_path = os.path.abspath(os.path.expanduser(da3_joint_teacher_path))
        self.use_da3_joint_supervision = bool(use_da3_joint_supervision)
        self.strict_da3_joint_cache = bool(strict_da3_joint_cache)
        self.da3_joint_teacher_dim = max(1, int(da3_joint_teacher_dim))
        if self.use_da3_joint_supervision and self.strict_da3_joint_cache and not os.path.isdir(
            self.da3_joint_teacher_path
        ):
            raise FileNotFoundError(f"DA3 joint teacher cache root not found: {self.da3_joint_teacher_path}")
        self._da3_joint_cache: OrderedDict[str, Optional[Dict[str, Any]]] = OrderedDict()
        self._da3_joint_cache_size = self._env_cache_size("GEORISK_DA3_JOINT_CACHE_SIZE", default=128)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.view_mode = (view_mode or "dual").strip().lower()
        if self.view_mode not in self.VIEW_MODE_TO_CAMERAS:
            raise ValueError(
                f"Unsupported view_mode '{self.view_mode}'. Use one of: {sorted(self.VIEW_MODE_TO_CAMERAS.keys())}"
            )
        self.selected_folders = list(self.VIEW_MODE_TO_CAMERAS[self.view_mode])
        if self.use_clearance_supervision and self.selected_folders != self.clearance_depth_cameras:
            raise ValueError(
                "Clearance supervision in GeoRisk requires --view_mode dual "
                f"with cameras {self.clearance_depth_cameras}; got view_mode={self.view_mode} "
                f"with cameras {self.selected_folders}."
            )
        self.stop_terminal_repeat = max(1, int(stop_terminal_repeat))
        self.stop_soft_r = float(stop_soft_r)
        self.stop_soft_tau = max(1e-6, float(stop_soft_tau))
        self.stop_label_clip_eps = min(max(float(stop_label_clip_eps), 0.0), 0.49)
        if stop_phase2_disable_dist is None:
            self.stop_phase2_disable_dist = float(self.stop_soft_r)
        else:
            self.stop_phase2_disable_dist = float(stop_phase2_disable_dist)
        self.stop_near_resample_rules = self._parse_stop_near_resample_rules(stop_near_resample_rules)
        self.stop_token_id = None
        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        if DEFAULT_STOP_TOKEN in vocab:
            self.stop_token_id = int(tokenizer.convert_tokens_to_ids(DEFAULT_STOP_TOKEN))
        else:
            logger.warning("Token %s is not in tokenizer vocab; stop label masking will be skipped.", DEFAULT_STOP_TOKEN)

        
        self.dataset_backend = "filesystem"
        self.dataset_variant = "unknown"
        self._wds_index_path: Optional[str] = None
        self._wds_shards: List[str] = []
        self._wds_samples: Dict[str, Dict[str, str]] = {}
        self._wds_sample_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._wds_cache_size = self._env_cache_size("GEORISK_WDS_CACHE_SIZE", default=16)
        self._warned_missing_lite_camera_metadata = False
        self._init_dataset_backend(data_path=self.data_path)

        with open(self.data_path, "r", encoding="utf-8") as f:
            data_obj = json.load(f)
        list_data_dict = self._resolve_data_entries(data_obj, max_samples=max_samples)
        if max_samples is not None:
            list_data_dict = list_data_dict[: max(0, max_samples)]
        list_data_dict = self._apply_stop_resampling(list_data_dict)
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    @staticmethod
    def _env_cache_size(name: str, default: int) -> int:
        raw = os.environ.get(name, str(default))
        try:
            value = int(raw)
        except Exception:
            logger.warning("Invalid %s=%r; fallback to %d.", name, raw, int(default))
            value = int(default)
        return max(0, min(1024, int(value)))

    @staticmethod
    def _parse_stop_near_resample_rules(rules_text: str) -> List[Tuple[float, float, int]]:
        rules: List[Tuple[float, float, int]] = []
        if rules_text is None:
            return rules
        for chunk in str(rules_text).split(";"):
            chunk = chunk.strip()
            if chunk == "":
                continue
            parts = [p.strip() for p in chunk.split(",")]
            if len(parts) != 3:
                logger.warning("Ignore invalid stop_near_resample_rules segment '%s' (expect lo,hi,repeat).", chunk)
                continue
            try:
                lo = float(parts[0])
                hi = float(parts[1])
                repeat = int(parts[2])
            except Exception:
                logger.warning("Ignore invalid stop_near_resample_rules segment '%s' (parse failed).", chunk)
                continue
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                logger.warning("Ignore invalid stop_near_resample_rules segment '%s' (bad range).", chunk)
                continue
            if repeat < 1:
                logger.warning("Ignore invalid stop_near_resample_rules segment '%s' (repeat<1).", chunk)
                continue
            rules.append((float(lo), float(hi), int(repeat)))
        rules.sort(key=lambda x: (x[0], x[1]))
        return rules

    def _trajectory_from_json_rel(self, json_rel: str) -> Optional[np.ndarray]:
        if self.dataset_backend == "webdataset":
            sample_key = self._json_rel_to_sample_key(json_rel)
            sample = self._load_webdataset_sample(sample_key, need_npz=False)
            merged_data = sample.get("merged_data", {})
        else:
            json_path = os.path.join(self.dataset_path, str(json_rel).replace("/", os.sep))
            merged_data = self._load_json(json_path)
        traj = np.asarray(merged_data.get("trajectory", []), dtype=np.float32)
        if traj.ndim != 2 or traj.shape[0] <= 0 or traj.shape[1] < 3:
            return None
        return traj

    def _repeat_from_distance_rules(self, distance_to_goal: float) -> int:
        d = float(distance_to_goal)
        for lo, hi, repeat in self.stop_near_resample_rules:
            if lo <= d < hi:
                return int(repeat)
        return 1

    def _apply_stop_resampling(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(entries) == 0:
            return entries
        has_near_rules = len(self.stop_near_resample_rules) > 0
        if self.stop_terminal_repeat <= 1 and not has_near_rules:
            return entries

        traj_cache: "OrderedDict[str, Optional[np.ndarray]]" = OrderedDict()
        traj_cache_size = self._env_cache_size("GEORISK_RESAMPLE_TRAJ_CACHE_SIZE", default=256)
        expanded: List[Dict[str, Any]] = []
        for item in entries:
            expanded.append(item)
            if not isinstance(item, dict):
                continue
            json_rel = str(item.get("json", ""))
            frame = int(item.get("frame", 0) or 0)
            if json_rel == "" or frame <= 0:
                continue

            traj = traj_cache.get(json_rel, None)
            if json_rel not in traj_cache:
                try:
                    traj = self._trajectory_from_json_rel(json_rel)
                except Exception:
                    traj = None
                if traj_cache_size > 0:
                    traj_cache[json_rel] = traj
                    traj_cache.move_to_end(json_rel)
                    while len(traj_cache) > traj_cache_size:
                        traj_cache.popitem(last=False)
            elif traj_cache_size > 0:
                traj_cache.move_to_end(json_rel)
            if traj is None:
                continue
            total_frames = int(len(traj))
            frame_idx = max(1, min(frame, total_frames))
            cur = traj[frame_idx - 1, :3]
            goal = traj[-1, :3]
            distance_to_goal = float(np.linalg.norm(cur - goal))

            repeat_near = self._repeat_from_distance_rules(distance_to_goal) if has_near_rules else 1
            repeat_terminal = self.stop_terminal_repeat if frame_idx == total_frames else 1
            repeat_count = max(int(repeat_near), int(repeat_terminal))

            for _ in range(max(0, repeat_count - 1)):
                expanded.append(dict(item))
        return expanded

    @staticmethod
    @lru_cache(maxsize=1024)
    def _load_json(path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_dense_trajectory(self, json_rel: str) -> np.ndarray:
        rel_parts = str(json_rel).replace("\\", "/").strip("/").split("/")
        if len(rel_parts) < 3:
            raise ValueError(f"Invalid trajectory json path for dense lookup: {json_rel}")
        dense_json_path = os.path.join(self.dense_dataset_path, *rel_parts[:-1], "merged_data_all.json")
        if not os.path.isfile(dense_json_path):
            raise FileNotFoundError(
                f"Dense trajectory file not found: {dense_json_path}. "
                "Set --dense_dataset_path to TravelUAV_original_decompressed_merged_all."
            )
        dense_data = self._load_json(dense_json_path)
        traj = dense_data.get("trajectory_all", None)
        if traj is None:
            traj = dense_data.get("trajectory", None)
        dense_traj = np.asarray(traj, dtype=np.float32)
        if dense_traj.ndim != 2 or dense_traj.shape[0] <= 0 or dense_traj.shape[1] < 3:
            raise ValueError(f"Invalid dense trajectory in {dense_json_path}: shape={dense_traj.shape}")
        return dense_traj

    @staticmethod
    def _resolve_dense_index(merged_data: Dict[str, Any], frame_idx: int, dense_len: int) -> int:
        indices = merged_data.get("index", None)
        if isinstance(indices, list) and len(indices) > 0:
            sparse_pos = max(0, min(int(frame_idx) - 1, len(indices) - 1))
            dense_idx = int(indices[sparse_pos])
        else:
            dense_idx = int(frame_idx) - 1
        return max(0, min(dense_idx, max(0, int(dense_len) - 1)))

    def _build_dense_target(
        self,
        dense_traj: np.ndarray,
        dense_idx: int,
        rotation_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        cur = np.asarray(dense_traj[dense_idx, :3], dtype=np.float32)

        target = np.zeros((self.traj_horizon, 3), dtype=np.float32)
        valid = np.zeros((self.traj_horizon,), dtype=np.float32)
        for out_idx in range(self.traj_horizon):
            src_idx = int(dense_idx) + out_idx + 1
            if 0 <= src_idx < len(dense_traj):
                target[out_idx] = np.asarray(dense_traj[src_idx, :3], dtype=np.float32) - cur
                valid[out_idx] = 1.0

        target = transform_point(target, rotation_matrix).astype(np.float32)
        return target, valid

    @staticmethod
    def _resolve_dataset_index_path(dataset_path: str) -> Optional[str]:
        if os.path.isfile(dataset_path):
            if os.path.basename(dataset_path).lower() == "dataset_index.json":
                return os.path.abspath(dataset_path)
            return None
        candidate = os.path.join(dataset_path, "dataset_index.json")
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        return None

    @staticmethod
    def _resolve_dataset_root(dataset_path: str, index_path: Optional[str]) -> str:
        if index_path is not None and os.path.abspath(dataset_path) == os.path.abspath(index_path):
            return os.path.dirname(index_path)
        return dataset_path

    @staticmethod
    def _scan_webdataset_members(shard_paths: List[str]) -> Dict[str, Dict[str, str]]:
        sample_map: Dict[str, Dict[str, str]] = {}
        for shard_path in shard_paths:
            with tarfile.open(shard_path, "r") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    name = member.name
                    if not (name.endswith(".json") or name.endswith(".npz")):
                        continue
                    key = name.rsplit(".", 1)[0]
                    if key not in sample_map:
                        sample_map[key] = {
                            "shard_path": shard_path,
                            "json_member": "",
                            "npz_member": "",
                        }
                    entry = sample_map[key]
                    if entry["shard_path"] != shard_path:
                        continue
                    if name.endswith(".json"):
                        entry["json_member"] = name
                    elif name.endswith(".npz"):
                        entry["npz_member"] = name
        return {k: v for k, v in sample_map.items() if v.get("json_member", "")}

    def _init_dataset_backend(self, data_path: str):
        index_path_from_dataset = self._resolve_dataset_index_path(self.dataset_path)
        index_path_from_data = self._resolve_dataset_index_path(data_path)
        index_path = index_path_from_dataset or index_path_from_data
        if index_path is None:
            self.dataset_backend = "filesystem"
            return

        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        shards = index_data.get("shards", [])
        if not isinstance(shards, list) or len(shards) == 0:
            raise ValueError(f"Invalid dataset_index.json: missing non-empty `shards` in {index_path}")

        root_dir = os.path.dirname(index_path)
        resolved_shards = []
        for shard in shards:
            shard_path = str(shard)
            if not os.path.isabs(shard_path):
                shard_path = os.path.join(root_dir, shard_path)
            shard_path = os.path.abspath(shard_path)
            if not os.path.exists(shard_path):
                raise FileNotFoundError(f"WebDataset shard not found: {shard_path}")
            resolved_shards.append(shard_path)

        self.dataset_backend = "webdataset"
        self.dataset_path = self._resolve_dataset_root(self.dataset_path, index_path_from_dataset)
        self.dataset_variant = str(index_data.get("dataset_variant", "unknown")).strip().lower()
        self._wds_index_path = index_path
        self._wds_shards = resolved_shards
        self._wds_samples = self._scan_webdataset_members(self._wds_shards)

    def _resolve_data_entries(self, data_obj: Union[List[Dict[str, Any]], Dict[str, Any]], max_samples: Optional[int]):
        if isinstance(data_obj, list):
            return data_obj
        if not isinstance(data_obj, dict):
            raise ValueError(
                f"Unsupported data json format in data_path: expect list/dict, got {type(data_obj).__name__}."
            )
        if isinstance(data_obj.get("samples"), list):
            return data_obj["samples"]
        if isinstance(data_obj.get("data"), list):
            return data_obj["data"]
        if self.dataset_backend == "webdataset" and isinstance(data_obj.get("shards"), list):
            return self._build_frame_entries_from_webdataset(max_samples=max_samples)
        raise ValueError(
            "Unsupported data json structure. Expected a frame-level list like "
            "[{\"json\": \".../merged_data.json\", \"frame\": n}, ...], "
            "or a dict containing `samples`/`data`."
        )

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _extract_webdataset_merged_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if isinstance(payload, dict) and "merged_data" in payload:
            return payload["merged_data"], payload.get("_meta", {})
        return payload, payload.get("_meta", {}) if isinstance(payload, dict) else ({}, {})

    def _load_webdataset_sample(self, sample_key: str, need_npz: bool = True) -> Dict[str, Any]:
        cached = self._wds_sample_cache.get(sample_key, None)
        if cached is not None and (not need_npz or cached.get("npz_bytes") is not None):
            self._wds_sample_cache.move_to_end(sample_key)
            return cached

        entry = self._wds_samples.get(sample_key, None)
        if entry is None:
            raise KeyError(f"Sample key not found in webdataset index: {sample_key}")

        with tarfile.open(entry["shard_path"], "r") as tar:
            json_member = entry.get("json_member", "")
            npz_member = entry.get("npz_member", "")
            if not json_member:
                raise KeyError(f"Sample {sample_key} missing json member in shard {entry['shard_path']}")
            json_file = tar.extractfile(json_member)
            if json_file is None:
                raise KeyError(f"Cannot extract {json_member} from {entry['shard_path']}")
            payload = json.loads(json_file.read().decode("utf-8"))
            npz_bytes = None
            if need_npz and npz_member:
                npz_file = tar.extractfile(npz_member)
                if npz_file is not None:
                    npz_bytes = npz_file.read()

        merged_data, meta = self._extract_webdataset_merged_payload(payload)
        sample = {
            "merged_data": merged_data,
            "meta": meta,
            "npz_bytes": npz_bytes,
        }
        self._wds_sample_cache[sample_key] = sample
        self._wds_sample_cache.move_to_end(sample_key)
        while len(self._wds_sample_cache) > self._wds_cache_size:
            self._wds_sample_cache.popitem(last=False)
        return sample

    def _build_frame_entries_from_webdataset(self, max_samples: Optional[int]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for sample_key in sorted(self._wds_samples.keys()):
            sample = self._load_webdataset_sample(sample_key, need_npz=False)
            merged_data = sample["merged_data"]
            meta = sample["meta"]

            length = self._safe_int(merged_data.get("length", 0), default=0)
            if length <= 0 and isinstance(merged_data.get("index"), list):
                length = len(merged_data["index"])
            if length <= 0 and isinstance(merged_data.get("trajectory"), list):
                length = len(merged_data["trajectory"])
            if length <= 0:
                continue

            if "__" in sample_key:
                map_name, uuid = sample_key.split("__", 1)
            else:
                map_name, uuid = "unknown_map", sample_key
            map_name = str(meta.get("map", map_name))
            uuid = str(meta.get("uuid", uuid))
            json_rel = f"{map_name}/{uuid}/merged_data.json"

            for frame_num in range(1, length + 1):
                entries.append({"json": json_rel, "frame": frame_num})
                if max_samples is not None and len(entries) >= max_samples:
                    return entries
        return entries

    def _json_rel_to_sample_key(self, json_rel: str) -> str:
        rel = str(json_rel).replace("\\", "/").strip("/")
        parts = rel.split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid `json` path in data item: {json_rel}")

        candidates: List[str] = []
        if len(parts) >= 3:
            candidates.append(f"{parts[-3]}__{parts[-2]}")
        candidates.append(f"{parts[0]}__{parts[1]}")
        for idx in range(len(parts) - 1):
            candidates.append(f"{parts[idx]}__{parts[idx + 1]}")

        seen = set()
        ordered_candidates = []
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                ordered_candidates.append(candidate)
        for candidate in ordered_candidates:
            if candidate in self._wds_samples:
                return candidate
        return ordered_candidates[0]

    @staticmethod
    def _normalize_camera_name(name: Any) -> Optional[str]:
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")
        name = str(name).strip().lower()
        if name == "":
            return None
        if "front" in name:
            return "frontcamera"
        if "rear" in name or "back" in name:
            return "rearcamera"
        if "left" in name:
            return "leftcamera"
        if "right" in name:
            return "rightcamera"
        if "down" in name or "bottom" in name:
            return "downcamera"
        return None

    @staticmethod
    def _parse_env_int_list(text: str) -> List[int]:
        values: List[int] = []
        for part in str(text).split(","):
            part = part.strip()
            if part == "":
                continue
            try:
                values.append(int(part))
            except Exception:
                continue
        return values

    def _load_rgb_tensor_from_npz(
        self,
        npz_source: Union[str, bytes, bytearray],
        npz_desc: str,
    ) -> Tuple[np.ndarray, Optional[List[int]], Dict[str, int]]:
        if isinstance(npz_source, (bytes, bytearray)):
            npz_obj = np.load(io.BytesIO(npz_source), allow_pickle=False)
        else:
            npz_obj = np.load(npz_source, allow_pickle=False)

        with npz_obj as data:
            rgb_key = None
            for key in self.NPZ_RGB_KEY_CANDIDATES:
                if key in data:
                    rgb_key = key
                    break
            if rgb_key is None:
                raise KeyError(
                    f"No RGB key {self.NPZ_RGB_KEY_CANDIDATES} found in {npz_desc}. Existing keys: {list(data.files)}"
                )
            rgb = data[rgb_key]
            frame_ids = data["frame_ids"].tolist() if "frame_ids" in data else None

            camera_pos_map: Dict[str, int] = {}
            if "camera_names" in data:
                camera_names = data["camera_names"].tolist()
                if not isinstance(camera_names, list):
                    camera_names = [camera_names]
                for pos, raw_name in enumerate(camera_names):
                    name = self._normalize_camera_name(raw_name)
                    if name is not None:
                        camera_pos_map.setdefault(name, pos)
            if not camera_pos_map and "camera_indices" in data:
                camera_indices = data["camera_indices"].tolist()
                if not isinstance(camera_indices, list):
                    camera_indices = [camera_indices]
                for pos, full_idx in enumerate(camera_indices):
                    full_idx = self._safe_int(full_idx, default=-1)
                    name = self.CAMERA_INDEX_TO_NAME.get(full_idx, None)
                    if name is not None:
                        camera_pos_map.setdefault(name, pos)

        if rgb.ndim != 5:
            raise ValueError(f"Unexpected RGB tensor rank in {npz_desc}: expect 5D, got shape={rgb.shape}")
        if rgb.shape[-1] != 3:
            raise ValueError(f"Unexpected RGB channel count in {npz_desc}: expect last dim=3, got shape={rgb.shape}")

        if not camera_pos_map:
            cam_count = int(rgb.shape[1])
            if cam_count >= len(self.RGB_FOLDERS):
                camera_pos_map = {name: idx for idx, name in enumerate(self.RGB_FOLDERS)}
            elif cam_count == 2:
                env_idx = os.environ.get("GEORISK_LITE_CAMERA_INDICES", "").strip()
                parsed_indices = self._parse_env_int_list(env_idx)
                if len(parsed_indices) >= 2:
                    for pos, idx in enumerate(parsed_indices[:cam_count]):
                        name = self.CAMERA_INDEX_TO_NAME.get(int(idx), None)
                        if name is not None:
                            camera_pos_map.setdefault(name, pos)
                    if camera_pos_map:
                        logger.debug(
                            "Resolved lite camera order from GEORISK_LITE_CAMERA_INDICES=%s for %s: %s",
                            env_idx,
                            npz_desc,
                            camera_pos_map,
                        )

                if not camera_pos_map:
                    env_order = os.environ.get("GEORISK_LITE_CAMERA_ORDER", "").strip()
                    if env_order:
                        for pos, raw_name in enumerate(env_order.split(",")[:cam_count]):
                            name = self._normalize_camera_name(raw_name)
                            if name is not None:
                                camera_pos_map.setdefault(name, pos)
                        if camera_pos_map:
                            logger.debug(
                                "Resolved lite camera order from GEORISK_LITE_CAMERA_ORDER=%s for %s: %s",
                                env_order,
                                npz_desc,
                                camera_pos_map,
                            )

                if not camera_pos_map:
                    camera_pos_map = {name: idx for idx, name in enumerate(self.DEFAULT_LITE_CAMERA_ORDER)}
                    if not self._warned_missing_lite_camera_metadata:
                        logger.warning(
                            "No camera metadata found in lite npz %s. Falling back to default order front,down. "
                            "If your lite order is different, set GEORISK_LITE_CAMERA_INDICES (e.g. 4,0) "
                            "or GEORISK_LITE_CAMERA_ORDER (e.g. down,front).",
                            npz_desc,
                        )
                        self._warned_missing_lite_camera_metadata = True
            elif cam_count == 1:
                camera_pos_map = {"frontcamera": 0}
            else:
                camera_pos_map = {self.RGB_FOLDERS[idx]: idx for idx in range(cam_count)}

        return rgb, frame_ids, camera_pos_map

    @staticmethod
    def _split_json_rel(json_rel: str) -> Tuple[str, str]:
        parts = str(json_rel).replace("\\", "/").strip("/").split("/")
        if len(parts) >= 3:
            return parts[-3], parts[-2]
        if len(parts) >= 2:
            return parts[0], parts[1]
        return "", ""

    @staticmethod
    def _pose_to_matrix(pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float32).reshape(-1)
        mat = np.eye(4, dtype=np.float32)
        if pose.shape[0] < 7:
            return mat
        x, y, z, qx, qy, qz, qw = [float(v) for v in pose[:7]]
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm < 1e-8:
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        else:
            qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wz = qw * qx, qw * qy, qw * qz
        mat[:3, :3] = np.asarray(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float32,
        )
        mat[:3, 3] = np.asarray([x, y, z], dtype=np.float32)
        return mat

    @staticmethod
    def _pose_from_raw_record(record: Any) -> Optional[np.ndarray]:
        if not isinstance(record, dict):
            return None
        pos = record.get("position", record.get("pos", None))
        quat = record.get("quaternion", record.get("orientation", record.get("rotation", None)))
        if isinstance(pos, dict):
            pos = [pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)]
        if isinstance(quat, dict):
            quat = [
                quat.get("x", 0.0),
                quat.get("y", 0.0),
                quat.get("z", 0.0),
                quat.get("w", 1.0),
            ]
        if pos is None:
            return None
        if quat is None:
            quat = [0.0, 0.0, 0.0, 1.0]
        try:
            pose = np.asarray(list(pos)[:3] + list(quat)[:4], dtype=np.float32)
        except Exception:
            return None
        if pose.shape[0] != 7 or not np.all(np.isfinite(pose)):
            return None
        return pose

    @staticmethod
    def _pose_from_trajectory(
        trajectory: np.ndarray,
        frame_idx: int,
        merged_data: Optional[Dict[str, Any]] = None,
        raw_frame_id: Optional[int] = None,
    ) -> np.ndarray:
        if merged_data is not None and raw_frame_id is not None:
            detailed = merged_data.get("trajectory_raw_detailed", None)
            if isinstance(detailed, dict):
                for key in (str(int(raw_frame_id)), f"{int(raw_frame_id):06d}"):
                    pose = GeoRiskDataset._pose_from_raw_record(detailed.get(key))
                    if pose is not None:
                        return pose
        sparse_zero = max(0, int(frame_idx) - 1)
        if merged_data is not None:
            sparse_raw = merged_data.get("trajectory_raw", None)
            if isinstance(sparse_raw, list) and len(sparse_raw) > sparse_zero:
                pose = GeoRiskDataset._pose_from_raw_record(sparse_raw[sparse_zero])
                if pose is not None:
                    return pose
        pose = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if trajectory.ndim == 2 and len(trajectory) > 0:
            idx = max(0, min(sparse_zero, len(trajectory) - 1))
            pose[:3] = np.asarray(trajectory[idx, :3], dtype=np.float32)
        return pose

    def _resolve_raw_frame_id(self, merged_data: Dict[str, Any], frame_idx: int) -> int:
        index_list = merged_data.get("index", None)
        if isinstance(index_list, list) and len(index_list) > 0:
            pos = max(0, min(int(frame_idx) - 1, len(index_list) - 1))
            return int(index_list[pos])
        return int(frame_idx) - 1

    def _extract_images_from_npz_by_raw_id(
        self,
        rgb: np.ndarray,
        frame_ids: Optional[List[int]],
        camera_pos_map: Dict[str, int],
        raw_frame_id: int,
        camera_names: List[str],
    ) -> List[np.ndarray]:
        if frame_ids is not None:
            frame_map = {int(fid): idx for idx, fid in enumerate(frame_ids)}
            frame_pos = frame_map.get(int(raw_frame_id), None)
            if frame_pos is None:
                frame_pos = max(0, min(int(raw_frame_id), len(rgb) - 1))
        else:
            frame_pos = max(0, min(int(raw_frame_id), len(rgb) - 1))
        frame = rgb[frame_pos]
        images: List[np.ndarray] = []
        for camera_name in camera_names:
            pos = camera_pos_map.get(camera_name, None)
            if pos is None or pos >= frame.shape[0]:
                raise KeyError(f"Camera {camera_name} is unavailable in RGB npz.")
            images.append(np.array(frame[pos], copy=True, order="C"))
        return images

    def _load_depth_sidecar(self, map_name: str, uuid: str) -> Optional[Dict[str, Any]]:
        if not self.use_clearance_supervision:
            return None
        key = f"{map_name}/{uuid}"
        cached = self._depth_sidecar_cache.get(key, None)
        if key in self._depth_sidecar_cache:
            self._depth_sidecar_cache.move_to_end(key)
            return cached
        npz_path = os.path.join(self.depth_dataset_path, map_name, uuid, "depth_imgs_uint8.npz")
        payload: Optional[Dict[str, Any]] = None
        if os.path.isfile(npz_path):
            try:
                with np.load(npz_path, allow_pickle=False) as data:
                    depth_key = None
                    for candidate in ("depths", "depth", "imgs"):
                        if candidate in data:
                            depth_key = candidate
                            break
                    if depth_key is not None:
                        depths = np.asarray(data[depth_key], dtype=np.uint8)
                        frame_ids = data["frame_ids"].tolist() if "frame_ids" in data else None
                        camera_pos_map: Dict[str, int] = {}
                        if "camera_names" in data:
                            camera_names = data["camera_names"].tolist()
                            if not isinstance(camera_names, list):
                                camera_names = [camera_names]
                            for pos, raw_name in enumerate(camera_names):
                                name = self._normalize_camera_name(raw_name)
                                if name is not None:
                                    camera_pos_map.setdefault(name, pos)
                        if not camera_pos_map and "camera_indices" in data:
                            camera_indices = data["camera_indices"].tolist()
                            if not isinstance(camera_indices, list):
                                camera_indices = [camera_indices]
                            for pos, full_idx in enumerate(camera_indices):
                                name = self.CAMERA_INDEX_TO_NAME.get(self._safe_int(full_idx, -1), None)
                                if name is not None:
                                    camera_pos_map.setdefault(name, pos)
                        if not camera_pos_map:
                            camera_pos_map = {name: idx for idx, name in enumerate(self.RGB_FOLDERS[: depths.shape[1]])}
                        payload = {
                            "depths": depths,
                            "frame_ids": frame_ids,
                            "camera_pos_map": camera_pos_map,
                        }
            except Exception as exc:
                logger.warning("Failed to read depth sidecar %s: %s", npz_path, exc)
                payload = None
        if self._depth_cache_size > 0:
            self._depth_sidecar_cache[key] = payload
            self._depth_sidecar_cache.move_to_end(key)
            while len(self._depth_sidecar_cache) > self._depth_cache_size:
                self._depth_sidecar_cache.popitem(last=False)
        return payload

    def _extract_depth_frame(
        self,
        depth_payload: Dict[str, Any],
        raw_frame_id: int,
        camera_name: str,
    ) -> Tuple[np.ndarray, float]:
        depths = depth_payload["depths"]
        frame_ids = depth_payload.get("frame_ids", None)
        camera_pos_map = depth_payload.get("camera_pos_map", {})
        if frame_ids is not None:
            frame_map = {int(fid): idx for idx, fid in enumerate(frame_ids)}
            frame_pos = frame_map.get(int(raw_frame_id), None)
            if frame_pos is None:
                return np.zeros((256, 256), dtype=np.uint8), 0.0
        else:
            frame_pos = max(0, min(int(raw_frame_id), len(depths) - 1))
        cam_pos = camera_pos_map.get(camera_name, None)
        if cam_pos is None or cam_pos >= depths.shape[1]:
            return np.zeros((256, 256), dtype=np.uint8), 0.0
        depth = np.asarray(depths[frame_pos, cam_pos], dtype=np.uint8)
        return depth, 1.0

    @staticmethod
    def _first_npz_key(data: np.lib.npyio.NpzFile, candidates: Tuple[str, ...]) -> Optional[str]:
        for key in candidates:
            if key in data:
                return key
        return None

    def _zero_da3_joint_payload(self) -> Dict[str, torch.Tensor]:
        view_count = len(self.clearance_depth_cameras)
        return {
            "da3_joint_feat": torch.zeros((view_count, 1, self.da3_joint_teacher_dim), dtype=torch.float32),
            "da3_joint_valid_mask": torch.tensor(0.0, dtype=torch.float32),
            "da3_joint_token_mask": torch.zeros((view_count, 1), dtype=torch.float32),
        }

    def _load_da3_joint_cache(self, map_name: str, uuid: str) -> Optional[Dict[str, Any]]:
        if not self.use_da3_joint_supervision:
            return None
        key = f"{map_name}/{uuid}"
        cached = self._da3_joint_cache.get(key, None)
        if key in self._da3_joint_cache:
            self._da3_joint_cache.move_to_end(key)
            return cached

        cache_dir = os.path.join(self.da3_joint_teacher_path, map_name, uuid)
        npz_path = os.path.join(cache_dir, "da3_joint_feats.npz")
        payload: Optional[Dict[str, Any]] = None
        if os.path.isfile(npz_path):
            try:
                with np.load(npz_path, allow_pickle=False) as data:
                    feat_key = self._first_npz_key(
                        data,
                        ("joint_tokens", "sliced", "da3_sliced", "da3_joint_tokens", "features", "feats"),
                    )
                    if feat_key is None:
                        raise KeyError("missing one of joint_feats/features/feats/da3_joint_feats")
                    feats = np.asarray(data[feat_key], dtype=np.float32)
                    if feats.ndim < 3:
                        raise ValueError(
                            f"DA3 joint cache must store token grids with shape [T,V,L,D] or [T,L,D], got {feats.shape}"
                        )
                    frame_ids = data["frame_ids"].astype(np.int64).tolist() if "frame_ids" in data else None
                    valid_mask = (
                        np.asarray(data["valid_mask"], dtype=np.float32).reshape(-1)
                        if "valid_mask" in data
                        else np.ones((feats.shape[0],), dtype=np.float32)
                    )
                    payload = {
                        "features": feats,
                        "frame_ids": frame_ids,
                        "valid_mask": valid_mask,
                    }
            except Exception as exc:
                logger.warning("Failed to read DA3 joint cache %s: %s", npz_path, exc)
                payload = None

        if self._da3_joint_cache_size > 0:
            self._da3_joint_cache[key] = payload
            self._da3_joint_cache.move_to_end(key)
            while len(self._da3_joint_cache) > self._da3_joint_cache_size:
                self._da3_joint_cache.popitem(last=False)
        return payload

    def _load_da3_joint_frame_file(self, map_name: str, uuid: str, raw_frame_id: int) -> Tuple[Optional[np.ndarray], float]:
        cache_dir = os.path.join(self.da3_joint_teacher_path, map_name, uuid)
        candidates = [
            os.path.join(cache_dir, "frames", f"{int(raw_frame_id):06d}", "da3_sliced.npy"),
            os.path.join(cache_dir, f"frame_{int(raw_frame_id):06d}.npz"),
            os.path.join(cache_dir, f"{int(raw_frame_id):06d}.npz"),
            os.path.join(cache_dir, f"frame_{int(raw_frame_id)}.npz"),
            os.path.join(cache_dir, f"{int(raw_frame_id)}.npz"),
        ]
        for path in candidates:
            if not os.path.isfile(path):
                continue
            try:
                if path.endswith(".npy"):
                    feat = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
                    return feat, 1.0
                else:
                    with np.load(path, allow_pickle=False) as data:
                        feat_key = self._first_npz_key(
                            data,
                            ("joint_tokens", "sliced", "da3_sliced", "da3_joint_tokens", "joint_feat", "feature", "feat"),
                        )
                        if feat_key is None:
                            raise KeyError("missing one of joint_tokens/sliced/da3_sliced/da3_joint_tokens")
                        feat = np.asarray(data[feat_key], dtype=np.float32)
                        valid = float(np.asarray(data["valid"]).reshape(-1)[0]) if "valid" in data else 1.0
                        return feat, valid
            except Exception as exc:
                logger.warning("Failed to read DA3 joint frame cache %s: %s", path, exc)
                return None, 0.0
        return None, 0.0

    def _build_da3_joint_inputs(self, map_name: str, uuid: str, raw_frame_id: int) -> Dict[str, torch.Tensor]:
        if not self.use_da3_joint_supervision:
            return {}
        if not map_name or not uuid:
            if self.strict_da3_joint_cache:
                raise FileNotFoundError("DA3 joint cache requires valid map/uuid.")
            return self._zero_da3_joint_payload()

        feat: Optional[np.ndarray] = None
        valid = 0.0
        payload = self._load_da3_joint_cache(map_name, uuid)
        if payload is not None:
            features = payload["features"]
            frame_ids = payload.get("frame_ids", None)
            if frame_ids is not None:
                frame_map = {int(fid): idx for idx, fid in enumerate(frame_ids)}
                idx = frame_map.get(int(raw_frame_id), None)
            else:
                idx = max(0, min(int(raw_frame_id), len(features) - 1))
            if idx is not None and 0 <= int(idx) < len(features):
                feat = np.asarray(features[int(idx)], dtype=np.float32)
                valid_mask = payload.get("valid_mask", None)
                valid = float(valid_mask[int(idx)]) if valid_mask is not None and int(idx) < len(valid_mask) else 1.0

        if feat is None:
            feat, valid = self._load_da3_joint_frame_file(map_name, uuid, raw_frame_id)

        if feat is None:
            if self.strict_da3_joint_cache:
                raise FileNotFoundError(
                    f"Missing DA3 joint teacher feature for {map_name}/{uuid} raw_frame_id={int(raw_frame_id)} "
                    f"under {self.da3_joint_teacher_path}."
                )
            return self._zero_da3_joint_payload()

        if feat.ndim == 2:
            feat = feat.reshape(1, feat.shape[0], feat.shape[1])
        if feat.ndim != 3:
            raise ValueError(
                f"DA3 sliced teacher must have shape [V,L,D] or [L,D] for {map_name}/{uuid}:{raw_frame_id}, got {feat.shape}."
            )
        if feat.shape[-1] != self.da3_joint_teacher_dim:
            raise ValueError(
                f"DA3 joint teacher dim mismatch for {map_name}/{uuid} raw_frame_id={int(raw_frame_id)}: "
                f"cache_dim={feat.shape[-1]}, expected={self.da3_joint_teacher_dim}. "
                "Set --da3_joint_teacher_dim / GEORISK_DA3_JOINT_TEACHER_DIM to match the cache."
            )
        expected_views = len(self.clearance_depth_cameras)
        if feat.shape[0] != expected_views:
            raise ValueError(
                f"DA3 sliced teacher view count mismatch for {map_name}/{uuid} raw_frame_id={int(raw_frame_id)}: "
                f"cache_views={feat.shape[0]}, expected={expected_views} ({self.clearance_depth_cameras})."
            )
        if not np.all(np.isfinite(feat)):
            if self.strict_da3_joint_cache:
                raise ValueError(f"Non-finite DA3 joint teacher feature for {map_name}/{uuid}:{raw_frame_id}.")
            return self._zero_da3_joint_payload()
        token_mask = np.ones((feat.shape[0], feat.shape[1]), dtype=np.float32)
        return {
            "da3_joint_feat": torch.from_numpy(feat.astype(np.float32, copy=False)),
            "da3_joint_valid_mask": torch.tensor(float(valid > 0.5), dtype=torch.float32),
            "da3_joint_token_mask": torch.from_numpy(token_mask),
        }

    def _extract_images_from_npz(
        self,
        rgb: np.ndarray,
        frame_ids: Optional[List[int]],
        camera_pos_map: Dict[str, int],
        merged_data: Dict,
        frame_num: int,
        npz_desc: str,
    ) -> List[np.ndarray]:
        frame_idx = max(0, min(frame_num - 1, len(rgb) - 1))
        index_list = merged_data.get("index", None)
        if frame_ids is not None and index_list is not None and len(index_list) >= frame_num:
            target_frame_id = int(index_list[frame_num - 1])
            frame_map = {int(fid): idx for idx, fid in enumerate(frame_ids)}
            if target_frame_id in frame_map:
                frame_idx = frame_map[target_frame_id]

        frame = rgb[frame_idx]
        missing = [camera for camera in self.selected_folders if camera not in camera_pos_map]
        if missing:
            available = sorted(camera_pos_map.keys())
            raise ValueError(
                f"Requested cameras {missing} are not available in {npz_desc}. "
                f"available={available}, view_mode={self.view_mode}. "
                "GeoRisk requires both front and down views."
            )
        selected_positions = [camera_pos_map[name] for name in self.selected_folders]
        max_pos = max(selected_positions)
        if frame.shape[0] <= max_pos:
            raise ValueError(
                f"RGB camera axis in {npz_desc} has shape={frame.shape}; cannot access position {max_pos} "
                f"for requested folders {self.selected_folders}."
            )
        return [np.array(frame[pos], copy=True, order="C") for pos in selected_positions]

    def _select_npz_path(self, traj_dir: str) -> Optional[str]:
        for filename in self.NPZ_CANDIDATE_FILENAMES:
            path = os.path.join(traj_dir, filename)
            if os.path.exists(path):
                return path
        npz_files = sorted(
            [os.path.join(traj_dir, name) for name in os.listdir(traj_dir) if name.endswith(".npz")]
        )
        return npz_files[0] if npz_files else None

    def _load_multiview_images_from_npz_source(
        self,
        npz_source: Union[str, bytes, bytearray],
        npz_desc: str,
        merged_data: Dict,
        frame_num: int,
    ) -> List[np.ndarray]:
        rgb, frame_ids, camera_pos_map = self._load_rgb_tensor_from_npz(npz_source=npz_source, npz_desc=npz_desc)
        return self._extract_images_from_npz(
            rgb=rgb,
            frame_ids=frame_ids,
            camera_pos_map=camera_pos_map,
            merged_data=merged_data,
            frame_num=frame_num,
            npz_desc=npz_desc,
        )

    def _load_multiview_images_from_filesystem(self, traj_dir: str, merged_data: Dict, frame_num: int) -> List[np.ndarray]:
        npz_path = self._select_npz_path(traj_dir)
        if npz_path is not None:
            return self._load_multiview_images_from_npz_source(
                npz_source=npz_path,
                npz_desc=npz_path,
                merged_data=merged_data,
                frame_num=frame_num,
            )

        index_list = merged_data.get("index", list(range(merged_data.get("length", frame_num))))
        real_index = index_list[max(0, min(frame_num - 1, len(index_list) - 1))]
        images = []
        for folder in self.selected_folders:
            img_path = os.path.join(traj_dir, folder, f"{int(real_index):06d}.png")
            if not os.path.exists(img_path):
                raise FileNotFoundError(img_path)
            images.append(np.array(Image.open(img_path).convert("RGB"), copy=True, order="C"))
        return images

    @staticmethod
    def _find_stage(trajectory: np.ndarray, frame_num: int) -> Tuple[str, np.ndarray]:
        def turning_stage(p0, p1, p2):
            prev_vec = p1 - p0
            now_vec = p2 - p1
            denom = (np.linalg.norm(prev_vec) + 1e-6) * (np.linalg.norm(now_vec) + 1e-6)
            delta_angle = np.arccos(np.clip(np.dot(prev_vec, now_vec) / denom, -1.0, 1.0)) * 180.0 / np.pi
            if 25 < delta_angle < 120:
                return "right" if int(np.cross(prev_vec, now_vec)) > 0 else "left"
            return "cruise"

        stage = "cruise"
        prev_vec = np.asarray([0.0, 0.0, -4.5], dtype=np.float32)
        if len(trajectory) < 2:
            return stage, prev_vec

        z_values = trajectory[:, 2]
        now_idx = max(0, min(frame_num - 1, len(z_values) - 1))
        future_idx = min(frame_num + 2, len(z_values) - 1)
        now_z = z_values[now_idx]
        future_z = z_values[future_idx]
        if now_z - future_z > 5:
            stage = "take off"
        elif now_z - future_z < -5:
            stage = "landing"

        if 1 <= now_idx < len(trajectory) - 1:
            prev_vec = np.asarray(trajectory[now_idx, :3] - trajectory[now_idx - 1, :3], dtype=np.float32)
            if stage == "cruise":
                stage = turning_stage(
                    trajectory[now_idx - 1, :2],
                    trajectory[now_idx, :2],
                    trajectory[now_idx + 1, :2],
                )
        return stage, prev_vec

    def _preprocess_multiview_images(self, images: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        safe_images: List[np.ndarray] = []
        for image in images:
            safe_images.append(np.array(image, copy=True, order="C"))
        processed = self.image_processor.preprocess(safe_images, return_tensors="pt")
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

        merge_size = self.image_processor.merge_size
        merged_grid_tokens = [
            int(image_grid_thw[idx].prod().item() // (merge_size**2))
            for idx in range(image_grid_thw.shape[0])
        ]
        return pixel_values, image_grid_thw, merged_grid_tokens

    def _build_clearance_inputs(
        self,
        map_name: str,
        uuid: str,
        frame_idx: int,
        merged_data: Dict[str, Any],
        trajectory_data: np.ndarray,
        rotation_matrix: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        view_count = len(self.clearance_frame_offsets) * len(self.clearance_depth_cameras)
        zero_depth = np.zeros((view_count, 256, 256), dtype=np.uint8)
        zero_view_poses = np.tile(np.eye(4, dtype=np.float32).reshape(1, 16), (view_count, 1))

        def make_zero_payload() -> Dict[str, torch.Tensor]:
            return {
                "clearance_depth_gt": torch.from_numpy(zero_depth.copy()),
                "clearance_depth_valid_mask": torch.zeros((view_count,), dtype=torch.float32),
                "clearance_geom_valid_mask": torch.tensor(0.0, dtype=torch.float32),
                "clearance_view_to_current_poses": torch.from_numpy(zero_view_poses.copy()),
                "clearance_target_to_current_rot": torch.from_numpy(
                    np.asarray(rotation_matrix, dtype=np.float32).T.copy()
                ),
            }

        if not self.use_clearance_supervision:
            return {}
        if not map_name or not uuid or map_name in self.clearance_bad_depth_maps:
            return make_zero_payload()

        depth_payload = self._load_depth_sidecar(map_name, uuid)
        if depth_payload is None:
            return make_zero_payload()

        total_sparse = int(len(trajectory_data))
        start_raw_id = self._resolve_raw_frame_id(merged_data, 1)
        start_pose = self._pose_to_matrix(
            self._pose_from_trajectory(
                trajectory_data,
                frame_idx=1,
                merged_data=merged_data,
                raw_frame_id=start_raw_id,
            )
        )
        current_raw_id = self._resolve_raw_frame_id(merged_data, frame_idx)
        current_pose = self._pose_to_matrix(
            self._pose_from_trajectory(
                trajectory_data,
                frame_idx=frame_idx,
                merged_data=merged_data,
                raw_frame_id=current_raw_id,
            )
        )
        try:
            current_inv = np.linalg.inv(current_pose)
        except Exception:
            current_inv = np.eye(4, dtype=np.float32)

        depth_gt = zero_depth.copy()
        depth_valid = np.zeros((view_count,), dtype=np.float32)
        view_poses = zero_view_poses.copy()
        view_idx = 0
        for offset in self.clearance_frame_offsets:
            sparse_idx = int(frame_idx) + int(offset)
            sparse_valid = 1.0 if 1 <= sparse_idx <= total_sparse else 0.0
            sparse_idx_clamped = max(1, min(sparse_idx, max(1, total_sparse)))
            raw_id = self._resolve_raw_frame_id(merged_data, sparse_idx_clamped)
            hist_pose = self._pose_to_matrix(
                self._pose_from_trajectory(
                    trajectory_data,
                    frame_idx=sparse_idx_clamped,
                    merged_data=merged_data,
                    raw_frame_id=raw_id,
                )
            )
            hist_to_current = (current_inv @ hist_pose).astype(np.float32)
            for cam_name in self.clearance_depth_cameras:
                depth, valid = self._extract_depth_frame(depth_payload, raw_id, cam_name)
                depth_gt[view_idx] = depth
                depth_valid[view_idx] = float(valid) * sparse_valid
                view_poses[view_idx] = hist_to_current.reshape(16)
                view_idx += 1

        return {
            "clearance_depth_gt": torch.from_numpy(depth_gt),
            "clearance_depth_valid_mask": torch.from_numpy(depth_valid),
            "clearance_geom_valid_mask": torch.tensor(float(depth_valid.sum() > 0), dtype=torch.float32),
            "clearance_view_to_current_poses": torch.from_numpy(view_poses),
            
            
            "clearance_target_to_current_rot": torch.from_numpy(
                (
                    np.asarray(rotation_matrix, dtype=np.float32).T
                    @ np.asarray(start_pose[:3, :3], dtype=np.float32).T
                    @ np.asarray(current_pose[:3, :3], dtype=np.float32)
                ).astype(np.float32)
            ),
        }

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        info = self.list_data_dict[i]
        json_rel = info["json"]
        frame_num = int(info["frame"])

        if self.dataset_backend == "webdataset":
            sample_key = self._json_rel_to_sample_key(json_rel)
            sample = self._load_webdataset_sample(sample_key, need_npz=True)
            merged_data = copy.deepcopy(sample["merged_data"])
            npz_bytes = sample.get("npz_bytes", None)
            if npz_bytes is None:
                raise FileNotFoundError(
                    f"Sample {sample_key} has no npz payload in webdataset shards. "
                    "convert_to_webdataset.py must include *.npz entries."
            )
            npz_name = str(sample.get("meta", {}).get("npz_name", "")).strip() or f"{sample_key}.npz"
            npz_desc = f"webdataset:{sample_key}:{npz_name}"
            rgb_pack = self._load_rgb_tensor_from_npz(npz_source=npz_bytes, npz_desc=npz_desc)
            images = self._extract_images_from_npz(
                rgb=rgb_pack[0],
                frame_ids=rgb_pack[1],
                camera_pos_map=rgb_pack[2],
                merged_data=merged_data,
                frame_num=frame_num,
                npz_desc=npz_desc,
            )
        else:
            traj_dir = os.path.join(self.dataset_path, *str(json_rel).replace("\\", "/").split("/")[:-1])
            json_path = os.path.join(self.dataset_path, str(json_rel).replace("/", os.sep))
            merged_data = copy.deepcopy(self._load_json(json_path))
            npz_path = self._select_npz_path(traj_dir)
            if npz_path is not None:
                rgb_pack = self._load_rgb_tensor_from_npz(npz_source=npz_path, npz_desc=npz_path)
                images = self._extract_images_from_npz(
                    rgb=rgb_pack[0],
                    frame_ids=rgb_pack[1],
                    camera_pos_map=rgb_pack[2],
                    merged_data=merged_data,
                    frame_num=frame_num,
                    npz_desc=npz_path,
                )
            else:
                rgb_pack = None
                images = self._load_multiview_images_from_filesystem(traj_dir, merged_data, frame_num)

        instruction = merged_data["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()

        trajectory_data = np.asarray(merged_data["trajectory"], dtype=np.float32)
        if len(trajectory_data) == 0:
            raise ValueError(f"Empty trajectory in {self.dataset_backend}:{json_rel}")
        frame_idx = max(1, min(frame_num, len(trajectory_data)))
        x, y = trajectory_data[-1][0], trajectory_data[-1][1]
        rotation_matrix = rotation_matrix_from_vector(float(x), float(y))
        dense_traj = self._load_dense_trajectory(json_rel)
        dense_idx = self._resolve_dense_index(merged_data, frame_idx=frame_idx, dense_len=len(dense_traj))
        trajectory_point_label, traj_valid_mask = self._build_dense_target(
            dense_traj=dense_traj,
            dense_idx=dense_idx,
            rotation_matrix=rotation_matrix,
        )

        stage, future_delta = self._find_stage(trajectory_data, frame_idx)
        future_delta = transform_point(future_delta, rotation_matrix)
        future_delta = future_delta / (np.linalg.norm(future_delta) + 1e-8)
        delta_str = ",".join([str(round(float(v), 1)) for v in future_delta])
        cur_pos = transform_point(trajectory_data[frame_idx - 1 : frame_idx, :3], rotation_matrix)[0]
        cur_pos_str = ",".join([str(round(float(v), 1)) for v in cur_pos])

        round1_user = observation_prompt(instruction, len(images))
        round2_user = trajectory_prompt(stage, delta_str, cur_pos_str)
        source = [
            {"from": "human", "value": round1_user},
            {"from": "gpt", "value": STOP_REPLY},
            {"from": "human", "value": round2_user},
            {"from": "gpt", "value": TRAJECTORY_REPLY},
        ]

        pixel_values, image_grid_thw, merged_grid_tokens = self._preprocess_multiview_images(images)
        merge_size = self.image_processor.merge_size

        data_dict = preprocess_qwen_visual(
            [source],
            self.tokenizer,
            grid_thw_image=merged_grid_tokens,
            grid_thw_video=None,
        )
        if self.stop_token_id is not None:
            stop_mask = data_dict["input_ids"] == self.stop_token_id
            data_dict["labels"][stop_mask] = IGNORE_INDEX
        
        if self.stop_token_id is not None:
            input_ids = data_dict["input_ids"]
            labels = data_dict["labels"]
            traj_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_TRAJ_TOKEN)
            if int(traj_token_id) < 0:
                traj_token_id = None
            for row in range(input_ids.shape[0]):
                stop_matches = torch.nonzero(input_ids[row] == self.stop_token_id, as_tuple=False).flatten()
                if traj_token_id is None:
                    continue
                wp_matches = torch.nonzero(input_ids[row] == int(traj_token_id), as_tuple=False).flatten()
                if len(stop_matches) == 0 or len(wp_matches) == 0:
                    continue
                start = int(stop_matches[0].item()) + 1
                end = int(wp_matches[-1].item())
                if start < end:
                    labels[row, start:end] = IGNORE_INDEX
        attention_mask = torch.ones_like(data_dict["input_ids"])
        position_ids, _ = get_rope_index_3(
            spatial_merge_size=merge_size,
            input_ids=data_dict["input_ids"],
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            second_per_grid_ts=None,
            attention_mask=attention_mask,
        )

        goal_point = trajectory_data[-1, :3]
        cur_point = trajectory_data[frame_idx - 1, :3]
        stop_distance_to_goal = float(np.linalg.norm(cur_point - goal_point))
        stop_logit = (self.stop_soft_r - stop_distance_to_goal) / self.stop_soft_tau
        stop_logit = float(np.clip(stop_logit, -60.0, 60.0))
        stop_label = float(1.0 / (1.0 + math.exp(-stop_logit)))
        if self.stop_label_clip_eps > 0.0:
            eps = float(self.stop_label_clip_eps)
            stop_label = float(min(max(stop_label, eps), 1.0 - eps))
        stop_valid_mask = 1.0
        
        
        is_terminal_frame = frame_idx >= len(trajectory_data)
        phase2_valid_mask = 0.0 if (is_terminal_frame or float(traj_valid_mask.sum()) <= 0.0) else 1.0

        map_name, uuid = self._split_json_rel(json_rel)
        result = {
            "input_ids": data_dict["input_ids"],
            "labels": data_dict["labels"],
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "trajectory": torch.tensor(trajectory_point_label, dtype=torch.float32),
            "traj_valid_mask": torch.tensor(traj_valid_mask, dtype=torch.float32),
            "stop_label": torch.tensor(stop_label, dtype=torch.float32),
            "stop_valid_mask": torch.tensor(stop_valid_mask, dtype=torch.float32),
            "stop_distance_to_goal": torch.tensor(stop_distance_to_goal, dtype=torch.float32),
            "phase2_valid_mask": torch.tensor(phase2_valid_mask, dtype=torch.float32),
            "prompt": round1_user,
        }
        result.update(
            self._build_clearance_inputs(
                map_name=map_name,
                uuid=uuid,
                frame_idx=frame_idx,
                merged_data=merged_data,
                trajectory_data=trajectory_data,
                rotation_matrix=rotation_matrix,
            )
        )
        raw_frame_id = self._resolve_raw_frame_id(merged_data, frame_idx)
        result.update(
            self._build_da3_joint_inputs(
                map_name=map_name,
                uuid=uuid,
                raw_frame_id=raw_frame_id,
            )
        )
        return result
