#!/usr/bin/env python


import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


WORKSPACE_ROOT = Path(os.environ.get("GEORISK_ASSETS_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_sample_key(json_rel: str) -> Optional[Tuple[str, str]]:
    parts = str(json_rel).replace("\\", "/").strip("/").split("/")
    if len(parts) >= 3:
        return parts[-3], parts[-2]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None


def collect_requested_frames(data_json: Path) -> "OrderedDict[Tuple[str, str], Set[int]]":
    payload = load_json(data_json)
    entries = payload.get("samples", payload.get("data", [])) if isinstance(payload, dict) else payload
    requested: "OrderedDict[Tuple[str, str], Set[int]]" = OrderedDict()
    for item in entries:
        if not isinstance(item, dict):
            continue
        key = parse_sample_key(item.get("json", ""))
        if key is None:
            continue
        try:
            sparse_frame = int(item.get("frame", 0) or 0)
        except (TypeError, ValueError):
            sparse_frame = 0
        if sparse_frame > 0:
            requested.setdefault(key, set()).add(sparse_frame)
    return requested


def sparse_to_raw_frame_id(merged_data: Dict, sparse_frame: int) -> int:
    indices = merged_data.get("index", None)
    if isinstance(indices, list) and indices:
        pos = max(0, min(int(sparse_frame) - 1, len(indices) - 1))
        return int(indices[pos])
    return max(0, int(sparse_frame) - 1)


def resolve_merged_data_path(dense_root: Path, map_name: str, uuid: str) -> Path:
    trajectory_root = dense_root / map_name / uuid
    candidates = (
        trajectory_root / "merged_data.json",
        trajectory_root / "merged_data_all.json",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def iter_required_raw_frames(
    requested: "OrderedDict[Tuple[str, str], Set[int]]",
    dense_root: Path,
) -> Iterable[Tuple[str, str, int, Optional[str]]]:
    for (map_name, uuid), sparse_frames in requested.items():
        merged_path = resolve_merged_data_path(dense_root, map_name, uuid)
        if not merged_path.is_file():
            for sparse_frame in sorted(sparse_frames):
                yield map_name, uuid, int(sparse_frame), f"missing source metadata: {merged_path}"
            continue
        try:
            merged_data = load_json(merged_path)
        except Exception as exc:
            for sparse_frame in sorted(sparse_frames):
                yield map_name, uuid, int(sparse_frame), f"invalid source metadata: {exc}"
            continue
        for sparse_frame in sorted(sparse_frames):
            yield map_name, uuid, sparse_to_raw_frame_id(merged_data, sparse_frame), None


def sampled_values_are_finite(array: np.ndarray) -> bool:
    if array.size == 0:
        return False
    flat = array.reshape(-1)
    positions = np.unique(np.linspace(0, flat.size - 1, min(4096, flat.size)).round().astype(np.int64))
    return bool(np.isfinite(flat[positions]).all())


def validate_cache_file(
    feature_path: Path,
    raw_frame_id: int,
    expected_views: int,
    expected_dim: int,
    expected_token_count: int,
    require_meta: bool,
    full_finite_check: bool,
) -> Optional[str]:
    if not feature_path.is_file():
        return "missing feature file"
    try:
        features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        return f"unreadable NPY: {exc}"
    if features.ndim != 3:
        return f"expected [V,L,D], got shape={tuple(features.shape)}"
    if int(features.shape[0]) != int(expected_views):
        return f"view mismatch: got={features.shape[0]} expected={expected_views}"
    if int(features.shape[1]) <= 0:
        return "empty token dimension"
    if int(expected_token_count) > 0 and int(features.shape[1]) != int(expected_token_count):
        return f"token count mismatch: got={features.shape[1]} expected={expected_token_count}"
    if int(features.shape[2]) != int(expected_dim):
        return f"feature dim mismatch: got={features.shape[2]} expected={expected_dim}"
    if full_finite_check:
        finite = bool(np.isfinite(np.asarray(features)).all())
    else:
        finite = sampled_values_are_finite(features)
    if not finite:
        return "non-finite feature values"

    meta_path = feature_path.parent / "meta.json"
    if require_meta and not meta_path.is_file():
        return "missing meta.json"
    if meta_path.is_file():
        try:
            meta = load_json(meta_path)
        except Exception as exc:
            return f"invalid meta.json: {exc}"
        if int(meta.get("raw_frame_id", -1)) != int(raw_frame_id):
            return f"meta raw_frame_id mismatch: got={meta.get('raw_frame_id')} expected={raw_frame_id}"
        cameras = list(meta.get("cameras", []))
        if cameras and cameras != ["frontcamera", "downcamera"]:
            return f"camera order mismatch: {cameras}"
        meta_dim = int(meta.get("expected_feature_dim", meta.get("feature_dim", expected_dim)))
        if meta_dim != int(expected_dim):
            return f"meta feature dim mismatch: got={meta_dim} expected={expected_dim}"
        meta_tokens = int(meta.get("tokens_per_view", features.shape[1]))
        if int(expected_token_count) > 0 and meta_tokens != int(expected_token_count):
            return f"meta token count mismatch: got={meta_tokens} expected={expected_token_count}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_root",
        default=str(WORKSPACE_ROOT / "TravelUAV_da3_large_joint_teacher"),
    )
    parser.add_argument(
        "--data_json",
        default=str(WORKSPACE_ROOT / "TravelUAV_data_json" / "data" / "uav_dataset" / "trainset.json"),
    )
    parser.add_argument(
        "--dense_dataset_root",
        default=str(WORKSPACE_ROOT / "TravelUAV_original_decompressed_merged_all"),
    )
    parser.add_argument("--expected_views", type=int, default=2)
    parser.add_argument("--expected_feat_dim", type=int, default=1024)
    parser.add_argument("--expected_tokens_per_view", type=int, default=64)
    parser.add_argument("--max_frames", type=int, default=0, help="0 validates every required frame.")
    parser.add_argument("--require_meta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--full_finite_check", action="store_true")
    parser.add_argument("--allow_incomplete", action="store_true")
    parser.add_argument("--report_json", default="")
    args = parser.parse_args()

    cache_root = Path(args.cache_root).expanduser().resolve()
    data_json = Path(args.data_json).expanduser().resolve()
    dense_root = Path(args.dense_dataset_root).expanduser().resolve()
    if not data_json.is_file():
        raise FileNotFoundError(f"Training manifest not found: {data_json}")
    if not cache_root.is_dir():
        raise FileNotFoundError(f"DA3 cache root not found: {cache_root}")
    if not dense_root.is_dir():
        raise FileNotFoundError(f"Dense dataset root not found: {dense_root}")

    requested = collect_requested_frames(data_json)
    required = list(iter_required_raw_frames(requested, dense_root))
    required = list(OrderedDict(((m, u, f), err) for m, u, f, err in required).items())
    if int(args.max_frames) > 0 and len(required) > int(args.max_frames):
        positions = np.linspace(0, len(required) - 1, int(args.max_frames)).round().astype(np.int64)
        required = [required[int(pos)] for pos in np.unique(positions)]

    valid = 0
    issues: List[Dict[str, object]] = []
    token_counts: List[int] = []
    for (map_name, uuid, raw_frame_id), source_error in required:
        feature_path = cache_root / map_name / uuid / "frames" / f"{int(raw_frame_id):06d}" / "da3_sliced.npy"
        error = source_error
        if error is None:
            error = validate_cache_file(
                feature_path=feature_path,
                raw_frame_id=int(raw_frame_id),
                expected_views=int(args.expected_views),
                expected_dim=int(args.expected_feat_dim),
                expected_token_count=int(args.expected_tokens_per_view),
                require_meta=bool(args.require_meta),
                full_finite_check=bool(args.full_finite_check),
            )
        if error is None:
            features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
            token_counts.append(int(features.shape[1]))
            valid += 1
        elif len(issues) < 1000:
            issues.append(
                {
                    "map": map_name,
                    "uuid": uuid,
                    "raw_frame_id": int(raw_frame_id),
                    "path": str(feature_path),
                    "error": str(error),
                }
            )

    checked = len(required)
    invalid = checked - valid
    report = {
        "cache_root": str(cache_root),
        "data_json": str(data_json),
        "dense_dataset_root": str(dense_root),
        "requested_trajectories": len(requested),
        "checked_frames": checked,
        "valid_frames": valid,
        "invalid_frames": invalid,
        "coverage": (float(valid) / float(checked)) if checked else 0.0,
        "expected_views": int(args.expected_views),
        "expected_feature_dim": int(args.expected_feat_dim),
        "expected_tokens_per_view": int(args.expected_tokens_per_view),
        "token_count_min": min(token_counts) if token_counts else None,
        "token_count_max": max(token_counts) if token_counts else None,
        "issues": issues,
    }
    print(json.dumps(report, indent=2))
    if args.report_json:
        report_path = Path(args.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    if checked == 0:
        return 2
    if invalid and not args.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
