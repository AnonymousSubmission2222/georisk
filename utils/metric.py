import os
import json
import numpy as np
import re
import tqdm
import argparse
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

AIRSIM_COLLISION_EVENT = "airsim_has_collided"
DEPTH_COLLISION_EVENT = "depth_le1_gt10pct"
COLLISION_CONDITIONS = (
    (AIRSIM_COLLISION_EVENT, "AirSim has_collided"),
    (DEPTH_COLLISION_EVENT, "five-view encoded depth <= 1 over 10%"),
)
SUCCESS_RADIUS_M = 20.0

def sort_key(filename):
    
    return int(re.search(r'\d+', filename).group())


def load_json(file_path):
    
    with open(file_path, 'r') as f:
        return json.load(f)


def _trajectory_dirs(path):
    return sorted(
        item
        for item in os.listdir(path)
        if os.path.isdir(os.path.join(path, item))
        and os.path.isfile(os.path.join(path, item, "ori_info.json"))
    )


def _load_termination_info(traj_path, ori_info):
    termination_path = os.path.join(traj_path, "termination_info.json")
    if os.path.isfile(termination_path):
        return load_json(termination_path)
    info = ori_info.get("termination_info", {})
    return info if isinstance(info, dict) else {}


def _load_trajectory_record(path, traj_dir):
    traj_path = os.path.join(path, traj_dir)
    log_dir = os.path.join(traj_path, "log")
    log_files = sorted(
        (item for item in os.listdir(log_dir) if item.endswith(".json")),
        key=sort_key,
    )
    frames = []
    positions = []
    for log_file in log_files:
        log_data = load_json(os.path.join(log_dir, log_file))
        frames.append(int(log_data.get("frame", sort_key(log_file))))
        positions.append(np.asarray(log_data["sensors"]["state"]["position"], dtype=np.float64))

    ori_info = load_json(os.path.join(traj_path, "ori_info.json"))
    ori_data = load_json(os.path.join(ori_info["ori_traj_dir"], "merged_data.json"))["trajectory_raw_detailed"]
    gt_positions = [np.asarray(item["position"], dtype=np.float64) for item in ori_data]
    termination_info = _load_termination_info(traj_path, ori_info)
    target_position = np.asarray(
        termination_info.get("target_position", gt_positions[-1]),
        dtype=np.float64,
    )
    collision_events = termination_info.get("collision_events")

    return {
        "name": traj_dir,
        "frames": frames,
        "positions": positions,
        "gt_positions": gt_positions,
        "target_position": target_position,
        "collision_events": collision_events if isinstance(collision_events, dict) else None,
        "original_success": traj_dir.startswith("success_"),
        "original_oracle": traj_dir.startswith("success_") or traj_dir.startswith("oracle_"),
    }


def _path_length(points):
    return float(sum(np.linalg.norm(points[i] - points[i - 1]) for i in range(1, len(points))))


def _evaluate_record(record, condition_key=None):
    positions = record["positions"]
    frames = record["frames"]
    triggered = False
    cutoff_frame = None
    if condition_key is not None:
        event = record["collision_events"].get(condition_key, {})
        if isinstance(event, dict) and bool(event.get("triggered", False)):
            triggered = True
            cutoff_frame = int(event["first_frame_index"])

    if triggered:
        selected_positions = [position for frame, position in zip(frames, positions) if frame <= cutoff_frame]
        if not selected_positions:
            raise ValueError(
                f"No saved position at or before collision cutoff {cutoff_frame} for {record['name']}."
            )
        success = False
        oracle = any(
            np.linalg.norm(position - record["target_position"]) <= SUCCESS_RADIUS_M
            for position in selected_positions
        )
    else:
        selected_positions = positions
        success = bool(record["original_success"])
        oracle = bool(record["original_oracle"])

    gt_positions = record["gt_positions"]
    ne = float(np.linalg.norm(gt_positions[-1] - selected_positions[-1]))
    gt_length = max(0.0, _path_length(gt_positions) - SUCCESS_RADIUS_M)
    pred_length = _path_length(selected_positions)
    spl = gt_length / max(gt_length, pred_length) if success and gt_length > 0.0 else 0.0
    return {
        "triggered": triggered,
        "cutoff_frame": cutoff_frame,
        "success": success,
        "oracle_success": oracle,
        "ne_m": ne,
        "spl": float(spl),
    }


def _aggregate_records(records, condition_key=None):
    values = [_evaluate_record(record, condition_key=condition_key) for record in records]
    count = len(values)
    if count == 0:
        raise ValueError("No trajectories available for metric calculation.")
    return {
        "episodes": count,
        "triggered_episodes": int(sum(item["triggered"] for item in values)),
        "cr_percent": float(np.mean([item["triggered"] for item in values]) * 100.0),
        "sr_percent": float(np.mean([item["success"] for item in values]) * 100.0),
        "osr_percent": float(np.mean([item["oracle_success"] for item in values]) * 100.0),
        "ne_m": float(np.mean([item["ne_m"] for item in values])),
        "spl_percent": float(np.mean([item["spl"] for item in values]) * 100.0),
    }


def _log_metric_profile(label, metrics, include_cr):
    logging.info(f"Metric profile: {label}")
    logging.info(f"Episodes: {metrics['episodes']}")
    if include_cr:
        logging.info(
            "Collision Rate (CR): %.2f%% (%d/%d)",
            metrics["cr_percent"],
            metrics["triggered_episodes"],
            metrics["episodes"],
        )
    logging.info(f"Success Rate (SR): {metrics['sr_percent']:.2f}%")
    logging.info(f"Oracle Success Rate (OSR): {metrics['osr_percent']:.2f}%")
    logging.info(f"Average Normalized Error (NE): {metrics['ne_m']:.2f}")
    logging.info(f"Average Success Path Length (SPL): {metrics['spl_percent']:.2f}%")


def calculate_ne(path, dirs, success_dirs):
    
    ne_list = []
    
    for traj_dir in tqdm.tqdm(dirs, desc='Calculating NE'):
        log_dir = os.path.join(path, traj_dir, 'log')
        logs = sorted(os.listdir(log_dir), key=sort_key)

        last_log_data = load_json(os.path.join(log_dir, logs[-1]))
        last_point = np.array(last_log_data["sensors"]['state']['position'])

        ori_info = load_json(os.path.join(path, traj_dir, 'ori_info.json'))
        ori_data = load_json(os.path.join(ori_info['ori_traj_dir'], 'merged_data.json'))['trajectory_raw_detailed']
        ori_last_point = np.array(ori_data[-1]['position'])

        ne = np.linalg.norm(ori_last_point - last_point)
        ne_list.append(ne)

    avg_ne = np.mean(np.array(ne_list))
    logging.info(f"Average Normalized Error (NE): {avg_ne:.2f}")


def calculate_spl(path, dirs, success_dirs):
    
    spl_list = []

    for traj_dir in tqdm.tqdm(dirs, desc='Calculating SPL'):
        if traj_dir not in success_dirs:
            spl_list.append(0)
            continue

        log_dir = os.path.join(path, traj_dir, 'log')
        logs = sorted(os.listdir(log_dir), key=sort_key)

        pred_length = 0
        pre_point = None
        for log in logs:
            log_data = load_json(os.path.join(log_dir, log))
            point = np.array(log_data["sensors"]['state']['position'])
            if pre_point is not None:
                pred_length += np.linalg.norm(pre_point - point)
            pre_point = point

        ori_info = load_json(os.path.join(path, traj_dir, 'ori_info.json'))
        ori_data = load_json(os.path.join(ori_info['ori_traj_dir'], 'merged_data.json'))['trajectory_raw_detailed']

        path_length = 0
        for i in range(len(ori_data) - 1):
            p1 = np.array(ori_data[i]['position'])
            p2 = np.array(ori_data[i + 1]['position'])
            path_length += np.linalg.norm(p2 - p1)
        path_length -= 20  

        spl = path_length / max(path_length, pred_length)
        spl = max(spl, 0) 
        spl_list.append(spl)

    avg_spl = np.mean(np.array(spl_list)) * 100
    logging.info(f"Average Success Path Length (SPL): {avg_spl:.2f}%")


def split_data(path, path_type):
    
    dirs = _trajectory_dirs(path)
    return_dirs = []

    if path_type == 'full':
        return [traj_dir for traj_dir in dirs if 'record' not in traj_dir and 'dino' not in traj_dir]
    
    for traj_dir in tqdm.tqdm(dirs, desc='Splitting data'):
        ori_info = load_json(os.path.join(path, traj_dir, 'ori_info.json'))
        ori_data = load_json(os.path.join(ori_info['ori_traj_dir'], 'merged_data.json'))['trajectory_raw_detailed']

        path_length = sum(np.linalg.norm(np.array(ori_data[i + 1]['position']) - np.array(ori_data[i]['position'])) for i in range(len(ori_data) - 1))

        if path_type == 'easy' and path_length <= 250:
            return_dirs.append(traj_dir)
        elif path_type == 'hard' and path_length > 250:
            return_dirs.append(traj_dir)
        elif 'unseen' in path_type:
            unseen_scenes = ['Carla_Town03', 'ModularPark']
            if path_type == 'unseen scene' and any(scene in ori_info['ori_traj_dir'] for scene in unseen_scenes):
                return_dirs.append(traj_dir)
            elif path_type == 'unseen object' and not any(scene in ori_info['ori_traj_dir'] for scene in unseen_scenes):
                return_dirs.append(traj_dir)
    
    return return_dirs


def analyze_results(root_dir, analysis_list, path_type_list):
    
    for analysis_item in analysis_list:
        analysis_path = os.path.join(root_dir, analysis_item)
        if not os.path.exists(analysis_path):
            continue
        logging.info(f"\nStarting analysis for type: {analysis_item}")
        collision_report = {
            "schema_version": 1,
            "collision_stop_policy": "airsim_has_collided_and_depth_le1_gt10pct_v1",
            "path_types": {},
        }

        for path_type in path_type_list:
            logging.info(f'\nAnalyzing for path type: {path_type}')
            analysis_dirs = split_data(analysis_path, path_type)
            records = [_load_trajectory_record(analysis_path, traj_dir) for traj_dir in analysis_dirs]
            full_metrics = _aggregate_records(records)
            _log_metric_profile("saved full trajectory", full_metrics, include_cr=False)

            missing_metadata = [record["name"] for record in records if record["collision_events"] is None]
            if missing_metadata:
                raise RuntimeError(
                    f"{len(missing_metadata)}/{len(records)} trajectories lack collision_events metadata. "
                    "Run evaluation with the dual-condition evaluator in a fresh output directory before "
                    "requesting condition-specific metrics."
                )

            path_report = {"saved_full_trajectory": full_metrics}
            for condition_key, condition_label in COLLISION_CONDITIONS:
                metrics = _aggregate_records(records, condition_key=condition_key)
                _log_metric_profile(
                    f"truncate at first {condition_label} event",
                    metrics,
                    include_cr=True,
                )
                path_report[condition_key] = metrics
            collision_report["path_types"][path_type] = path_report

        report_path = os.path.join(analysis_path, "dual_collision_metrics.json")
        with open(report_path, "w") as report_file:
            json.dump(collision_report, report_file, indent=2)
        logging.info(f"Saved dual-condition metrics to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the evaluation results for trajectory prediction.")
    parser.add_argument('--root_dir', type=str, required=True, help="The root directory of the dataset.")
    parser.add_argument('--analysis_list', type=str, nargs='+', required=True, help="List of analysis items to process.")
    parser.add_argument('--path_type_list', type=str, nargs='+', required=True, help="List of path types to analyze.")

    args = parser.parse_args()

    analyze_results(args.root_dir, args.analysis_list, args.path_type_list)
