import json
from src.vlnce_src.sensor_json import write_sensor_json
import random
import shutil

import cv2
import numpy as np
from utils.utils import *
from src.common.param import args
import torch.backends.cudnn as cudnn
from src.vlnce_src.env_uav import AirVLNENV, RGB_FOLDER, DEPTH_FOLDER
from src.vlnce_src.collision_events import (
    AIRSIM_COLLISION_EVENT,
    DEPTH_COLLISION_EVENT,
    DEPTH_ENCODED_THRESHOLD,
    DEPTH_FRACTION_THRESHOLD,
    DUAL_COLLISION_REASON,
    evaluate_encoded_depth_collision,
)


def setup(manual_init_distributed_mode=False):
    if not manual_init_distributed_mode:
        init_distributed_mode()

    seed = 100 + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = False

def CheckPort():
    pid = FromPortGetPid(int(args.DDP_MASTER_PORT))
    if pid is not None:
        print('DDP_MASTER_PORT ({}) is being used'.format(args.DDP_MASTER_PORT))
        return False

    return True


def initialize_env_eval(dataset_path, save_path, eval_json_path):
    train_env = AirVLNENV(batch_size=args.batchSize, dataset_path=dataset_path, save_path=save_path, eval_json_path=eval_json_path)
    return train_env

        
def save_to_dataset_eval(episodes, path, ori_traj_dir, termination_info=None):
    root_path = os.path.join(path)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    folder_names = ['log'] + RGB_FOLDER + DEPTH_FOLDER
    for folder_name in folder_names:
        os.makedirs(os.path.join(root_path, folder_name), exist_ok=True)
    print(root_path)
    save_logs(episodes, root_path)
    save_images(episodes, root_path)

    ori_obj = os.path.join(ori_traj_dir, 'object_description.json')
    target_obj = os.path.join(root_path, 'object_description.json')
    shutil.copy2(ori_obj, target_obj)
    ori_info = {'ori_traj_dir': ori_traj_dir}
    if termination_info is not None:
        ori_info['termination_info'] = termination_info
        ori_info['stop_dist_thresh_met'] = bool(termination_info.get('stop_dist_thresh_met', False))
        ori_info['stop_prob_thresh_met'] = bool(termination_info.get('stop_prob_thresh_met', False))
    with open(os.path.join(path, 'ori_info.json'), 'w') as f:
        json.dump(ori_info, f)
    if termination_info is not None:
        with open(os.path.join(path, 'termination_info.json'), 'w') as f:
            json.dump(termination_info, f)

def save_logs(episodes, trajectory_dir):
    save_dir = os.path.join(trajectory_dir, 'log')
    for idx, episode in enumerate(episodes):
        info = {'frame': idx, 'sensors': episode['sensors']}
        write_sensor_json(os.path.join(save_dir, str(idx).zfill(6) + '.json'), info)

def save_images(episodes, trajectory_dir):
    for idx, episode in enumerate(episodes):
        if 'rgb' in episode:
            for cid, camera_name in enumerate(RGB_FOLDER):
                image = episode['rgb'][cid]
                cv2.imwrite(os.path.join(trajectory_dir, camera_name, str(idx).zfill(6) + '.png'), image)
        if 'depth' in episode:
            for cid, camera_name in enumerate(DEPTH_FOLDER):
                image = episode['depth'][cid]
                cv2.imwrite(os.path.join(trajectory_dir, camera_name, str(idx).zfill(6) + '.png'), image)

def load_object_description():
    object_desc_dict = dict()
    with open(args.object_name_json_path, 'r') as f:
        file = json.load(f)
        for item in file:
            object_desc_dict[item['object_name']] = item['object_desc']
    return object_desc_dict

class BatchIterator:
    def __init__(self, env: AirVLNENV):
        self.env = env
    
    def __len__(self):
        return len(self.env.data)
    
    def __next__(self):
        batch = self.env.next_minibatch()
        if batch is None:
            raise StopIteration
        return batch
    
    def __iter__(self):
        batch = self.env.next_minibatch()
        if batch is None:
            raise StopIteration
        return batch


class EvalBatchState:
    def __init__(self, batch_size, env_batchs, env, assist):
        self.batch_size = batch_size
        self.eval_env = env
        self.assist = assist
        self.episodes = [[] for _ in range(batch_size)]
        
        
        self.is_padding = [bool(b.get('__pad__', False)) for b in env_batchs]
        self.target_positions = [b['object_position'] for b in env_batchs]
        self.object_infos = [self._get_object_info(b) for b in env_batchs]
        self.trajs = [b['trajectory'] for b in env_batchs]
        self.ori_data_dirs = [b['trajectory_dir'] for b in env_batchs]
        self.dones = [False] * batch_size
        self.predict_dones = [False] * batch_size
        self.predict_done_details = [{} for _ in range(batch_size)]
        self.collisions = [False] * batch_size
        self.termination_reasons = [None] * batch_size
        self.termination_details = [None] * batch_size
        self.success = [False] * batch_size
        self.oracle_success = [False] * batch_size
        self.early_end = [False] * batch_size
        
        self.skips = self.is_padding.copy()
        self.distance_to_ends = [[] for _ in range(batch_size)]
        self.step_positions = [[] for _ in range(batch_size)]
        self.step_distances = [[] for _ in range(batch_size)]
        self.collision_events = [
            {AIRSIM_COLLISION_EVENT: None, DEPTH_COLLISION_EVENT: None}
            for _ in range(batch_size)
        ]
        self.envs_to_pause = []
        
        self._initialize_batch_data()

    def _get_object_info(self, batch):
        object_desc_dict = self._load_object_description()
        return object_desc_dict.get(batch['object']['asset_name'].replace("AA", ""))

    def _load_object_description(self):
        with open(args.object_name_json_path, 'r') as f:
            return {item['object_name']: item['object_desc'] for item in json.load(f)}

    def _initialize_batch_data(self):
        outputs = self.eval_env.reset()
        observations, self.dones, self.collisions, self.oracle_success = [list(x) for x in zip(*outputs)]
        
        for i in range(self.batch_size):
            if i in self.envs_to_pause:
                continue
            self.episodes[i].append(observations[i][-1])
            self.distance_to_ends[i].append(self._calculate_distance(observations[i][-1], self.target_positions[i]))
            self.termination_reasons[i] = observations[i][-1].get("termination_reason")
            self.termination_details[i] = observations[i][-1].get("termination_detail")
            self._update_collision_events(i, observations[i][-1], bool(self.collisions[i]))

    def _calculate_distance(self, observation, target_position):
        return np.linalg.norm(np.array(observation['sensors']['state']['position']) - np.array(target_position))

    def _get_position(self, observation):
        return np.asarray(observation['sensors']['state']['position'], dtype=np.float32)

    def _event_frame(self, i, observation):
        return {
            "triggered": True,
            "first_frame_index": int(len(self.episodes[i]) - 1),
            "first_action_step": int(len(self.step_positions[i])),
            "position": [float(x) for x in self._get_position(observation).tolist()],
        }

    def _record_airsim_collision(self, i, observation):
        if self.collision_events[i][AIRSIM_COLLISION_EVENT] is not None:
            return
        event = self._event_frame(i, observation)
        collision = observation.get("sensors", {}).get("state", {}).get("collision", {})
        if isinstance(collision, dict):
            event.update({
                "has_collided": bool(collision.get("has_collided", True)),
                "time_stamp": int(collision.get("time_stamp", 0) or 0),
                "object_name": str(collision.get("object_name", "") or ""),
            })
        else:
            event["has_collided"] = True
        self.collision_events[i][AIRSIM_COLLISION_EVENT] = event
        print(
            f"collision event: {AIRSIM_COLLISION_EVENT} batch={i} "
            f"frame={event['first_frame_index']}"
        )

    def _record_depth_collision(self, i, observation):
        depth_result = evaluate_encoded_depth_collision(
            observation.get("depth", []),
            DEPTH_FOLDER,
        )
        if (
            depth_result["triggered"]
            and self.collision_events[i][DEPTH_COLLISION_EVENT] is None
        ):
            event = self._event_frame(i, observation)
            event.update(depth_result)
            self.collision_events[i][DEPTH_COLLISION_EVENT] = event
            print(
                f"collision event: {DEPTH_COLLISION_EVENT} batch={i} "
                f"frame={event['first_frame_index']} "
                f"views={','.join(event['triggered_views'])}"
            )

    def _both_collision_events_triggered(self, i):
        return all(self.collision_events[i][key] is not None for key in (
            AIRSIM_COLLISION_EVENT,
            DEPTH_COLLISION_EVENT,
        ))

    def _clear_single_airsim_termination(self, i, observation):
        if self.termination_reasons[i] != "physical_collision":
            return
        self.dones[i] = False
        self.collisions[i] = False
        self.termination_reasons[i] = None
        self.termination_details[i] = None
        observation["termination_reason"] = None
        observation["termination_detail"] = None
        sim_state = self.eval_env.sim_states[i]
        sim_state.is_end = False
        sim_state.is_collisioned = False
        sim_state.termination_reason = None
        sim_state.termination_detail = None

    def _set_dual_collision_terminal(self, i, observation):
        detail = {
            "policy": "airsim_has_collided_and_depth_le1_gt10pct_v1",
            "events": self._serialized_collision_events(i),
        }
        self.dones[i] = True
        self.collisions[i] = True
        self.termination_reasons[i] = DUAL_COLLISION_REASON
        self.termination_details[i] = detail
        observation["termination_reason"] = DUAL_COLLISION_REASON
        observation["termination_detail"] = detail
        sim_state = self.eval_env.sim_states[i]
        sim_state.is_end = True
        sim_state.is_collisioned = True
        sim_state.termination_reason = DUAL_COLLISION_REASON
        sim_state.termination_detail = detail
        self._classify_terminal_by_trajectory(i, allow_success=False)

    def _serialized_collision_events(self, i):
        return {
            AIRSIM_COLLISION_EVENT: (
                self.collision_events[i][AIRSIM_COLLISION_EVENT]
                or {"triggered": False}
            ),
            DEPTH_COLLISION_EVENT: (
                self.collision_events[i][DEPTH_COLLISION_EVENT]
                or {"triggered": False}
            ),
        }

    def _update_collision_events(self, i, observation, airsim_triggered):
        if airsim_triggered:
            self._record_airsim_collision(i, observation)
        self._record_depth_collision(i, observation)

        if self._both_collision_events_triggered(i):
            self._set_dual_collision_terminal(i, observation)
        elif airsim_triggered:
            
            
            self._clear_single_airsim_termination(i, observation)

    def _classify_terminal_by_trajectory(self, i, allow_success=False):
        cur_dist = float(self.distance_to_ends[i][-1]) if len(self.distance_to_ends[i]) > 0 else float("inf")
        min_dist_hist = (
            float(np.min(np.asarray(self.distance_to_ends[i], dtype=np.float32)))
            if len(self.distance_to_ends[i]) > 0
            else cur_dist
        )
        success_radius = 20.0
        self.success[i] = bool(allow_success and cur_dist <= success_radius)
        ever_enter_success_radius = bool(min_dist_hist <= success_radius)
        oracle_hit = bool(self.oracle_success[i] or ever_enter_success_radius)
        self.oracle_success[i] = bool((not self.success[i]) and oracle_hit)

    def _recent_position_radius(self, positions):
        points = np.asarray(positions, dtype=np.float32)
        if len(points) == 0:
            return float("inf"), None
        
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        candidate_radii = distances.max(axis=1)
        best_idx = int(np.argmin(candidate_radii))
        return float(candidate_radii[best_idx]), points[best_idx].tolist()

    def _recent_distances_all_moving_away(self, distances):
        if len(distances) < 30:
            return False
        tail = [float(x) for x in distances[-30:]]
        return all(tail[j] > tail[j - 1] for j in range(1, len(tail)))

    def _update_spatial_early_stop(self, i):
        if self.termination_reasons[i] is not None:
            return
        if len(self.step_positions[i]) >= 30:
            recent_positions = self.step_positions[i][-30:]
            radius, center = self._recent_position_radius(recent_positions)
            if radius <= 30.0:
                self.termination_reasons[i] = "local_stagnation"
                self.termination_details[i] = {
                    "window_steps": 30,
                    "radius_threshold_m": 30.0,
                    "observed_radius_m": radius,
                    "center": center,
                }
                self._classify_terminal_by_trajectory(i, allow_success=False)
                self.dones[i] = True
                return
        if self._recent_distances_all_moving_away(self.step_distances[i]):
            self.termination_reasons[i] = "moving_away_30"
            self.termination_details[i] = {
                "window_steps": 30,
                "first_distance_m": float(self.step_distances[i][-30]),
                "last_distance_m": float(self.step_distances[i][-1]),
            }
            self._classify_terminal_by_trajectory(i, allow_success=False)
            self.dones[i] = True

    def update_from_env_output(self, outputs):
        observations, self.dones, self.collisions, self.oracle_success = [list(x) for x in zip(*outputs)]
        for i in range(self.batch_size):
            if i in self.envs_to_pause:
                continue
            for j in range(len(observations[i])):
                self.episodes[i].append(observations[i][j])
            self.distance_to_ends[i].append(self._calculate_distance(observations[i][-1], self.target_positions[i]))
            self.termination_reasons[i] = observations[i][-1].get("termination_reason")
            self.termination_details[i] = observations[i][-1].get("termination_detail")
            self.step_positions[i].append(self._get_position(observations[i][-1]))
            self.step_distances[i].append(float(self.distance_to_ends[i][-1]))
            self._update_collision_events(i, observations[i][-1], bool(self.collisions[i]))
            self._update_spatial_early_stop(i)

    def get_assist_notices(self):
        return self.assist.get_assist_notice(self.episodes, self.trajs, self.object_infos, self.target_positions)

    def update_metric(self):
        for i in range(self.batch_size):
            if self.dones[i]:
                continue
            if self.predict_dones[i] and not self.skips[i]:
                self._classify_terminal_by_trajectory(i, allow_success=True)
                self.dones[i] = True
                    
    def check_batch_termination(self, t):
        for i in range(self.batch_size):
            if t == args.maxWaypoints:
                self.dones[i] = True
            if self.dones[i] and not self.skips[i]:
                predict_detail = {}
                if isinstance(self.predict_done_details, list) and i < len(self.predict_done_details):
                    if isinstance(self.predict_done_details[i], dict):
                        predict_detail = self.predict_done_details[i]
                stop_prob_thresh_met = bool(predict_detail.get("stop_prob_triggered", False))
                done_reasons = []
                terminal_reason = self.termination_reasons[i]
                if terminal_reason is not None:
                    done_reasons.append(terminal_reason)
                elif self.collisions[i]:
                    done_reasons.append("physical_collision")
                if self.predict_dones[i]:
                    if self.success[i]:
                        done_reasons.append("successful_stop")
                    else:
                        done_reasons.append("premature_stop")
                if stop_prob_thresh_met:
                    done_reasons.append("stop_prob_thresh_met")
                if self.success[i]:
                    done_reasons.append("success")
                if self.oracle_success[i] and terminal_reason != "sim_error":
                    done_reasons.append("oracle_success")
                if t == args.maxWaypoints:
                    done_reasons.append("max_waypoints")
                if not done_reasons:
                    done_reasons.append("done_flag")

                self.envs_to_pause.append(i)
                prex = ''
                if self.success[i]:
                    prex = 'success_'
                    print(i, " has succeed!")
                elif terminal_reason == "sim_error":
                    prex = "sim_error_"
                    print(i, " has simulator error; exclude from metrics and rerun.")
                elif self.oracle_success[i]:
                    prex = "oracle_"
                    print(i, " has oracle succeed!")
                new_traj_name = prex +  self.ori_data_dirs[i].split('/')[-1]
                new_traj_dir = os.path.join(args.eval_save_path, new_traj_name)
                termination_info = {
                    "done_reasons": done_reasons,
                    "terminal_reason": terminal_reason,
                    "terminal_detail": self.termination_details[i],
                    "stop_dist_thresh_met": False,
                    "stop_prob_thresh_met": stop_prob_thresh_met,
                    "stop_mode": predict_detail.get("stop_mode"),
                    "predict_done": bool(self.predict_dones[i]),
                    "predict_done_by_dino": predict_detail.get("predict_done_by_dino"),
                    "predict_done_by_stop": predict_detail.get("predict_done_by_stop"),
                    "pred_stop_prob": predict_detail.get("pred_stop_prob"),
                    "stop_prob_threshold": predict_detail.get("stop_prob_threshold"),
                    "stop_prob_consecutive": predict_detail.get("stop_prob_consecutive"),
                    "stop_prob_streak": predict_detail.get("stop_prob_streak"),
                    "collision_stop_policy": "airsim_has_collided_and_depth_le1_gt10pct_v1",
                    "collision_both_triggered": bool(self._both_collision_events_triggered(i)),
                    "collision_events": self._serialized_collision_events(i),
                    "depth_collision_config": {
                        "views": list(DEPTH_FOLDER),
                        "encoded_depth_threshold": int(DEPTH_ENCODED_THRESHOLD),
                        "fraction_threshold": float(DEPTH_FRACTION_THRESHOLD),
                        "comparison": "encoded_depth <= 1 fraction > 0.10 in any view",
                    },
                    "target_position": [float(x) for x in self.target_positions[i]],
                }
                save_to_dataset_eval(
                    self.episodes[i],
                    new_traj_dir,
                    self.ori_data_dirs[i],
                    termination_info=termination_info,
                )
                self.skips[i] = True
                print(
                    i,
                    " has finished! reason:",
                    ",".join(done_reasons),
                    "| stop_prob_thresh_met=",
                    stop_prob_thresh_met,
                    "| pred_stop_prob=",
                    predict_detail.get("pred_stop_prob"),
                    "| stop_prob_threshold=",
                    predict_detail.get("stop_prob_threshold"),
                    "| stop_prob_streak=",
                    predict_detail.get("stop_prob_streak"),
                    "/",
                    predict_detail.get("stop_prob_consecutive"),
                )
        return np.array(self.skips).all()
