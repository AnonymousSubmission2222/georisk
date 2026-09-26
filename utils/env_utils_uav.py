import math
import numba as nb
import airsim
import numpy as np
import copy

from src.common.param import args

from utils.logger import logger


class SimState:
    def __init__(self, index=-1,                                                                                                                                                        
                 step=0,
                 raw_trajectory_info={},
                 ):
        self.index = index
        self.step = step
        self.raw_trajectory_info = copy.deepcopy(raw_trajectory_info) 
        self.trajectory = [{'sensor': {'state':self.raw_trajectory_info['trajectory'][0]}}] 
        self.is_end = False
        self.oracle_success = False
        self.is_collisioned = False
        self.termination_reason = None
        self.termination_detail = None
        self.stuck_streak = 0
        self.predict_start_index = 0
        self.history_start_indexes = [0]
        self.SUCCESS_DISTANCE = 20
        self.pre_carrot_idx = 0
        self.start_point_nearest_node_token = None
        self.end_point_nearest_node_token = None
        self.progress = 0.0
        self.waypoint = {}
        self.unique_path = None


    @property
    def state(self): 
        return self.trajectory[-1]['sensors']['state']

    @property
    def pose(self): 
        return self.trajectory[-1]['sensors']['state']['position'] + self.trajectory[-1]['sensors']['state']['orientation']

class ENV:
    def __init__(self, load_scenes: list):
        self.batch = None

    def set_batch(self, batch):
        self.batch = copy.deepcopy(batch)
        return

    def get_obs_at(self, index: int, state: SimState):
        assert self.batch is not None, 'batch is None'
        item = self.batch[index]
        oracle_success = state.oracle_success

        teacher_action_path = None
        done = state.is_end or state.termination_reason is not None

        return (teacher_action_path, done, oracle_success), state

