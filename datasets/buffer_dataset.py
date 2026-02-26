import numpy as np
import torch
import os
from torch.utils.data import Dataset
from tqdm import tqdm

from .flexible_dataset import FlexibleWindowDataset
from utils.math.sbto_utils import compute_task_params
from utils.math.math_tools import yaw_from_quat, yaw_to_rot_matrix

class BufferDataset(FlexibleWindowDataset):
    """
    Dataset that loads trajectories from an in-memory buffer instead of files.
    Inherits all feature computation, normalization, and windowing from FlexibleWindowDataset.

    Buffer format:
        - dict of arrays: {key: (B, T, D)} or {key: list of (T, D)}
        - list of dicts:  [{key: (T, D)}, ...]

    Supported schemas (same as FlexibleWindowDataset):
        - RL Rollout: body_pos_w, body_quat_w, joint_pos, object_pos_w, object_quat_w
        - SBTO:       base_xyz_quat, actuator_pos, obj_0_xyz_quat
    """
    def __init__(self, 
        data_buffer, 
        config, 
        task_params=None,
        norm_path=None, 
        calculate_stats=True, 
        training_cfg=None,
    ):
        
        super().__init__(
            data_buffer=data_buffer,
            config=config,
            task_params=task_params,
            norm_path=norm_path,
            calculate_stats=calculate_stats,
            training_cfg=training_cfg,
        )