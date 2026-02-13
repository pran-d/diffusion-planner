import numpy as np
import torch
import os
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.flexible_dataset import FlexibleWindowDataset

class BufferDataset(FlexibleWindowDataset):
    """
    Dataset that loads trajectories from an in-memory buffer.
    Buffer can be:
    1. A dictionary of batched arrays {key: (B, T, ...)}
    2. A list of dictionaries [{key: (T, ...)}, ...]
    """
    def __init__(self, 
        data_buffer, 
        config, 
        feature_order=None, 
        norm_path=None, 
        calculate_stats=True, 
        noise_cfg=None, 
        add_noise=False,
        add_goal_noise=False
    ):
        
        self.data_buffer = data_buffer
        self.noise_cfg = noise_cfg or {}
        
        # Config Copy
        self.num_observations = config.get("num_observations", 45)
        self.num_features = config.get("num_features", 48)
        self.history_size = config.get("state_history", 4)
        self.window_size = config.get("num_timesteps", 50) // config.get("downsample", 1)
        self.stride = config.get("stride", 1)
        self.downsample = config.get("downsample", 1)
        self.start_timestep = config.get("start_timestep", 0)

        self.add_noise = add_noise
        self.add_goal_noise = add_goal_noise
        
        self.feature_order = feature_order or [
            "delta_xy", "delta_yaw", 
            "joints", "body_z", "body_rot6d",
            "obj_rel_pos", "obj_rel_rot6d"
        ]

        # Standardize data_buffer to dict of iterables (batched arrays or lists)
        if isinstance(self.data_buffer, list):
            if len(self.data_buffer) > 0:
                keys = self.data_buffer[0].keys()
                # Transpose list of dicts to dict of lists
                new_buffer = {k: [item.get(k) for item in self.data_buffer] for k in keys}
                self.data_buffer = new_buffer
            else:
                self.data_buffer = {}
        
        # Ensure we have a valid dictionary in the end
        if not isinstance(self.data_buffer, dict):
            raise ValueError("BufferDataset expects a list of dicts or a dict of batches.")

        self.file_paths = []
             
        # Index
        self.indices = []
        self._index_dataset()
        
        # Normalization
        self.norm_path = norm_path
        self.stats = {}
        if calculate_stats:
            self._calculate_stats()
        elif norm_path and os.path.exists(norm_path):
            self._load_stats()

    def _index_dataset(self):
        self.indices = []
        
        keys = list(self.data_buffer.keys())
        if not keys: return
        
        # Find a representative key to determine batch size / lengths
        # Prefer known temporal keys to be safe
        rep_key = keys[0]
        for k in ['body_pos_w', 'base_xyz_quat', 'joint_pos', 'actuator_pos']:
            if k in self.data_buffer:
                rep_key = k
                break
        
        data_coll = self.data_buffer[rep_key]
        num_trajs = len(data_coll)

        trajs = []
        for i in range(num_trajs):
            traj_data = data_coll[i]
            # Handle both numpy/torch tensors (shape[0]) and lists (len)
            T = traj_data.shape[0] if hasattr(traj_data, 'shape') else len(traj_data)
            trajs.append((i, T))
                
        # 2. Build Sliding Windows
        # Windows are defined in downsampled steps
        req_len_down = self.history_size + self.window_size
        
        for batch_idx, raw_T in trajs:
            down_T = raw_T // self.downsample
            
            if down_T < req_len_down:
                continue
            
            # Valid start indices (in downsampled space)
            last_start = down_T - req_len_down
            
            for t in range(0, last_start + 1, self.stride):
                # (file_idx=0 is dummy, batch_idx, t_start=t)
                self.indices.append((0, batch_idx, t))
            
    def _get_single_traj(self, file_idx, batch_idx):
        """
        Get processed (T, D) dict from buffer.
        """
        raw = {}
        
        # Helper to extract and downsample
        def extract_k(key):
            if key not in self.data_buffer: return None
            # Retrieve trajectory
            val = self.data_buffer[key][batch_idx]
            # Downsample along time dimension
            return val[::self.downsample]
        
        # Check standard keys
        keys = self.data_buffer.keys()
        
        # Case 1: Standard 'body_pos_w' format
        if 'body_pos_w' in keys:
            body_pos = extract_k('body_pos_w')     # (T, 3) or (T, 1, 3)
            body_quat = extract_k('body_quat_w')
            
            if body_pos is not None and body_pos.ndim == 3:
                # If shape is (T, 1, 3), squeeze.
                if body_pos.shape[1] == 1:
                    body_pos = body_pos[:, 0, :]
                else:
                    # Likely (T, NumBodies, 3).Let's take index 0 for base.
                    body_pos = body_pos[:, 0, :]

            if body_quat is not None and body_quat.ndim == 3:
                if body_quat.shape[1] == 1:
                     body_quat = body_quat[:, 0, :]
                else:
                     body_quat = body_quat[:, 0, :]
                
            raw['base'] = np.concatenate([body_pos, body_quat], axis=-1)
            raw['joints'] = extract_k('joint_pos')
            
            op = extract_k('object_pos_w')
            oq = extract_k('object_quat_w')
            raw['obj'] = np.concatenate([op, oq], axis=-1)
            
        # Case 2: 'base_xyz_quat' format (e.g. from simpler loggers)
        elif 'base_xyz_quat' in keys:
             raw['base'] = extract_k('base_xyz_quat')
             raw['joints'] = extract_k('actuator_pos')
             if 'obj_0_xyz_quat' in keys:
                 raw['obj'] = extract_k('obj_0_xyz_quat')
                 
        # Optional Task Params (can be passed in buffer)
        if 'task_params' in keys:
             # Typically (Batch, D) - no time dim, so no downsample needed
             # or (Batch, Time, D) - if time-varying
             tp = self.data_buffer['task_params'][batch_idx]
             raw['task_params'] = tp
                 
        return raw


    def _compute_transform(self, raw_data, t_start, task_params=None):
        """
        Override to inject task params from buffer if they exist
        """
        feats, anchor = super()._compute_transform(raw_data, t_start)
        
        if task_params is not None:
             feats['task_params'] = task_params
             anchor['task_params'] = task_params
        elif 'task_params' in raw_data:
             tp = raw_data['task_params']
             feats['task_params'] = tp
             anchor['task_params'] = tp
             
        return feats, anchor