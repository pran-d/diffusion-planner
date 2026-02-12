import numpy as np
import torch
import os
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.flexible_dataset import FlexibleWindowDataset

class BufferDataset(FlexibleWindowDataset):
    """
    Dataset that loads trajectories from an in-memory buffer (dictionary of batched arrays).
    """
    def __init__(self, 
        data_buffer, # {key: (B, T, ...)}
        config, 
        feature_order=None, 
        norm_path=None, # Optional path to SAVE stats, or just use in memory
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
        
        # Index
        self.indices = []
        self._index_dataset()
        
        # Normalization
        self.norm_path = norm_path
        self.stats = {}
        if calculate_stats:
            self._calculate_stats()
            
    def _index_dataset(self):
        print("Indexing buffer dataset...")
        
        # Handle List Input by batching it immediately (User requested simplification)
        if isinstance(self.data_buffer, list):
            print(f"Batching {len(self.data_buffer)} trajectories from list...")
            batched = {}
            if len(self.data_buffer) > 0:
                keys = self.data_buffer[0].keys()
                for k in keys:
                    try:
                        # Stack to (B, T, ...)
                        batched[k] = np.stack([item[k] for item in self.data_buffer], axis=0)
                    except ValueError as e:
                        print(f"Warning: Could not stack key {k}: {e}")
            self.data_buffer = batched
            
        data = self.data_buffer
        B, T = 1, 0
        
        if 'body_pos_w' in data:
            arr = data['body_pos_w']
            if arr.ndim == 4: # (B, T, K, 3)
                B, T = arr.shape[0], arr.shape[1]
            else:
                 # If passed non-batched (T, ...), assume 1 batch
                 T = arr.shape[0]
                 B = 1
        elif 'base_xyz_quat' in data:
             arr = data['base_xyz_quat']
             if arr.ndim == 3: # (B, T, D)
                 B, T = arr.shape[0], arr.shape[1]
             else:
                 T = arr.shape[0]
                 B = 1
        else:
            raise ValueError("Data buffer missing recognized keys (body_pos_w or base_xyz_quat)")
            
        T_down = T // self.downsample
        min_start = self.start_timestep // self.downsample
        w_size = self.window_size + self.history_size
        max_start = max(min_start, T_down - w_size)
        
        if max_start >= min_start:
             starts = np.arange(min_start, max_start + 1, self.stride)
             for b in range(B):
                 for s in starts:
                     # (dummy_file_idx=0, batch_idx=b, t_start=s)
                     self.indices.append((0, b, s))
                     
        print(f"Indexed {len(self.indices)} windows from buffer.")

    def _load_raw_trajectory(self, dummy_fpath, batch_idx):
        """
        Load trajectory from buffer at batch_idx.
        Ignores dummy_fpath.
        """
        raw = {}
        data = self.data_buffer
        
        if 'body_pos_w' in data:
            # Check if batched (4D)
            is_batched = data['body_pos_w'].ndim == 4
            
            def extract(key):
                arr = data[key]
                if is_batched:
                    arr = arr[batch_idx]
                return arr[::self.downsample]
                
            body_pos = extract('body_pos_w')
            body_quat = extract('body_quat_w')
            raw['base'] = np.concatenate([body_pos[:, 0, :], body_quat[:, 0, :]], axis=-1)
            
            raw['joints'] = extract('joint_pos')
            
            obj_pos = extract('object_pos_w')
            obj_quat = extract('object_quat_w')
            raw['obj'] = np.concatenate([obj_pos, obj_quat], axis=-1)
            
        elif 'base_xyz_quat' in data:
             is_batched = data['base_xyz_quat'].ndim == 3
             def extract(key):
                arr = data[key]
                if is_batched:
                    arr = arr[batch_idx]
                return arr[::self.downsample]

             raw['base'] = extract('base_xyz_quat')
             raw['joints'] = extract('actuator_pos')
             raw['obj'] = extract('obj_0_xyz_quat')
             
        # Add task params if present in buffer
        if 'task_params' in data:
             tp = data['task_params']
             if tp.ndim == 2: # (B, D)
                 raw['task_params'] = tp[batch_idx]
             else:
                 raw['task_params'] = tp
                 
        return raw

    def _compute_transform(self, raw_data, t_start, task_params=None):
        """
        Override to allow injecting external task parameters (e.g. goal condition).
        """
        feats, anchor = super()._compute_transform(raw_data, t_start)
        
        if task_params is not None:
             feats['task_params'] = task_params
             anchor['task_params'] = task_params
             
        return feats, anchor

    def __getitem__(self, idx):
        # Override to ignore file path lookups
        _, batch_idx, t_start = self.indices[idx]
        
        # No fpath needed
        raw_traj = self._load_raw_trajectory(None, batch_idx)
        features, anchor = self._compute_transform(raw_traj, t_start)
        
        window_parts = []
        for key in self.feature_order:
            if key in features:                
                part = torch.from_numpy(features[key]).float() 
                if self.add_noise:
                    history_slice = part[:self.history_size] 
                    part[:self.history_size] = self._add_obs_noise(history_slice, key)
                part = self._normalize(key, part)
                window_parts.append(part)
            else:
                raise ValueError(f"Feature {key} needed but not computed.")
                
        window_tensor = torch.cat(window_parts, dim=-1)
        
        current_state = window_tensor[:self.history_size, (self.num_features-self.num_observations):].clone()
        future_states = window_tensor[self.history_size:, :].clone()

        task_params = torch.from_numpy(features["task_params"]).float()
        if self.add_goal_noise:
            task_params = self._add_obs_noise(task_params, "task_params")
        task_params = self._normalize("task_params", task_params)

        return future_states, current_state, task_params, anchor

    def _calculate_stats(self):
        print("Calculating normalization stats from buffer (min/max)...")
        mins = {}
        maxs = {}
        
        # Iterate all indices
        for idx in tqdm(range(len(self.indices))):
            _, batch_idx, t_start = self.indices[idx]
            raw_traj = self._load_raw_trajectory(None, batch_idx)
            feats, _ = self._compute_transform(raw_traj, t_start)
            
            for k, v in feats.items():
                v = v.astype(np.float64)
                v_min = np.min(v, axis=0)
                v_max = np.max(v, axis=0)
                
                if k not in mins:
                    mins[k] = v_min
                    maxs[k] = v_max
                else:
                    mins[k] = np.minimum(mins[k], v_min)
                    maxs[k] = np.maximum(maxs[k], v_max)
        
        self.stats = {}
        for k in mins:
            self.stats[f"min_{k}"] = torch.as_tensor(mins[k]).float()
            self.stats[f"max_{k}"] = torch.as_tensor(maxs[k]).float()
            
        # Also compute global stats for denormalize_global
        self._compute_global_stats()
        
    def _compute_global_stats(self):
        mins = []
        maxs = []
        for key in self.feature_order:
             mk, MK = f"min_{key}", f"max_{key}"
             if mk in self.stats and MK in self.stats:
                 mins.append(self.stats[mk])
                 maxs.append(self.stats[MK])
        
        if mins:
            self.global_min = torch.cat(mins, dim=-1)
            self.global_max = torch.cat(maxs, dim=-1)
