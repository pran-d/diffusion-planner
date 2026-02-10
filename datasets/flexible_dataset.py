import numpy as np
from typing import List, Dict, Union, Optional
import torch
import os
import glob
import re
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.math.sbto_utils import compute_sbto_components
from utils.math.math_tools import yaw_from_quat, yaw_to_rot_matrix
from utils.math.rotation_conversions import (
    rotation_6d_to_matrix, 
    matrix_to_rotation_6d, 
    axis_angle_to_quaternion, 
    quaternion_to_matrix,
    matrix_to_quaternion
)

class FlexibleWindowDataset(Dataset):
    """
    Dataset that loads raw absolute trajectories, windows them, 
    transforms to relative coordinates on-the-fly, 
    and assembles features based on a configurable order.
    """
    def __init__(self, 
                 data_root=None, 
                 config=None, 
                 feature_order=None, 
                 norm_path=None, 
                 calculate_stats=False,
                 noise_cfg=None, 
                 add_noise=False,
                 add_goal_noise=False,
                 data_buffer: Optional[List[Dict]] = None,
                 task_params_buffer: Optional[List[Dict]] = None):
        
        self.data_root = data_root
        self.data_buffer = data_buffer
        self.task_params_buffer = task_params_buffer
        self.noise_cfg = noise_cfg or {}
        self.num_observations = config.get("num_observations", 45)
        self.num_features = config.get("num_features", 48)
        self.window_size = config.get("window_size", 50)
        self.history_size = config.get("state_history", 4)
        self.stride = config.get("stride", 1)
        self.downsample = config.get("downsample", 1)
        self.add_noise = add_noise
        self.add_goal_noise = add_goal_noise
        
        # Default feature order if not provided
        self.feature_order = feature_order or [
            "delta_xy", "delta_yaw", 
            "joints", "body_z", "body_rot6d",
            "obj_rel_pos", "obj_rel_rot6d"
        ]
        
        # Load file list
        if self.data_buffer is None:
            if not self.data_root:
                raise ValueError("Either data_root or data_buffer must be provided.")
            self.file_paths = sorted(glob.glob(os.path.join(data_root, "*.npz")))
            if not self.file_paths:
                print(f"Warning: No .npz files found in {data_root}")
            
            # Load files into memory buffer
            print(f"Loading {len(self.file_paths)} files into memory...")
            self.data_buffer = []
            for fpath in tqdm(self.file_paths, desc="Loading Data"):
                with np.load(fpath, allow_pickle=True) as data:
                    self.data_buffer.append({k: data[k] for k in data.files})
        else:
            self.file_paths = [] 
        
        # Index all windows: (file_idx, start_time_idx)
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
        """
        Iterate through buffer to determine valid window start indices.
        """
        print("Indexing dataset...")
        
        # Index from memory
        for i, data in enumerate(self.data_buffer):
            try:
                # Identify Length & Batch
                B, T = 1, 0
                if 'body_pos_w' in data:
                    arr = data['body_pos_w']
                    if arr.ndim == 4: # (N, T, K, 3)
                        B, T = arr.shape[0], arr.shape[1]
                    else:
                        T = arr.shape[0]
                else:
                    continue 
                
                # Apply downsampling
                T_down = T // self.downsample
                min_start = 0
                max_start = T_down - 1
                                    
                if max_start >= min_start:
                    starts = np.arange(min_start, max_start + 1, self.stride)
                    for b in range(B):
                        for s in starts:
                            self.indices.append((i, b, s))
            except Exception as e:
                print(f"Error indexing buffer item {i}: {e}")
        print(f"Indexed {len(self.indices)} windows.")

    def _load_raw_trajectory(self, buffer_idx, batch_idx):
        """Generalized loader for different schema."""
        raw = {}
        
        data = self.data_buffer[buffer_idx]
        
        # Helper to extract from data dict or file
        def process_data(data_dict):
            is_batched = False
            if 'body_pos_w' in data_dict:
                # Determine batching
                is_batched = data_dict['body_pos_w'].ndim == 4
                
                def extract(key):
                    arr = data_dict[key]
                    if is_batched:
                        arr = arr[batch_idx]
                    return arr[::self.downsample]
                
                # Base
                body_pos = extract('body_pos_w') # (T, K, 3)
                body_quat = extract('body_quat_w')
                raw['base'] = np.concatenate([body_pos[:, 0, :], body_quat[:, 0, :]], axis=-1)
                
                # Joints
                raw['joints'] = extract('joint_pos')
                
                # Object
                obj_pos = extract('object_pos_w')
                obj_quat = extract('object_quat_w')
                raw['obj'] = np.concatenate([obj_pos, obj_quat], axis=-1)

        process_data(data)
        
        if self.task_params_buffer is not None:
            # Extract task params
            tp_data = self.task_params_buffer[buffer_idx]

            # We need to construct the final vector.
            # If tp_data is dict:
            vals = []
            if isinstance(tp_data, dict):
                for k, v in tp_data.items():
                    arr = np.array(v)
                    # Use is_batched status from data directly
                    if 'body_pos_w' in data and data['body_pos_w'].ndim == 4:
                        arr = arr[batch_idx]
                    
                    if arr.ndim == 0: arr = arr[None]
                    vals.append(arr)
                raw['task_params'] = np.concatenate(vals, axis=-1)
            else:
                # Assume it is the array directly
                arr = np.array(tp_data)
                if 'body_pos_w' in data and data['body_pos_w'].ndim == 4:
                    arr = arr[batch_idx]
                raw['task_params'] = arr
        
        if 'task_params' not in raw:
             pass 

        return raw

    def _compute_task_params(self, obj_seq):
        """
        Compute task params from object sequence if strictly needed.
        Default: return final object pose (goal).
        """
        if len(obj_seq) > 0:
            return obj_seq[-1]
        return np.zeros(7)

    def _compute_transform(self, raw_data, t_start, task_params=None):
        """
        Compute relative features for the window starting at t_start.
        Returns a dictionary of available features.
        """
        w_size = self.window_size + self.history_size
        
        # Handle boundary/padding
        read_start = max(0, t_start)
        read_end = t_start + w_size
        
        base = raw_data['base'][read_start : read_end]      # (valid_len, 7)
        joints = raw_data['joints'][read_start : read_end]  # (valid_len, 29)
        obj = raw_data['obj'][read_start : read_end]        # (valid_len, 7)
        
        # 1. Pad Front (if t_start < 0)
        pad_front = max(0, -t_start)
        if pad_front > 0:
            base = np.pad(base, ((pad_front, 0), (0,0)), mode='edge')
            joints = np.pad(joints, ((pad_front, 0), (0,0)), mode='edge')
            obj = np.pad(obj, ((pad_front, 0), (0,0)), mode='edge')
        
        # 2. Pad Back (if total length < w_size)
        curr_len = base.shape[0]
        if curr_len < w_size:
            pad_back = w_size - curr_len
            base = np.pad(base, ((0, pad_back), (0,0)), mode='edge')
            joints = np.pad(joints, ((0, pad_back), (0,0)), mode='edge')
            obj = np.pad(obj, ((0, pad_back), (0,0)), mode='edge')
            
        # Define Reference Frame (at history_size - 1)
        # This is the "current state" anchor
        ref_idx = min(self.history_size - 1, w_size - 1)
        
        # --- Computations (SBTO Logic) ---
        # Use centralized logic from sbto_utils
        comps, anchors = compute_sbto_components(
            base=base[None, ...],   # (1, W, 7)
            joints=joints[None, ...], 
            obj=obj[None, ...],
            ref_idx=ref_idx
        )
        
        # Unpack to (W, ...)
        feats = {k: v[0] for k, v in comps.items()}

        if task_params is None:
            if 'task_params' in raw_data:
                task_params = raw_data['task_params']
            else:
                task_params = self._compute_task_params(raw_data['obj'])
            
        feats['task_params'] = task_params
        
        anchor = {}
        anchor['ref_pos'] = anchors['ref_pos'][0, 0]
        anchor['ref_quat'] = anchors['ref_quat'][0, 0]
        anchor['task_params'] = task_params

        return feats, anchor
    
    def _normalize(self, key, tensor):
        """Apply mean-std normalization."""
        if not self.stats:
            return tensor
            
        mean_k = f"mean_{key}"
        std_k = f"std_{key}"
        
        if mean_k in self.stats and std_k in self.stats:
            mean = self.stats[mean_k].to(tensor.device)
            std = self.stats[std_k].to(tensor.device)
            return (tensor - mean) / (std + 1e-6)
            
        return tensor
    
    def _add_rotation_noise(self, tensor, noise_level):
        """Add noise to rotation features (e.g., 6D)."""
        # tensor: (T, 6)
        
        # 1. Convert to Rotation Matrix (T, 3, 3)
        rot_mat = rotation_6d_to_matrix(tensor)
        
        # 2. Generate random axis-angle noise
        # axis_angle = randn * noise_level represents a random rotation vector
        perturbation_axis_angle = torch.randn_like(tensor[:, :3]) * noise_level
        
        # 3. Convert perturbation to Matrix
        perturbation_quat = axis_angle_to_quaternion(perturbation_axis_angle)
        perturbation_mat = quaternion_to_matrix(perturbation_quat)
        
        # 4. Apply perturbation: R_new = R * R_perturbation (Local perturbation)
        rot_mat_noisy = torch.bmm(rot_mat, perturbation_mat)
        
        # 5. Convert back to 6D
        tensor_noisy = matrix_to_rotation_6d(rot_mat_noisy)
        
        return tensor_noisy
    
    def _add_obs_noise(self, tensor, key):
        """Add noise based on config."""
        if key in ["obj_rel_rot6d", "body_rot6d"]:
            return self._add_rotation_noise(tensor, noise_level=self.noise_cfg[key])
        elif key in self.noise_cfg:
            noise_level = self.noise_cfg[key]
            noise = torch.randn_like(tensor) * noise_level
            return tensor + noise
        return tensor

    def __getitem__(self, idx):
        file_idx, batch_idx, t_start = self.indices[idx]
        
        if self.data_buffer is not None:
            # Use file_idx as index into buffer
            raw_traj = self._load_raw_trajectory(file_idx, batch_idx)
        else:
            fpath = self.file_paths[file_idx]
            raw_traj = self._load_raw_trajectory(fpath, batch_idx)
        
        features, anchor = self._compute_transform(raw_traj, t_start)
        
        # Assemble Windowed Feature Vector
        window_parts = []
        for key in self.feature_order:
            if key in features:                
                part = torch.from_numpy(features[key]).float() 
                if self.add_noise:
                    history_slice = part[:self.history_size] 
                    part[:self.history_size] = self._add_obs_noise(history_slice, key)
                part = self._normalize(key, part)
                part = part 
                window_parts.append(part)
            else:
                raise ValueError(f"Feature {key} needed but not computed.")
                
        window_tensor = torch.cat(window_parts, dim=-1) # (W, Total_Dim)
        
        # Current State 
        current_state = window_tensor[:self.history_size, (self.num_features-self.num_observations):].clone()

        # Future trajectory
        future_states = window_tensor[self.history_size:, :].clone()

        task_params = torch.from_numpy(features["task_params"]).float()
        if self.add_goal_noise:
            task_params = self._add_obs_noise(task_params, "task_params")
        task_params = self._normalize("task_params", task_params)

        return future_states, current_state, task_params, anchor

    def _calculate_stats(self):
        """Iterate entire dataset to compute mean/std for each feature type."""
        print("Calculating normalization stats (mean/std)...")
        sums = {}
        sq_sums = {}
        counts = {}
        
        # Use a subset if dataset is huge, else full pass
        for idx in tqdm(range(len(self.indices))):
            file_idx, batch_idx, t_start = self.indices[idx]
            
            if self.data_buffer is not None:
                raw_traj = self._load_raw_trajectory(file_idx, batch_idx)
            else:
                raw_traj = self._load_raw_trajectory(self.file_paths[file_idx], batch_idx)

            feats, _ = self._compute_transform(raw_traj, t_start)
            
            for k, v in feats.items():
                # Apply noise during stats calculation if enabled
                v = v.astype(np.float64) # Use float64 for precision accumulation
                v_sum = np.sum(v, axis=0)
                v_sq_sum = np.sum(v**2, axis=0)
                v_count = v.shape[0]
                
                if k not in sums:
                    sums[k] = v_sum
                    sq_sums[k] = v_sq_sum
                    counts[k] = v_count
                else:
                    sums[k] += v_sum
                    sq_sums[k] += v_sq_sum
                    counts[k] += v_count
        
        # Store
        self.stats = {}
        for k in sums:
            mean = sums[k] / counts[k]
            # Var = E[X^2] - (E[X])^2
            var = (sq_sums[k] / counts[k]) - (mean ** 2)
            var = np.maximum(var, 0) # Clip negative 0
            std = np.sqrt(var)
           
            # Clip small std to prevent division explosion
            std = np.maximum(std, 1e-4)

            self.stats[f"mean_{k}"] = torch.as_tensor(mean).float()
            self.stats[f"std_{k}"] = torch.as_tensor(std).float()
            
        # Save
        if self.norm_path:
            os.makedirs(os.path.dirname(self.norm_path), exist_ok=True)
            # Save as numpy dict
            np_stats = {k: v.numpy() for k, v in self.stats.items()}
            np.savez(self.norm_path, **np_stats)
            print(f"Stats saved to {self.norm_path}")

    def _load_stats(self):
        print(f"Loading stats from {self.norm_path}")
        with np.load(self.norm_path, allow_pickle=True) as data:
            for k in data.keys():
                self.stats[k] = torch.from_numpy(data[k]).float()
        
        # Precompute global stats for concatenated features
        means = []
        stds = []
        for key in self.feature_order:
             # Check if key exists in stats (it should if we computed it)
             # Note: stats keys are "mean_key", "std_key"
             mk, sk = f"mean_{key}", f"std_{key}"
             if mk in self.stats and sk in self.stats:
                 means.append(self.stats[mk])
                 stds.append(self.stats[sk])
             else:
                 # Warn or handle missing
                 pass
        
        if means:
            self.global_mean = torch.cat(means, dim=-1)
            self.global_std = torch.cat(stds, dim=-1)

    def denormalize_global(self, tensor):
        """Denormalize a concatenated feature tensor using global stats."""
        if hasattr(self, 'global_mean') and hasattr(self, 'global_std'):
            device = tensor.device
            mean = self.global_mean.to(device)
            std = self.global_std.to(device)
            return (tensor * std) + mean
        return tensor

    def __len__(self):
        return len(self.indices)
