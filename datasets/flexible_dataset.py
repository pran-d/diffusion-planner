import numpy as np
import torch
import os
import glob
import re
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.math.sbto_utils import compute_sbto_components
from utils.math.rotation_conversions import (
    rotation_6d_to_matrix, 
    matrix_to_rotation_6d, 
    axis_angle_to_quaternion, 
    quaternion_to_matrix
)

class FlexibleWindowDataset(Dataset):
    """
    Dataset that loads raw absolute trajectories, windows them, 
    transforms to relative coordinates on-the-fly, 
    and assembles features based on a configurable order.
    """
    def __init__(self, 
        data_root, 
        config, 
        feature_order=None, 
        norm_path=None, 
        calculate_stats=False,
        noise_cfg=None, 
        add_noise=False,
        add_goal_noise=False
    ):
        
        self.data_root = data_root
        self.noise_cfg = noise_cfg or {}
        self.num_observations = config.get("num_observations", 45)
        self.num_features = config.get("num_features", 48)
        self.history_size = config.get("state_history", 4)
        self.window_size = config.get("num_timesteps", 50) // config.get("downsample", 1)
        self.stride = config.get("stride", 1)
        self.downsample = config.get("downsample", 1)
        self.start_timestep = config.get("start_timestep", 0)

        self.add_noise = add_noise
        self.add_goal_noise = add_goal_noise
        
        # Default feature order if not provided
        self.feature_order = feature_order or [
            "delta_xy", "delta_yaw", 
            "joints", "body_z", "body_rot6d",
            "obj_rel_pos", "obj_rel_rot6d"
        ]
        
        # Load file list
        task_list_path = config.get("task_list_path", None)
        
        # Try to resolve path if provided
        if task_list_path:
             # 1. Absolute or CWD relative
             if not os.path.exists(task_list_path):
                 # 2. Relative to data_root (usually includes train_path)
                 possible_1 = os.path.join(data_root, task_list_path)
                 if os.path.exists(possible_1):
                     task_list_path = possible_1
                 else:
                     # 3. Relative to base dir_path (if separate)
                     base_dir = config.get("dir_path", "")
                     possible_2 = os.path.join(base_dir, task_list_path)
                     if os.path.exists(possible_2):
                         task_list_path = possible_2
                         
        if task_list_path and os.path.exists(task_list_path):
             print(f"Loading task list from {task_list_path}")
             with open(task_list_path, 'r') as f:
                 tasks = yaml.safe_load(f)
             
             if isinstance(tasks, dict):
                 if 'tasks' in tasks: tasks = tasks['tasks']
                 else: tasks = list(tasks.keys())
                 
             self.file_paths = []
             for t in tasks:
                 p = os.path.join(data_root, t, "best_trajectory.npz")
                 if os.path.exists(p):
                     self.file_paths.append(p)
                 else:
                     print(f"Warning: {p} not found.")
        else:
            if task_list_path:
                print(f"Warning: task_list_path '{task_list_path}' configured but not found. Falling back to glob.")
            self.file_paths = sorted(glob.glob(os.path.join(data_root, "**/*.npz"), recursive=True))

        if not self.file_paths:
            print(f"Warning: No .npz files found in {data_root}")
        
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
        Iterate through files to determine valid window start indices.
        Mimics create_dataset.py padding and windowing logic.
        """
        print("Indexing dataset...")
        for i, fpath in enumerate(self.file_paths):
            try:
                with np.load(fpath, allow_pickle=True) as data:
                     # Identify Length & Batch
                    B, T = 1, 0
                    if 'body_pos_w' in data:
                        arr = data['body_pos_w']
                        if arr.ndim == 4: # (N, T, K, 3)
                            B, T = arr.shape[0], arr.shape[1]
                        else:
                            T = arr.shape[0]
                    elif 'base_xyz_quat' in data:
                         arr = data['base_xyz_quat']
                         if arr.ndim == 3: # (N, T, D)
                             B, T = arr.shape[0], arr.shape[1]
                         else: # (T, D)
                             T = arr.shape[0]
                    else:
                        continue 
                    
                    # Apply downsampling effectively reduces T
                    T_down = T // self.downsample

                    T_padded = T_down + 2    
                    w_size = self.window_size + self.history_size
                    num_windows = T_padded - w_size + 1
                    
                    if num_windows > 0:
                        starts = np.arange(0, num_windows, self.stride)
                        for b in range(B):
                            for s in starts:
                                self.indices.append((i, b, s))
                            
            except Exception as e:
                print(f"Error indexing {fpath}: {e}")
        print(f"Indexed {len(self.indices)} windows.")

    def _load_raw_trajectory(self, fpath, batch_idx):
        """Generalized loader for different schema."""
        raw = {}
        with np.load(fpath, allow_pickle=True) as data:
            # RL Rollout Schema
            if 'body_pos_w' in data:
                # Determine batching
                is_batched = data['body_pos_w'].ndim == 4
                
                def extract(key):
                    arr = data[key]
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
                
            elif 'base_xyz_quat' in data:
                # SBTO OmniRetarget Dataset Schema
                # Check for batching
                is_batched = data['base_xyz_quat'].ndim == 3
                
                def extract(key):
                    arr = data[key]
                    if is_batched:
                        arr = arr[batch_idx]
                    return arr[::self.downsample]

                raw['base'] = extract('base_xyz_quat')
                raw['joints'] = extract('actuator_pos')
                raw['obj'] = extract('obj_0_xyz_quat')
                
                # Velocities
                if 'base_linvel_angvel' in data:
                    raw['base_vel'] = extract('base_linvel_angvel')
                if 'actuator_vel' in data:
                    raw['joints_vel'] = extract('actuator_vel')
                if 'obj_0_linvel_angvel' in data:
                    raw['obj_vel'] = extract('obj_0_linvel_angvel')

                
        return raw

    def _compute_task_params(self, full_obj_traj):
        """
        Compute task parameters from FULL object trajectory.
        obj_traj: (T_full, 7)
        """
        return full_obj_traj[-1, :2]

    def _compute_transform(self, raw_data, t_start):
        """
        Compute relative features for the window starting at t_start.
        Returns a dictionary of available features.
        """
        w_size = self.window_size + self.history_size

        # Calculate raw indices based on padded timeline (pad=1 at start)
        raw_start = t_start - 1
        raw_end = raw_start + w_size
        raw_len = raw_data['base'].shape[0]

        # Handle boundary/padding
        read_start = max(0, raw_start)
        read_end = min(raw_len, raw_end)
        
        # Extract available data
        base = raw_data['base'][read_start : read_end]      # (valid_len, 7)
        joints = raw_data['joints'][read_start : read_end]  # (valid_len, 29)
        obj = raw_data['obj'][read_start : read_end]        # (valid_len, 7)
        
        base_vel = None
        joints_vel = None
        obj_vel = None
        if 'base_vel' in raw_data:
            base_vel = raw_data['base_vel'][read_start : read_end]
        if 'joints_vel' in raw_data:
            joints_vel = raw_data['joints_vel'][read_start : read_end]
        if 'obj_vel' in raw_data:
            obj_vel = raw_data['obj_vel'][read_start : read_end]

        # 1. Pad Front (if raw_start < 0)
        pad_front = max(0, -raw_start)
        if pad_front > 0:
            base = np.pad(base, ((pad_front, 0), (0,0)), mode='edge')
            joints = np.pad(joints, ((pad_front, 0), (0,0)), mode='edge')
            obj = np.pad(obj, ((pad_front, 0), (0,0)), mode='edge')
            if base_vel is not None: base_vel = np.pad(base_vel, ((pad_front, 0), (0,0)), mode='edge')
            if joints_vel is not None: joints_vel = np.pad(joints_vel, ((pad_front, 0), (0,0)), mode='edge')
            if obj_vel is not None: obj_vel = np.pad(obj_vel, ((pad_front, 0), (0,0)), mode='edge')
        
        # 2. Pad Back (if raw_end > raw_len)
        pad_back = max(0, raw_end - raw_len)
        if pad_back > 0:
            base = np.pad(base, ((0, pad_back), (0,0)), mode='edge')
            joints = np.pad(joints, ((0, pad_back), (0,0)), mode='edge')
            obj = np.pad(obj, ((0, pad_back), (0,0)), mode='edge')
            if base_vel is not None: base_vel = np.pad(base_vel, ((0, pad_back), (0,0)), mode='edge')
            if joints_vel is not None: joints_vel = np.pad(joints_vel, ((0, pad_back), (0,0)), mode='edge')
            if obj_vel is not None: obj_vel = np.pad(obj_vel, ((0, pad_back), (0,0)), mode='edge')
            
        # Define Reference Frame.
        ref_idx = min(self.history_size - 1, w_size - 1)
        
        # --- Computations (SBTO Logic) ---
        # Use centralized logic from sbto_utils
        comps, anchors = compute_sbto_components(
            base=base[None, ...],   # (1, W, 7)
            joints=joints[None, ...], 
            obj=obj[None, ...],
            ref_idx=ref_idx,
            base_vel=base_vel[None, ...] if base_vel is not None else None,
            joints_vel=joints_vel[None, ...] if joints_vel is not None else None,
            obj_vel=obj_vel[None, ...] if obj_vel is not None else None,
        )
        
        # Unpack to (W, ...)
        feats = {k: v[0] for k, v in comps.items()}

        task_params = self._compute_task_params(obj)
        feats['task_params'] = task_params
        
        anchor = {}
        anchor['ref_pos'] = anchors['ref_pos'][0, 0]
        anchor['ref_quat'] = anchors['ref_quat'][0, 0]
        anchor['task_params'] = task_params

        return feats, anchor
    
    def _normalize(self, key, tensor):
        """Apply min-max normalization to [-1, 1]."""
        if not self.stats:
            return tensor
            
        min_k = f"min_{key}"
        max_k = f"max_{key}"
        
        if min_k in self.stats and max_k in self.stats:
            min_val = self.stats[min_k].to(tensor.device)
            max_val = self.stats[max_k].to(tensor.device)
            
            denom = max_val - min_val
            denom[denom < 1e-6] = 1.0

            return 2 * (tensor - min_val) / denom - 1
            
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
        """Iterate entire dataset (or subset) to compute min/max for each feature type."""
        print("Calculating normalization stats (min/max)...")
        mins = {}
        maxs = {}
        
        # Use a subset if dataset is huge, else full pass
        for idx in range(len(self.indices)):
            file_idx, batch_idx, t_start = self.indices[idx]
            raw_traj = self._load_raw_trajectory(self.file_paths[file_idx], batch_idx)
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
        
        # Store
        self.stats = {}
        for k in mins:
            self.stats[f"min_{k}"] = torch.as_tensor(mins[k]).float()
            self.stats[f"max_{k}"] = torch.as_tensor(maxs[k]).float()
            
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
        mins = []
        maxs = []
        for key in self.feature_order:
             # Check if key exists in stats (it should if we computed it)
             # Note: stats keys are "min_key", "max_key"
             mk, MK = f"min_{key}", f"max_{key}"
             if mk in self.stats and MK in self.stats:
                 mins.append(self.stats[mk])
                 maxs.append(self.stats[MK])
             else:
                 # Warn or handle missing
                 pass
        
        if mins:
            self.global_min = torch.cat(mins, dim=-1)
            self.global_max = torch.cat(maxs, dim=-1)

    def denormalize_global(self, tensor):
        """Denormalize a concatenated feature tensor using global stats."""
        if hasattr(self, 'global_min') and hasattr(self, 'global_max'):
            device = tensor.device
            min_val = self.global_min.to(device)
            max_val = self.global_max.to(device)
            return ((tensor + 1) / 2) * (max_val - min_val) + min_val
        return tensor

    def __len__(self):
        return len(self.indices)
