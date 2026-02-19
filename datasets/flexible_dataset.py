import numpy as np
import torch
import os
import glob
import re
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.math.sbto_utils import compute_sbto_components, yaw_to_rot_matrix, yaw_from_quat
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
        add_goal_noise=False,
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
        self.normalize_goal_vec = config.get("normalize_goal_vec", True)

        self.add_noise = add_noise
        self.add_goal_noise = add_goal_noise
        
        # Default feature order if not provided
        self.feature_order = feature_order or [
            "delta_xy", "delta_yaw",
            "obj_delta_xy",
            "joints", "body_z", "body_rot6d",
            "obj_rel_pos", "obj_rel_rot6d",
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
                 file_names = ["top_trajectories.npz", "best_trajectory_rand.npz"]
                 for file_name in file_names:
                     p = os.path.join(data_root, t, file_name)
                     if os.path.exists(p):
                         self.file_paths.append(p)
                         break
                     print(f"Warning: {p} not found.")
        else:
            if task_list_path:
                print(f"Warning: task_list_path '{task_list_path}' configured but not found. Falling back to glob.")
            self.file_paths = sorted(glob.glob(os.path.join(data_root, "**/*.npz"), recursive=True))

        if not self.file_paths:
            print(f"Warning: No .npz files found in {data_root}")
        
        # 1. Index dataset (determine B, T for all files)
        self.indices = []
        self.traj_lengths = []
        self._index_dataset()
        
        # 2. Preload processed data to RAM (Speed Optimization for small-medium datasets)
        # Replacing LRU cache with full load (~500MB for 4k trajectories)
        self.ram_cache = []
        self._preload_dataset()

        # Normalization
        self.norm_path = norm_path
        self.stats = {}
        if calculate_stats:
            self._calculate_stats()
        elif norm_path and os.path.exists(norm_path):
            self._load_stats()

    def _preload_dataset(self):
        print(f"Preloading {len(self.file_paths)} files into RAM...")
        for fpath in tqdm(self.file_paths, desc="Loading Data"):
            self.ram_cache.append(self._load_and_process_file(fpath))

    def _load_and_process_file(self, fpath):
        """
        Loads file and returns dict of (N, T, D) arrays with downsampling applied.
        """
        raw = {}
        processed = {}
        try:
            with np.load(fpath, allow_pickle=True) as data:
                # RL Rollout Schema
                if 'body_pos_w' in data:
                    is_batched = data['body_pos_w'].ndim == 4
                    # Helper to get (N, T, D)
                    def extract(key):
                        arr = data[key] # (N, T, ...) or (T, ...)
                        if not is_batched:
                            arr = arr[None, ...] # Make (1, T, ...)
                        return arr[:, ::self.downsample]
                    
                    # Base (Merge pos+quat)
                    body_pos = extract('body_pos_w') 
                    body_quat = extract('body_quat_w')
                    # Expect (N, T, K, 3). Take K=0
                    # If (N, T, D), then just take it.
                    if body_pos.ndim == 4:
                         processed['base'] = np.concatenate([body_pos[:, :, 0, :], body_quat[:, :, 0, :]], axis=-1)
                    else:
                         # Assume already (N, T, 3) 
                         processed['base'] = np.concatenate([body_pos, body_quat], axis=-1)

                    processed['joints'] = extract('joint_pos')
                    processed['obj'] = np.concatenate([extract('object_pos_w'), extract('object_quat_w')], axis=-1)
                    
                elif 'base_xyz_quat' in data:
                    # SBTO Schema
                    is_batched = data['base_xyz_quat'].ndim == 3
                     # Helper
                    def extract(key):
                        arr = data[key]
                        if not is_batched:
                            arr = arr[None, ...]
                        return arr[:, ::self.downsample]

                    processed['base'] = extract('base_xyz_quat')
                    processed['joints'] = extract('actuator_pos')
                    processed['obj'] = extract('obj_0_xyz_quat')
                    
                    if 'base_linvel_angvel' in data:
                        processed['base_vel'] = extract('base_linvel_angvel')
                    if 'actuator_vel' in data:
                        processed['joints_vel'] = extract('actuator_vel')
                    if 'obj_0_linvel_angvel' in data:
                        processed['obj_vel'] = extract('obj_0_linvel_angvel')
                
                # Metadata
                if 'fps' in data:
                    fps_val = data['fps'].item() if data['fps'].ndim == 0 else data['fps'][0]
                    B_dim = processed['base'].shape[0]
                    processed['fps'] = np.repeat(fps_val, B_dim)
                    
        except Exception as e:
            print(f"Failed to load {fpath}: {e}")
            # Return dummy or handle error? indices should prevent access if validation worked.
            # But indices() only reads header. 
            pass 
            
        return processed

    def _get_single_traj(self, file_idx, batch_idx):
        """Get (T, D) dict for a specific trajectory."""
        file_data = self.ram_cache[file_idx]
        return {k: v[batch_idx] for k, v in file_data.items()}
            
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
                    T_padded = T_down + 1
                    self.traj_lengths.append(T_down)
                    
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

    def _compute_transform(self, raw_traj, t_start):
        # raw_traj contains full trajectory (downsampled)
        # t_start is the index of the start of the window
        
        T_traj = raw_traj['base'].shape[0]
        w_size = self.window_size + self.history_size
        raw_start = t_start - 1
        raw_end = raw_start + w_size
        
        # Read with padding
        read_start = max(0, raw_start)
        read_end = min(T_traj, raw_end)
        
        # Slice
        b_slice = raw_traj['base'][read_start:read_end]
        j_slice = raw_traj['joints'][read_start:read_end]
        o_slice = raw_traj['obj'][read_start:read_end]
        
        # # Pad
        pad_front = max(0, -raw_start)
        pad_back = max(0, raw_end - T_traj)
        
        if pad_front > 0 or pad_back > 0:
            p = ((pad_front, pad_back), (0,0))
            b_slice = np.pad(b_slice, p, mode='edge')
            j_slice = np.pad(j_slice, p, mode='edge')
            o_slice = np.pad(o_slice, p, mode='edge')
            
        # Velocity
        if 'base_vel' in raw_traj:
            bv = raw_traj['base_vel'][read_start:read_end]
            jv = raw_traj['joints_vel'][read_start:read_end]
            ov = raw_traj['obj_vel'][read_start:read_end]
            if pad_front > 0 or pad_back > 0:
                p = ((pad_front, pad_back), (0,0))
                bv = np.pad(bv, p, mode='edge')
                jv = np.pad(jv, p, mode='edge')
                ov = np.pad(ov, p, mode='edge')
        else:
            bv, jv, ov = None, None, None
            
        # Reference frame for SBTO
        # "Current state" is usually history_size.
        ref_idx = min(self.history_size - 1, w_size - 1)
        
        # Batched SBTO call (Expects B, T, D)
        b_in = b_slice[np.newaxis, ...]
        j_in = j_slice[np.newaxis, ...]
        o_in = o_slice[np.newaxis, ...]
        
        bv_in = bv[np.newaxis, ...] if bv is not None else None
        jv_in = jv[np.newaxis, ...] if jv is not None else None
        ov_in = ov[np.newaxis, ...] if ov is not None else None

        features_b, anchor_b = compute_sbto_components(
            b_in, j_in, o_in, ref_idx,
            base_vel=bv_in, joints_vel=jv_in, obj_vel=ov_in
        )
        
        # Unwrap batch dim
        features = {k: np.array(v[0]) for k, v in features_b.items() if v is not None}
        
        # Helper to squeeze anchor dims
        def _sq(x):
            if x.shape[0] == 1: return x.squeeze(0)
            return x

        anchor = {k: np.array(_sq(v[0])) for k, v in anchor_b.items()}
        anchor["final_obj_pos"] = raw_traj['obj'][-1, :3]

        # Add task params
        features["task_params"] = self._compute_task_params(raw_traj['base'], raw_traj['obj'], start_idx=read_start)
        
        # Add metadata to anchor
        if 'fps' in raw_traj:
            anchor['fps'] = np.array(raw_traj['fps'])

        return features, anchor

    def _compute_task_params(self, base_traj, obj_traj, start_idx=0):
        # Default: Object X, Y displacement from start to end of trajectory
        if obj_traj.shape[0] > 0:
            disp_vector_global = (obj_traj[-1, :3] - obj_traj[start_idx, :3])
            R_ref_inv = yaw_to_rot_matrix(-yaw_from_quat(base_traj[start_idx, 3:]))
            disp_vector = (R_ref_inv @ disp_vector_global)[:2] # (2,)
            if self.normalize_goal_vec:
                disp_vector_norm = np.linalg.norm(disp_vector)
                if disp_vector_norm > 1e-6:
                    disp_vector = disp_vector / disp_vector_norm
            return disp_vector
        return np.zeros(2)

    def _normalize(self, key, val):
        min_k = self.stats.get(f"min_{key}")
        max_k = self.stats.get(f"max_{key}")
        
        if min_k is None or max_k is None:
            return val
        
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val).float()
            
        dev = val.device
        min_v = min_k.to(dev)
        max_v = max_k.to(dev)
        
        # simple min-max to [-1, 1]
        diff = max_v - min_v
        # Prevent div zero
        mask = diff < 1e-6
        if mask.any():
            diff = diff.clone()
            diff[mask] = 1.0
            
        norm = (val - min_v) / diff
        norm = norm * 2 - 1
        return norm
    
    def _denormalize(self, key, val):
        min_k = self.stats.get(f"min_{key}")
        max_k = self.stats.get(f"max_{key}")
        
        if min_k is None or max_k is None:
            return val
            
        dev = val.device
        min_v = min_k.to(dev)
        max_v = max_k.to(dev)

        denorm = (val + 1) / 2 * (max_v - min_v) + min_v
        return denorm

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
        
        # Use RAM-cached data
        raw_traj = self._get_single_traj(file_idx, batch_idx)
        
        # Compute features
        features, anchor = self._compute_transform(raw_traj, t_start)

        # Assemble Windowed Feature Vector
        window_parts = []
        for key in self.feature_order:
            if key in features:                
                part = torch.from_numpy(features[key]).float()
                part = self._normalize(key, part)
                if self.add_noise:
                    history_slice = part[:self.history_size] 
                    part[:self.history_size] = self._add_obs_noise(history_slice, key)
                window_parts.append(part)
            else:
                raise ValueError(f"Feature {key} needed but not computed.")
                
        window_tensor = torch.cat(window_parts, dim=-1) # (W, Total_Dim)
        
        # Current State 
        current_state = window_tensor[:self.history_size, (self.num_features-self.num_observations):].clone()

        # Future trajectory
        future_states = window_tensor[self.history_size:, :].clone()

        task_params = torch.from_numpy(features["task_params"]).float()
        task_params = self._normalize("task_params", task_params)
        if self.add_goal_noise:
            task_params = self._add_obs_noise(task_params, "task_params")

        return future_states, current_state, task_params, anchor


    def _calculate_stats(self):
        """
        Optimized stats calculation:
        1. Groups windows by file to minimize IO (load once per file).
        2. Vectorizes operations where possible in large chunks.
        """
        print("Calculating normalization stats (min/max)...")
        mins = {}
        maxs = {}
        
        # Group indices by file/batch to avoid repeated IO
        file_map = {}
        for idx in range(len(self.indices)):
            file_idx, batch_idx, t_start = self.indices[idx]
            key = (file_idx, batch_idx)
            if key not in file_map:
                file_map[key] = []
            file_map[key].append(t_start)
            
        # Process each file once
        for (file_idx, batch_idx), starts in tqdm(file_map.items(), desc="Computing stats"):
            raw_traj = self._get_single_traj(file_idx, batch_idx)
                        
            # 2. Process Windows in Batches
            BATCH_SIZE = 256 # Process windows in chunks to manage memory
            
            w_size = self.window_size + self.history_size
            ref_idx = min(self.history_size, w_size - 1)
            raw_len = raw_traj['base'].shape[0]

            # Pre-allocate generic buffer lists
            batch_data = {
                "base": [], "joints": [], "obj": [],
                "base_vel": [], "joints_vel": [], "obj_vel": []
            }
            batch_task_params = []
            
            has_vel = 'base_vel' in raw_traj
            
            for s in starts:
                raw_start = s - 1
                raw_end = raw_start + w_size
                
                # Logic mimicking _compute_transform padding
                read_start = max(0, raw_start)
                read_end = min(raw_len, raw_end)
                
                # Slicing
                b_slice = raw_traj['base'][read_start:read_end]
                j_slice = raw_traj['joints'][read_start:read_end]
                o_slice = raw_traj['obj'][read_start:read_end]

                # Task Params
                if 'task_params' in raw_traj:
                    tp = raw_traj['task_params']
                else:
                    tp = self._compute_task_params(raw_traj['base'], raw_traj['obj'], start_idx=read_start)
                batch_task_params.append(tp)
                
                pad_front = max(0, -raw_start)
                pad_back = max(0, raw_end - raw_len)
                
                if pad_front > 0 or pad_back > 0:
                    pad_width = ((pad_front, pad_back), (0,0))
                    b_slice = np.pad(b_slice, pad_width, mode='edge')
                    j_slice = np.pad(j_slice, pad_width, mode='edge')
                    o_slice = np.pad(o_slice, pad_width, mode='edge')
                
                batch_data["base"].append(b_slice)
                batch_data["joints"].append(j_slice)
                batch_data["obj"].append(o_slice)
                
                if has_vel:
                     bv = raw_traj['base_vel'][read_start:read_end] if 'base_vel' in raw_traj else np.zeros_like(b_slice) 
                     jv = raw_traj['joints_vel'][read_start:read_end] if 'joints_vel' in raw_traj else np.zeros_like(j_slice)
                     ov = raw_traj['obj_vel'][read_start:read_end] if 'obj_vel' in raw_traj else np.zeros_like(o_slice) 
                     
                     if pad_front > 0 or pad_back > 0:
                         pad_width = ((pad_front, pad_back), (0,0))
                         bv = np.pad(bv, pad_width, mode='edge')
                         jv = np.pad(jv, pad_width, mode='edge')
                         ov = np.pad(ov, pad_width, mode='edge')
                     batch_data["base_vel"].append(bv)
                     batch_data["joints_vel"].append(jv)
                     batch_data["obj_vel"].append(ov)

                if len(batch_data["base"]) >= BATCH_SIZE:
                    self._update_batch_stats(mins, maxs, batch_data, ref_idx, has_vel, task_params=batch_task_params)
                    batch_data = {k: [] for k in batch_data}
                    batch_task_params = []
            
            if len(batch_data["base"]) > 0:
                 self._update_batch_stats(mins, maxs, batch_data, ref_idx, has_vel, task_params=batch_task_params)
                 batch_task_params = []

        # Store to stats dict
        self.stats = {}
        
        for k in mins:
            self.stats[f"min_{k}"] = torch.as_tensor(mins[k]).float()
            self.stats[f"max_{k}"] = torch.as_tensor(maxs[k]).float()

        # Shared Normalization Logic
        shared_keys = ["delta_xy", "obj_delta_xy", "task_params"]
        present_shared = [k for k in shared_keys if f"min_{k}" in self.stats]
        
        if len(present_shared) > 1:
            # Collect XY mins/maxs
            xy_mins = []
            xy_maxs = []
            
            for k in present_shared:
                v_min = self.stats[f"min_{k}"]
                v_max = self.stats[f"max_{k}"]
                
                # Take first 2 dims (assuming X, Y are 0, 1)
                xy_mins.append(v_min[:2])
                xy_maxs.append(v_max[:2])
            
            # Compute global min/max for XY
            global_min_xy = torch.min(torch.stack(xy_mins), dim=0)[0]
            global_max_xy = torch.max(torch.stack(xy_maxs), dim=0)[0]
                        
            # Update stats in place
            for k in present_shared:
                # Update Min
                curr_min = self.stats[f"min_{k}"].clone()
                curr_min[:2] = global_min_xy
                self.stats[f"min_{k}"] = curr_min
                
                # Update Max
                curr_max = self.stats[f"max_{k}"].clone()
                curr_max[:2] = global_max_xy
                self.stats[f"max_{k}"] = curr_max
                
        # Save
        if self.norm_path:
            os.makedirs(os.path.dirname(self.norm_path), exist_ok=True)
            # Save as numpy dict
            np_stats = {k: v.numpy() for k, v in self.stats.items()}
            np.savez(self.norm_path, **np_stats)
            print(f"Stats saved to {self.norm_path}")
            
        self._update_global_stats()

    def _update_batch_stats(self, mins, maxs, batch_data, ref_idx, has_vel, task_params=None):
        base = np.stack(batch_data["base"]) # (B, W, 7)
        joints = np.stack(batch_data["joints"])
        obj = np.stack(batch_data["obj"])
        
        base_vel = np.stack(batch_data["base_vel"]) if has_vel else None
        joints_vel = np.stack(batch_data["joints_vel"]) if has_vel else None
        obj_vel = np.stack(batch_data["obj_vel"]) if has_vel else None
        
        comps, _ = compute_sbto_components(
            base, joints, obj, ref_idx,
            base_vel=base_vel, joints_vel=joints_vel, obj_vel=obj_vel
        )

        if task_params:
            # list of (D,) -> (B, D) -> (B, 1, D) so it matches (B, W, D) logic
            tp_stack = np.stack(task_params) # (B, D)
            if tp_stack.ndim == 2:
                tp_stack = tp_stack[:, np.newaxis, :]
            comps['task_params'] = tp_stack
        
        for k, v in comps.items():
            if v is None: continue
            v = v.astype(np.float64)
            # v is (B, W, D). Min/Max over B(0) and W(1)
            v_min = np.min(v, axis=(0, 1))
            v_max = np.max(v, axis=(0, 1))
            
            if k not in mins:
                mins[k] = v_min
                maxs[k] = v_max
            else:
                mins[k] = np.minimum(mins[k], v_min)
                maxs[k] = np.maximum(maxs[k], v_max)

    def _load_stats(self):
        print(f"Loading stats from {self.norm_path}")
        with np.load(self.norm_path, allow_pickle=True) as data:
            for k in data.keys():
                self.stats[k] = torch.from_numpy(data[k]).float()
        
        self._update_global_stats()
        
    def _update_global_stats(self):
        # Precompute global stats for concatenated features
        mins = []
        maxs = []
        for key in self.feature_order:
             # Check if key exists in stats
             mk, MK = f"min_{key}", f"max_{key}"
             if mk in self.stats and MK in self.stats:
                 mins.append(self.stats[mk])
                 maxs.append(self.stats[MK])
        
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
