import numpy as np
import torch
import os
import glob
import re
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.math.sbto_utils import compute_sbto_components, yaw_to_rot_matrix, yaw_from_quat, rot6d_to_rot, quat_to_rot, compute_task_params, rot_to_6d
from utils.math.rotation_conversions import ( 
    axis_angle_to_quaternion, 
)

class FlexibleWindowDataset(Dataset):
    """
    Dataset that loads raw absolute trajectories, windows them, 
    transforms to relative coordinates on-the-fly, 
    and assembles features based on a configurable order.
    """
    def __init__(self, 
        data_buffer, 
        config, 
        task_params=None,
        norm_path=None, 
        calculate_stats=False,
        training_cfg=None,  
    ):
        
        self.noise_cfg = training_cfg.get("state_conditioning_noise_level", {})
        self.num_observations = config.get("num_observations", 45)
        self.num_features = config.get("num_features", 48)
        self.history_size = config.get("state_history", 4)
        self.window_size = config.get("num_timesteps", 50) // config.get("downsample", 1)
        self.stride = config.get("stride", 1)
        self.downsample = config.get("downsample", 1)
        self.normalize_goal_vec = config.get("normalize_goal_vec", True)
        self.num_task_params = config.get("num_task_params", 2)
        self.normalization_type = config.get("normalization_type", "mean_std")  # 'mean_std' or 'min_max'
        self.feature_order = config.get("feature_order", 
                [
                    "delta_xy", "delta_yaw", "obj_delta_xy",
                    "joints", "body_z", "body_rot6d", 
                    "obj_rel_pos", "obj_rel_rot6d",
                ]
        )

        self.key_mapping = config.get("key_mapping", {})
        
        print(f"Using feature order: {self.feature_order}")
        print(f"Using normalization type: {self.normalization_type}")
        print(f"Using key mapping: {self.key_mapping}")

        self.add_noise = training_cfg.get("add_obs_noise", False)
        self.add_goal_noise = training_cfg.get("add_goal_noise", False)

        self._raw_buffer = data_buffer
        self._task_params = task_params
        
        # ---- Preload buffer into ram_cache ----
        self.ram_cache = []
        self.traj_lengths = []
        self.indices = []

        self._preload_from_buffer()
        self._index_dataset()

        # ---- Normalization ----
        self.norm_path = norm_path
        self.stats = {}
        if calculate_stats:
            self._calculate_stats()
        elif norm_path and os.path.exists(norm_path):
            self._load_stats()


    def _standardize_buffer(self):
        """Normalize buffer into {key: list_of_(T,D)_arrays}."""
        buf = self._raw_buffer

        if isinstance(buf, list):
            if len(buf) == 0:
                return {}
            keys = buf[0].keys()
            return {k: [item.get(k) for item in buf] for k in keys}

        if isinstance(buf, dict):
            standardized = {}
            # Determine expected batch count from a temporal key
            num_trajs = None
            if 'joints' in buf.keys():
                v = buf['joints']
                if isinstance(v, (list, tuple)):
                    num_trajs = len(v)
                elif isinstance(v, np.ndarray) and v.ndim >= 3:
                    num_trajs = v.shape[0]

            for key, val in buf.items():
                if isinstance(val, (list, tuple)):
                    standardized[key] = list(val)
                elif isinstance(val, np.ndarray):
                    # Split along batch dim if first dim matches num_trajs
                    if num_trajs is not None and val.shape[0] == num_trajs and val.ndim >= 2:
                        standardized[key] = [val[i] for i in range(val.shape[0])]
                    elif val.ndim >= 3:
                        standardized[key] = [val[i] for i in range(val.shape[0])]
                    else:
                        standardized[key] = [val]
                elif isinstance(val, torch.Tensor):
                    val_np = val.cpu().numpy()
                    if num_trajs is not None and val_np.shape[0] == num_trajs and val_np.ndim >= 2:
                        standardized[key] = [val_np[i] for i in range(val_np.shape[0])]
                    elif val_np.ndim >= 3:
                        standardized[key] = [val_np[i] for i in range(val_np.shape[0])]
                    else:
                        standardized[key] = [val_np]
                else:
                    standardized[key] = [val]
            return standardized

        raise ValueError("BufferDataset expects a dict or list of dicts.")
    
    def get_k(self, key):
        """Helper to get mapped key from config."""
        return self.key_mapping.get(key, key)

    def _preload_from_buffer(self):
        """
        Process _raw_buffer (list of dicts from load_dataset.py) into ram_cache.
        Each dict has keys like 'base', 'joints', 'obj' with shapes (N_i, T_i, D).
        Each dict becomes one entry in ram_cache.

        If _task_params is provided, it should be a flat list/array of length
        sum(N_i), one per trajectory across all files. goal_obj_world is computed
        from task_params and stored as (N_i, 3) in each file dict.
        """
        if not self._raw_buffer:
            return

        total_trajs = 0
        tp_offset = 0  # running offset into the flat task_params list

        for file_data in self._raw_buffer:
            if not file_data:
                continue

            # Ensure batch dimension exists on temporal arrays
            for k in list(file_data.keys()):
                v = file_data[k]
                if isinstance(v, np.ndarray) and k in (
                    'base', 'joints', 'obj',
                    'base_vel', 'joints_vel', 'obj_vel', 'ee_rel_pos',
                ):
                    if v.ndim == 2:  # (T, D) -> (1, T, D)
                        file_data[k] = v[None]

            N_i = file_data['base'].shape[0]

            # Compute goal_obj_world from task_params if provided
            if self._task_params is not None:
                goal_world = np.zeros((N_i, 3), dtype=np.float64)
                for j in range(N_i):
                    goal_local = np.asarray(self._task_params[tp_offset + j], dtype=np.float64)

                    # Initial pelvis pose & initial object pos (first frame)
                    init_base = file_data['base'][j, 0, :]   # (7,) [x,y,z,qw,qx,qy,qz]
                    init_obj  = file_data['obj'][j, 0, :3]    # (3,)

                    init_yaw = yaw_from_quat(init_base[3:7])
                    R_local_to_world = yaw_to_rot_matrix(init_yaw)

                    goal_3d = np.zeros(3, dtype=np.float64)
                    n = min(len(goal_local), 3)
                    goal_3d[:n] = goal_local[:n]

                    goal_world[j] = (R_local_to_world @ goal_3d[:, None])[:, 0] + init_obj

                file_data['goal_obj_world'] = goal_world  # (N_i, 3)
                tp_offset += N_i

            self.ram_cache.append(file_data)
            total_trajs += N_i

        print(f"Loaded {len(self.ram_cache)} files ({total_trajs} total trajectories) into RAM cache.")

    # Keys that are non-temporal (no time dimension to pad)
    _NON_TEMPORAL_KEYS = {'goal_obj_world', 'fps'}

    def _get_single_traj(self, file_idx, batch_idx):
        """Get padded (T, D) dict for a specific trajectory.
        Non-temporal keys (goal_obj_world, fps) are passed through without padding.
        """
        file_data = self.ram_cache[file_idx]

        pad_start = self.history_size
        pad_end = self.traj_lengths[file_idx] - (file_data["base"][batch_idx].shape[0] + pad_start)

        pad_kwargs = lambda arr: np.pad(
            arr,
            ((pad_start, pad_end), (0, 0)) if arr.ndim == 2 else ((0, 0),),
            mode='edge',
        )

        result = {}
        for k, v in file_data.items():
            if k in self._NON_TEMPORAL_KEYS:
                # Non-temporal — just index batch dim, no padding
                result[k] = v[batch_idx] if hasattr(v, '__getitem__') and np.asarray(v).ndim >= 1 else v
            else:
                result[k] = pad_kwargs(v[batch_idx])
        return result
        

    def _index_dataset(self):
        """
        Iterate through files to determine valid window start indices.
        Mimics create_dataset.py padding and windowing logic.
        """
        print("Indexing dataset...")
        for i in range(len(self.ram_cache)):
            try:
                data = self.ram_cache[i] if i < len(self.ram_cache) else self._load_and_process_file(self.file_paths[i])
                
                # Identify Length & Batch
                B, T = 1, 0
                if 'joints' in data:
                    arr = data['joints']
                    if arr.ndim == 3: # (N, T, 29)
                        B, T = arr.shape[0], arr.shape[1]
                    else:
                        T = arr.shape[0]
                else:
                    print("Warning: 'joints' key not found in data. Skipping indexing for this file.")
                    continue 
                
                # Apply downsampling effectively reduces T
                T_down = T // self.downsample

                pad_start = self.history_size
                pad_end = self.window_size - (T_down + pad_start) % self.window_size + self.history_size
            
                T_padded = T_down + pad_start + pad_end
                self.traj_lengths.append(T_padded)
                
                w_size = self.window_size
                max_start = T_padded - w_size
                
                if max_start > 0:
                    starts = np.arange(0, max_start + 1, self.stride)
                    for b in range(B):
                        for s in starts:
                            self.indices.append((i, b, s))

            except Exception as e:
                print(f"Error indexing {self.file_paths[i]}: {e}")
        print(f"Indexed {len(self.indices)} windows.")

    def _compute_transform(self, raw_traj, t_start):
        # raw_traj contains full trajectory (downsampled)
        # t_start is the index of the start of the window
        
        T_traj = raw_traj['base'].shape[0]
        w_size = self.window_size
        raw_end = t_start + w_size
        
        # Read with padding
        read_start = max(0, t_start)
        read_end = min(T_traj, raw_end)
        
        # Slice
        b_slice = raw_traj['base'][read_start:read_end]
        j_slice = raw_traj['joints'][read_start:read_end]
        o_slice = raw_traj['obj'][read_start:read_end]
        # Velocity
        if 'base_vel' in raw_traj:
            bv = raw_traj['base_vel'][read_start:read_end]
            jv = raw_traj['joints_vel'][read_start:read_end]
            ov = raw_traj['obj_vel'][read_start:read_end]
        else:
            bv, jv, ov = None, None, None
            
        # Reference frame for SBTO
        # "Current state" is usually history_size.
        ref_idx = self.history_size - 1
        
        # Batched SBTO call (Expects B, T, D)
        b_in = b_slice[np.newaxis, ...]
        j_in = j_slice[np.newaxis, ...]
        o_in = o_slice[np.newaxis, ...]
        
        bv_in = bv[np.newaxis, ...] if bv is not None else None
        jv_in = jv[np.newaxis, ...] if jv is not None else None
        ov_in = ov[np.newaxis, ...] if ov is not None else None

        # Pre-computed ee_rel_pos
        ee_in = None
        if 'ee_rel_pos' in raw_traj:
            ee_slice = raw_traj['ee_rel_pos'][read_start:read_end]
            ee_in = ee_slice[np.newaxis, ...]

        features_b, anchor_b = compute_sbto_components(
            b_in, j_in, o_in, ref_idx,
            base_vel=bv_in, joints_vel=jv_in, obj_vel=ov_in,
            ee_rel_pos=ee_in
        )
        
        # Unwrap batch dim
        features = {k: np.array(v[0]) for k, v in features_b.items() if v is not None}
        
        # Helper to squeeze anchor dims
        def _sq(x):
            if x.shape[0] == 1: return x.squeeze(0)
            return x

        anchor = {k: np.array(_sq(v[0])) for k, v in anchor_b.items()}
        # Use external goal if available, otherwise fall back to trajectory's final obj pos
        if 'goal_obj_world' in raw_traj:
            goal_world = raw_traj['goal_obj_world']  # (3,)
            anchor["final_obj_pos"] = goal_world
            features["task_params"] = self._compute_task_params(
                current_robot_state=raw_traj['base'][read_start, 3:],
                current_obj_state=raw_traj['obj'][read_start, :3],
                desired_obj_pos=goal_world,
            )
        else:
            anchor["final_obj_pos"] = raw_traj['obj'][-1, :3]
            features["task_params"] = self._compute_task_params(raw_traj['base'], raw_traj['obj'], start_idx=read_start)
        
        # Add metadata to anchor
        if 'fps' in raw_traj:
            anchor['fps'] = np.array(raw_traj['fps'])

        return features, anchor

    def _compute_task_params(self, base_traj, obj_traj, start_idx=0):
        # Default: Object X, Y displacement from start to end of trajectory
        return compute_task_params(
            current_robot_state=base_traj[start_idx, 3:],
            current_obj_state=obj_traj[start_idx, :3],
            desired_obj_pos=obj_traj[-1, :3],
            normalize_goal_vec=self.normalize_goal_vec,
            num_task_params=self.num_task_params,
        )

    def _normalize(self, key, val):
        if self.normalization_type == "min_max":
            min_k = self.stats.get(f"min_{key}")
            max_k = self.stats.get(f"max_{key}")
            
            if min_k is None or max_k is None:
                return val
            
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val).float()
                
            dev = val.device
            min_v = min_k.to(dev)
            max_v = max_k.to(dev)
            
            # (val - min) / (max - min). Range [0, 1]
            denom = max_v - min_v
            denom[denom < 1e-6] = 1.0 
            norm = (val - min_v) / denom
            return norm
        else:
            mean_k = self.stats.get(f"mean_{key}")
            std_k = self.stats.get(f"std_{key}")
            
            if mean_k is None or std_k is None:
                return val
            
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val).float()
                
            dev = val.device
            mean_v = mean_k.to(dev)
            std_v = std_k.to(dev)
            
            norm = (val - mean_v) / std_v
            return norm
    
    def _denormalize(self, key, val):
        if self.normalization_type == "min_max":
            min_k = self.stats.get(f"min_{key}")
            max_k = self.stats.get(f"max_{key}")
            
            if min_k is None or max_k is None:
                return val
                
            dev = val.device
            min_v = min_k.to(dev)
            max_v = max_k.to(dev)

            # val * (max - min) + min
            denom = max_v - min_v
            denorm = val * denom + min_v
            return denorm
        else:
            mean_k = self.stats.get(f"mean_{key}")
            std_k = self.stats.get(f"std_{key}")
            
            if mean_k is None or std_k is None:
                return val
                
            dev = val.device
            mean_v = mean_k.to(dev)
            std_v = std_k.to(dev)

            denorm = val * std_v + mean_v
            return denorm

    def _add_rotation_noise(self, tensor, noise_level):
        """Add noise to rotation features (e.g., 6D)."""
        # tensor: (T, 6)
        
        # 1. Convert to Rotation Matrix (T, 3, 3)
        rot_mat = rot6d_to_rot(tensor.cpu().numpy())
        
        # 2. Generate random axis-angle noise
        # axis_angle = randn * noise_level represents a random rotation vector
        perturbation_axis_angle = torch.randn_like(tensor[:, :3]) * noise_level
        
        # 3. Convert perturbation to Matrix
        perturbation_quat = axis_angle_to_quaternion(perturbation_axis_angle)
        perturbation_mat = quat_to_rot(perturbation_quat.cpu().numpy())
        
        # 4. Apply perturbation: R_new = R * R_perturbation (Local perturbation)
        rot_mat_noisy = perturbation_mat @ rot_mat
        
        # 5. Convert back to 6D
        tensor_noisy = torch.from_numpy(rot_to_6d(rot_mat_noisy)).to(tensor.device)
        
        return tensor_noisy
    
    def _add_obs_noise(self, tensor, key):
        """Add noise based on config."""
        # if key in ["obj_rel_rot6d", "body_rot6d"]:
        #     return self._add_rotation_noise(tensor, noise_level=self.noise_cfg[key])
        if key in self.noise_cfg:
            noise_level = self.noise_cfg[key]
            noise = torch.randn_like(tensor) * noise_level
            tensor = tensor + noise
        if key == "task_params" and self.noise_cfg.get("task_magnitude", 0) > 0:
            tensor[..., 2] *= torch.rand_like(tensor[..., 2]) * self.noise_cfg["task_magnitude"] # Large noise for goal distance to encourage generalization
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
        future_states = window_tensor.clone()

        task_params = torch.from_numpy(features["task_params"]).float()
        task_params = self._normalize("task_params", task_params)
        if self.add_goal_noise:
            task_params = self._add_obs_noise(task_params, "task_params")

        return future_states, current_state, task_params, anchor


    def _calculate_stats(self):
        """
        Optimized stats calculation for Mean-Std:
        1. Groups windows by file to minimize IO.
        2. Accumulates sum, sum_sq, and count for each feature.
        """
        print("Calculating normalization stats (mean/std)...")
        accumulators = {}
        
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
            BATCH_SIZE = 256
            
            w_size = self.window_size
            ref_idx = self.history_size - 1
            raw_len = raw_traj['base'].shape[0]

            batch_data = {
                "base": [], "joints": [], "obj": [],
                "base_vel": [], "joints_vel": [], "obj_vel": [],
                "ee_rel_pos": [],
            }
            batch_task_params = []
            
            has_vel = 'base_vel' in raw_traj
            has_ee = 'ee_rel_pos' in raw_traj
            
            for s in starts:
                raw_start = s
                raw_end = raw_start + w_size
                
                # Logic mimicking _compute_transform padding
                read_start = max(0, raw_start)
                read_end = min(raw_len, raw_end)
                
                # Slicing
                b_slice = raw_traj['base'][read_start:read_end]
                j_slice = raw_traj['joints'][read_start:read_end]
                o_slice = raw_traj['obj'][read_start:read_end]

                if 'task_params' in raw_traj:
                    tp = raw_traj['task_params']
                else:
                    tp = self._compute_task_params(raw_traj['base'], raw_traj['obj'], start_idx=read_start)
                batch_task_params.append(tp)
                batch_data["base"].append(b_slice)
                batch_data["joints"].append(j_slice)
                batch_data["obj"].append(o_slice)
                
                if has_vel:
                     bv = raw_traj['base_vel'][read_start:read_end] if 'base_vel' in raw_traj else np.zeros_like(b_slice) 
                     jv = raw_traj['joints_vel'][read_start:read_end] if 'joints_vel' in raw_traj else np.zeros_like(j_slice)
                     ov = raw_traj['obj_vel'][read_start:read_end] if 'obj_vel' in raw_traj else np.zeros_like(o_slice) 
                     
                     batch_data["base_vel"].append(bv)
                     batch_data["joints_vel"].append(jv)
                     batch_data["obj_vel"].append(ov)
                
                if has_ee:
                    ee = raw_traj['ee_rel_pos'][read_start:read_end]
                    batch_data["ee_rel_pos"].append(ee)

                if len(batch_data["base"]) >= BATCH_SIZE:
                    self._update_batch_stats(accumulators, batch_data, ref_idx, has_vel, task_params=batch_task_params)
                    batch_data = {
                        "base": [], "joints": [], "obj": [],
                        "base_vel": [], "joints_vel": [], "obj_vel": [],
                        "ee_rel_pos": [],
                    }
                    batch_task_params = []
            
            if len(batch_data["base"]) > 0:
                 self._update_batch_stats(accumulators, batch_data, ref_idx, has_vel, task_params=batch_task_params)
                 batch_task_params = []

        # Calculate Mean and Std
        self.stats = {}
        for k, acc in accumulators.items():
            if self.normalization_type == "min_max":
                min_v = acc['min']
                max_v = acc['max']

                # Avoid zero range
                diff = max_v - min_v
                diff[diff < 1e-6] = 1.0

                self.stats[f"min_{k}"] = torch.as_tensor(min_v).float()
                self.stats[f"max_{k}"] = torch.as_tensor(max_v).float()
                # Store diff for convenience if desired, but we can compute dynamically
        
            else:
                count = acc['count']
                mean = acc['sum'] / count
                
                # Variance = E[x^2] - (E[x])^2
                # Use max(0, var) to avoid numerical error -> sqrt
                var = (acc['sq_sum'] / count) - (mean ** 2)
                std = np.sqrt(np.maximum(var, 1e-8))
                
                # Avoid divide by zero
                std[std < 1e-6] = 1.0

                self.stats[f"mean_{k}"] = torch.as_tensor(mean).float()
                self.stats[f"std_{k}"] = torch.as_tensor(std).float()

        # Shared Normalization Logic if needed (e.g. sharing delta_xy mean/std)
        # Note: Usually we want separate means but maybe shared std? 
        # For now, let's keep separate unless requested.
        
        # Save
        if self.norm_path:
            os.makedirs(os.path.dirname(self.norm_path), exist_ok=True)
            np_stats = {k: v.numpy() for k, v in self.stats.items()}
            np.savez(self.norm_path, **np_stats)
            print(f"Stats saved to {self.norm_path}")
            
        self._update_global_stats()

    def _update_batch_stats(self, accumulators, batch_data, ref_idx, has_vel, task_params=None):
        base = np.stack(batch_data["base"]) 
        joints = np.stack(batch_data["joints"])
        obj = np.stack(batch_data["obj"])
        
        base_vel = np.stack(batch_data["base_vel"]) if has_vel else None
        joints_vel = np.stack(batch_data["joints_vel"]) if has_vel else None
        obj_vel = np.stack(batch_data["obj_vel"]) if has_vel else None
        
        ee_rel_pos = None
        if "ee_rel_pos" in batch_data and len(batch_data["ee_rel_pos"]) > 0:
             ee_rel_pos = np.stack(batch_data["ee_rel_pos"])

        comps, _ = compute_sbto_components(
            base, joints, obj, ref_idx,
            base_vel=base_vel, joints_vel=joints_vel, obj_vel=obj_vel,
            ee_rel_pos=ee_rel_pos
        )

        if task_params:
            tp_stack = np.stack(task_params)
            if tp_stack.ndim == 2:
                tp_stack = tp_stack[:, np.newaxis, :]
            comps['task_params'] = tp_stack
        
        for k, v in comps.items():
            if v is None: continue
            v = v.astype(np.float64)
            # v is (B, W, D). Combine B and W
            B, W, D = v.shape
            v_flat = v.reshape(-1, D)
            count = v_flat.shape[0]
            
            s = np.sum(v_flat, axis=0) # (D,)
            ss = np.sum(v_flat**2, axis=0) # (D,)

            if k not in accumulators:
                if self.normalization_type == "mean_std":
                    accumulators[k] = {'sum': s, 'sq_sum': ss, 'count': count}
                else:
                    accumulators[k] = {'min': np.min(v_flat, axis=0), 'max': np.max(v_flat, axis=0)}
            else:
                if self.normalization_type == "mean_std":
                    accumulators[k]['sum'] += s
                    accumulators[k]['sq_sum'] += ss
                    accumulators[k]['count'] += count
                else:
                    accumulators[k]['min'] = np.minimum(accumulators[k]['min'], np.min(v_flat, axis=0))
                    accumulators[k]['max'] = np.maximum(accumulators[k]['max'], np.max(v_flat, axis=0))

    def _load_stats(self):
        print(f"Loading stats from {self.norm_path}")
        with np.load(self.norm_path, allow_pickle=True) as data:
            for k in data.keys():
                self.stats[k] = torch.from_numpy(data[k]).float()
        
        self._update_global_stats()
        
    def _update_global_stats(self):
        # Precompute global stats for concatenated features
        means = []
        stds = []
        mins = []
        maxs = []
        
        for key in self.feature_order:
             if self.normalization_type == "min_max":
                 mk, MK = f"min_{key}", f"max_{key}"
                 if mk in self.stats and MK in self.stats:
                     mins.append(self.stats[mk])
                     maxs.append(self.stats[MK])
             else:
                 mk, sk = f"mean_{key}", f"std_{key}"
                 if mk in self.stats and sk in self.stats:
                     means.append(self.stats[mk])
                     stds.append(self.stats[sk])
        
        if self.normalization_type == "min_max":
            if mins:
                self.global_min = torch.cat(mins, dim=-1)
                self.global_max = torch.cat(maxs, dim=-1)
        else:
            if means:
                self.global_mean = torch.cat(means, dim=-1)
                self.global_std = torch.cat(stds, dim=-1)

    def denormalize_global(self, tensor):
        """Denormalize a concatenated feature tensor using global stats."""
        if self.normalization_type == "min_max":
            if hasattr(self, 'global_min') and hasattr(self, 'global_max'):
                device = tensor.device
                min_val = self.global_min.to(device)
                max_val = self.global_max.to(device)
                denom = max_val - min_val
                # clamp just in case? no, linear
                return tensor * denom + min_val
            return tensor
            
        else:
            if hasattr(self, 'global_mean') and hasattr(self, 'global_std'):
                device = tensor.device
                mean_val = self.global_mean.to(device)
                std_val = self.global_std.to(device)
                return tensor * std_val + mean_val
            return tensor

    def __len__(self):
        return len(self.indices)
