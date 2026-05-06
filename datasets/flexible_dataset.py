import numpy as np
import torch
import os
import glob
import re
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.math.sbto_utils import compute_sbto_components, yaw_to_rot_matrix, yaw_from_quat, rot6d_to_rot, quat_to_rot, compute_task_params, compute_task_params_reorient, rot_to_6d, build_feature_layout
from utils.math.rotation_conversions import ( 
    axis_angle_to_quaternion, 
)
from utils.data.style_conditioning import (
    classify_style_from_motion,
    classify_style_from_task_text,
    load_task_text_map,
    one_hot_style,
    style_name_to_id,
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
        motion_styles=None,
    ):
        training_cfg = training_cfg or {}
        
        self.noise_cfg = training_cfg.get("state_conditioning_noise_level", {})
        self.num_observations = config.get("num_observations", 45)
        self.num_features = config.get("num_features", 48)
        self.history_size = config.get("state_history", 4)
        self.window_size = config.get("num_timesteps", 50) // config.get("downsample", 1)
        self.stride = config.get("stride", 1)
        self.downsample = config.get("downsample", 1)
        self.start_timestep = config.get("start_timestep", 0)
        self.normalize_goal_vec = config.get("normalize_goal_vec", True)
        self.num_task_params = config.get("num_task_params", 2)
        self.task_type = config.get("task_type", "pickplace")
        self.normalization_type = config.get("normalization_type", "mean_std")  # 'mean_std' or 'min_max'
        self.feature_order = config.get("feature_order", 
                [
                    "delta_xy", "delta_yaw", "obj_delta_xy", "obj_z",
                    "joints", "body_z", "body_rot6d", 
                    "obj_rel_pos", "obj_rel_rot6d",
                ]
        )

        self.key_mapping = config.get("key_mapping", {})
        style_cfg = config.get("style_conditioning", {})
        self.style_condition = bool(style_cfg.get("enabled", False))
        self.num_styles = int(style_cfg.get("num_styles", 3))
        self.style_detection_mode = str(style_cfg.get("detection_mode", "motion_then_text"))
        self._style_rng = np.random.default_rng(int(style_cfg.get("seed", 42)))
        self._task_text_by_folder = load_task_text_map(style_cfg.get("tasks_file", None))
        
        self.state_condition = config.get("state_condition", True)
        self.task_condition = config.get("task_condition", True)
        
        self.full_feature_order = config.get("full_feature_order", self.feature_order)
        self.full_num_features = config.get("full_num_features", self.num_features)
        self.full_num_observations = config.get("full_num_observations", self.num_observations)
        
        print(f"Using feature order: {self.feature_order}")
        print(f"Using normalization type: {self.normalization_type}")
        print(f"Using key mapping: {self.key_mapping}")
        if self.style_condition:
            print(
                f"Style conditioning enabled (num_styles={self.num_styles}, "
                f"mode={self.style_detection_mode})"
            )

        self.add_noise = training_cfg.get("add_obs_noise", False)
        self.add_goal_noise = training_cfg.get("add_goal_noise", False)

        # ── G1 left/right mirror symmetry augmentation ──────────────
        aug_cfg = config.get("augmentation", {})
        mirror_cfg = aug_cfg.get("mirror_symmetry", {})
        self.mirror_enabled = mirror_cfg.get("enabled", False)
        self.mirror_prob = mirror_cfg.get("prob", 0.5)
        if self.mirror_enabled:
            print(f"Mirror symmetry augmentation enabled (prob={self.mirror_prob})")

        # ── Carry-walk approach-phase augmentation ───────────────────
        # Uses trajectories where the robot walks while holding the box.
        # Robot pose is kept unchanged; the box is frozen at its final drop
        # location throughout, making it look like the robot walks toward a
        # stationary box and bends to interact with it.
        approach_cfg = aug_cfg.get("approach_phase", {})
        self.approach_aug_enabled = approach_cfg.get("enabled", False)
        self.carry_walk_aug_prob  = float(approach_cfg.get("carry_walk_aug_prob", 0.5))
        self._approach_rng = np.random.default_rng(int(approach_cfg.get("seed", 123)))
        if self.approach_aug_enabled:
            print(
                f"Carry-walk approach augmentation enabled "
                f"(prob={self.carry_walk_aug_prob})"
            )

        self._raw_buffer = data_buffer
        self._task_params = task_params
        self._motion_styles = motion_styles
        
        # ---- Preload buffer into ram_cache ----
        self.ram_cache = []
        self.traj_lengths = []
        self.real_traj_lengths = []  # actual (pre-padding) lengths per file
        self.indices = []

        self._preload_from_buffer()
        self._index_dataset()

        # ---- Normalization ----
        self.norm_path = norm_path
        self.stats = {}

        self.max_obj_displacement = config.get("max_obj_displacement", None)

        if norm_path and os.path.exists(norm_path) and not calculate_stats:
            self._load_stats()
        else:
            self._calculate_stats()
        
        if self.max_obj_displacement is None and self.task_type != "reorient":
            self.max_obj_displacement = self.stats.get("max_task_params")[-1]
            print("Using max_obj_displacement =", self.max_obj_displacement)

    @property
    def feature_dims(self):
        """Return an array of per-feature dimensions matching feature_order."""
        layout = build_feature_layout()
        return np.array([layout[k] for k in self.feature_order])


    def get_k(self, key):
        """Helper to get mapped key from config key_mapping."""
        return self.key_mapping.get(key, key)

    def _convert_schema(self, raw_data):
        """
        Convert a single raw .npz dict to standardized keys.

        Output temporal keys (shape ≥2D, will get batch dim added later):
            base     - (..., T, 7)  pelvis xyz + quaternion
            joints   - (..., T, 29) joint positions
            obj      - (..., T, 7)  object xyz + quaternion
            base_vel, joints_vel, obj_vel, ee_rel_pos  (optional)

        Non-temporal keys passed through: fps

        Schemas detected:
            1. Already converted  — 'base' and 'joints' present
            2. RL Rollout         — body_pos_w (or key-mapped, e.g. root_pos)
            3. SBTO               — base_xyz_quat
        """
        keys = set(raw_data.keys())

        # Convert any torch tensors to numpy
        for k in list(raw_data.keys()):
            if isinstance(raw_data[k], torch.Tensor):
                raw_data[k] = raw_data[k].cpu().numpy()

        # ---- Already in standardized format ----
        if 'base' in keys and 'joints' in keys:
            return {k: np.asarray(v) if isinstance(v, np.ndarray) else v
                    for k, v in raw_data.items()}

        processed = {}

        # ---- RL Rollout schema (body_pos_w or key-mapped equivalent) ----
        bp_key = 'body_pos_w' if 'body_pos_w' in keys else (
            self.get_k('body_pos_w') if self.get_k('body_pos_w') in keys else None
        )

        if bp_key is not None:
            bq_key = 'body_quat_w' if 'body_quat_w' in keys else self.get_k('body_quat_w')
            jp_key = 'joint_pos'   if 'joint_pos'   in keys else self.get_k('joint_pos')
            op_key = 'object_pos_w' if 'object_pos_w' in keys else self.get_k('object_pos_w')
            oq_key = 'object_quat_w' if 'object_quat_w' in keys else self.get_k('object_quat_w')

            bp = np.asarray(raw_data[bp_key], dtype=np.float64)
            bq = np.asarray(raw_data[bq_key], dtype=np.float64)

            # body_pos_w may have a bodies dim — take pelvis (0):
            #   batched:  (N, T, K, 3) -> (N, T, 3)
            #   single:   (T, K, 3)    -> (T, 3)
            if bp.ndim == 4:
                bp = bp[:, :, 0, :]
            elif bp.ndim == 3 and bp_key in ['body_pos_w']:
                bp = bp[:, 0, :]
            if bq.ndim == 4:
                bq = bq[:, :, 0, :]
            elif bq.ndim == 3 and bq_key in ['body_quat_w']:
                bq = bq[:, 0, :]

            processed['base'] = np.concatenate([bp, bq], axis=-1)
            processed['joints'] = np.asarray(raw_data[jp_key], dtype=np.float64)

            op = np.asarray(raw_data[op_key], dtype=np.float64)
            oq = np.asarray(raw_data[oq_key], dtype=np.float64)
            processed['obj'] = np.concatenate([op, oq], axis=-1)

            # Velocity keys (RL / key-mapped RL)
            vel_groups = [
                ('root_lin_vel', 'root_ang_vel', 'base_vel'),
                ('dof_vel', None, 'joints_vel'),
                ('object_lin_vel', 'object_ang_vel', 'obj_vel'),
            ]
            for lin_k, ang_k, out_k in vel_groups:
                if lin_k in keys:
                    lv = np.asarray(raw_data[lin_k], dtype=np.float64)
                    if ang_k and ang_k in keys:
                        av = np.asarray(raw_data[ang_k], dtype=np.float64)
                        processed[out_k] = np.concatenate([lv, av], axis=-1)
                    else:
                        processed[out_k] = lv

            if 'ee_rel_pos' in keys:
                processed['ee_rel_pos'] = np.asarray(raw_data['ee_rel_pos'], dtype=np.float64)

        # ---- SBTO schema ----
        elif 'base_xyz_quat' in keys:
            processed['base'] = np.asarray(raw_data['base_xyz_quat'], dtype=np.float64)
            processed['joints'] = np.asarray(raw_data['actuator_pos'], dtype=np.float64)

            if 'obj_0_xyz_quat' in keys:
                processed['obj'] = np.asarray(raw_data['obj_0_xyz_quat'], dtype=np.float64)

            if 'base_linvel_angvel' in keys:
                processed['base_vel'] = np.asarray(raw_data['base_linvel_angvel'], dtype=np.float64)
            if 'actuator_vel' in keys:
                processed['joints_vel'] = np.asarray(raw_data['actuator_vel'], dtype=np.float64)
            if 'obj_0_linvel_angvel' in keys:
                processed['obj_vel'] = np.asarray(raw_data['obj_0_linvel_angvel'], dtype=np.float64)
            if 'ee_rel_pos' in keys:
                processed['ee_rel_pos'] = np.asarray(raw_data['ee_rel_pos'], dtype=np.float64)

        # ---- OmniRetarget qpos schema ----
        elif 'qpos' in keys:
            qpos = np.asarray(raw_data['qpos'], dtype=np.float64)  # (T, D)

            # Floating base: qpos[..., 0:7] = [qw, qx, qy, qz, x, y, z]
            # Reorder to standardized [x, y, z, qw, qx, qy, qz]
            base_raw = qpos[..., :7]
            processed['base'] = np.concatenate(
                [base_raw[..., 4:7], base_raw[..., 0:4]], axis=-1
            )

            # Joint positions: qpos[..., 7:36]
            processed['joints'] = qpos[..., 7:36]

            # Object pose (optional): qpos[..., 36:43] = [qw, qx, qy, qz, x, y, z]
            # Reorder to standardized [x, y, z, qw, qx, qy, qz]
            if qpos.shape[-1] == 43:
                obj_raw = qpos[..., 36:43]
                processed['obj'] = np.concatenate(
                    [obj_raw[..., 4:7], obj_raw[..., 0:4]], axis=-1
                )

        else:
            print(f"Warning: Unrecognized buffer schema. Keys: {keys}")
            return None

        # Pass through non-temporal metadata
        if 'fps' in raw_data:
            processed['fps'] = raw_data['fps']
        if 'source_file_path' in raw_data:
            processed['source_file_path'] = raw_data['source_file_path']
        if 'source_folder_name' in raw_data:
            processed['source_folder_name'] = raw_data['source_folder_name']

        return processed

    def _preload_from_buffer(self):
        """
        Process _raw_buffer into ram_cache.

        Accepts:
          - list of dicts  (from preload_dataset or list-of-traj buffers)
          - single dict    (from RL buffer with batched arrays)

        For each dict: detects schema via _convert_schema, ensures batch dim,
        applies start_timestep / downsampling, computes goal_obj_world.
        """
        if not self._raw_buffer:
            return

        start_ts = self.start_timestep
        ds = self.downsample

        # Ensure buffer is a list of dicts
        buf = self._raw_buffer
        if isinstance(buf, dict):
            buf = [buf]

        total_trajs = 0
        tp_offset = 0  # running offset into the flat task_params list
        style_offset = 0
        aug_entries = []  # synthetic approach-phase trajectories accumulated here

        for raw_data in buf:
            if not raw_data:
                continue

            # Schema detection & key conversion
            processed = self._convert_schema(raw_data)
            if not processed:
                continue

            # Ensure batch dim on temporal arrays: (T, D) -> (1, T, D)
            for k in list(processed.keys()):
                v = processed[k]
                if isinstance(v, np.ndarray) and k not in self._NON_TEMPORAL_KEYS:
                    if v.ndim == 2:
                        processed[k] = v[None]

            # Apply start_timestep and downsampling
            for k in list(processed.keys()):
                if (k not in self._NON_TEMPORAL_KEYS
                        and isinstance(processed[k], np.ndarray)
                        and processed[k].ndim == 3):
                    processed[k] = processed[k][:, start_ts::ds]

            N_i = processed['base'].shape[0]

            # Compute goal_obj_world from task_params if provided
            if self._task_params is not None:
                goal_world = np.zeros((N_i, 3), dtype=np.float64)
                for j in range(N_i):
                    goal_local = np.asarray(self._task_params[tp_offset + j], dtype=np.float64)

                    init_base = processed['base'][j, 0, :]   # (7,) [x,y,z,qw,qx,qy,qz]
                    init_obj  = processed['obj'][j, 0, :3]    # (3,)

                    init_yaw = yaw_from_quat(init_base[3:7])
                    R = yaw_to_rot_matrix(init_yaw)

                    goal_3d = np.zeros(3, dtype=np.float64)
                    n = min(len(goal_local), 3)
                    goal_3d[:n] = goal_local[:n]

                    goal_world[j] = (R @ goal_3d[:, None])[:, 0] + init_obj

                processed['goal_obj_world'] = goal_world  # (N_i, 3)
                tp_offset += N_i

            if self.style_condition:
                style_ids = np.zeros((N_i,), dtype=np.int64)
                style_onehot = np.zeros((N_i, self.num_styles), dtype=np.float32)
                for j in range(N_i):
                    if self._motion_styles is not None:
                        sid = self._motion_styles[style_offset + j]
                        if isinstance(sid, str):
                            sid = style_name_to_id(sid)
                    else:
                        sid = self._infer_style_for_trajectory(processed, j)
                    style_ids[j] = int(sid)
                    style_onehot[j] = one_hot_style(sid, num_styles=self.num_styles)
                processed['style_id'] = style_ids
                processed['style_onehot'] = style_onehot
                if self._motion_styles is not None:
                    style_offset += N_i

            self.ram_cache.append(processed)
            total_trajs += N_i

            # ── Carry-walk approach-phase augmentation ───────────────────
            if self.approach_aug_enabled:
                # Derive per-trajectory real lengths in downsampled frames.
                if 'traj_lengths' in raw_data:
                    raw_lens = np.asarray(raw_data['traj_lengths'], dtype=int)
                    real_lens_ds = np.maximum(1, (raw_lens - start_ts - 1) // ds + 1)
                    if len(real_lens_ds) < N_i:
                        real_lens_ds = np.full(N_i, processed['base'].shape[1], dtype=int)
                elif 'traj_length' in raw_data:
                    v = int(np.asarray(raw_data['traj_length']))
                    real_lens_ds = np.full(N_i, max(1, (v - start_ts - 1) // ds + 1), dtype=int)
                else:
                    real_lens_ds = np.full(N_i, processed['base'].shape[1], dtype=int)

                vel_keys = [k for k in ('base_vel', 'joints_vel', 'obj_vel', 'ee_rel_pos')
                            if k in processed]

                for b in range(N_i):
                    if self._approach_rng.random() > self.carry_walk_aug_prob:
                        continue

                    if not self._is_carry_walk_trajectory(processed, b):
                        continue

                    T_real = min(int(real_lens_ds[b]), processed['base'].shape[1])
                    base_b   = processed['base'][b,   :T_real]
                    obj_b    = processed['obj'][b,    :T_real]
                    joints_b = processed['joints'][b, :T_real]

                    extras = {k: processed[k][b, :T_real] for k in vel_keys}

                    aug = self._try_carry_approach_augmentation(
                        base_b, obj_b, joints_b,
                        extra_arrays=extras if extras else None,
                    )
                    if aug is None:
                        continue

                    if self.style_condition and 'style_id' in processed:
                        src_sid = int(processed['style_id'][b])
                        aug['style_id']     = np.array([src_sid], dtype=np.int64)
                        aug['style_onehot'] = one_hot_style(
                            src_sid, num_styles=self.num_styles
                        )[np.newaxis].astype(np.float32)

                    aug_entries.append(aug)

        # Append augmented trajectories as individual single-traj ram_cache entries.
        for aug_dict in aug_entries:
            self.ram_cache.append(aug_dict)
            total_trajs += 1

        if aug_entries:
            print(f"  Added {len(aug_entries)} synthetic approach-phase trajectories.")

        print(f"Loaded {len(self.ram_cache)} files ({total_trajs} total trajectories) into RAM cache.")

    # Keys that are non-temporal (no time dimension to pad)
    _NON_TEMPORAL_KEYS = {
        'goal_obj_world',
        'goal_obj_quat',
        'fps',
        'source_file_path',
        'source_folder_name',
        'style_id',
        'style_onehot',
    }

    def _infer_style_for_trajectory(self, processed: dict, traj_idx: int) -> int:
        base = processed['base'][traj_idx]
        joints = processed['joints'][traj_idx]
        obj = processed['obj'][traj_idx]

        motion_style_id, motion_info = classify_style_from_motion(base, joints, obj)

        # Prefer file stem as lookup key (data may be flat .npz files).
        task_text = ""
        fpath = processed.get('source_file_path')
        if isinstance(fpath, np.ndarray):
            fpath = fpath.item() if fpath.ndim == 0 else fpath[traj_idx]
        if fpath:
            stem = os.path.splitext(os.path.basename(str(fpath)))[0]
            task_text = self._task_text_by_folder.get(stem, "")
        if not task_text:
            folder = processed.get('source_folder_name')
            if isinstance(folder, np.ndarray):
                folder = folder.item() if folder.ndim == 0 else folder[traj_idx]
            if folder:
                task_text = self._task_text_by_folder.get(str(folder), "")
        text_style_id = classify_style_from_task_text(task_text, rng=self._style_rng)

        # Keep explicit mixed kick+pick random rule from task text when available.
        task_text_l = str(task_text).lower()
        if ("kick" in task_text_l) and any(k in task_text_l for k in ["pick", "hold", "place", "carry"]) and (text_style_id is not None):
            return int(text_style_id)

        mode = self.style_detection_mode
        if mode == "motion_only":
            style_id = motion_style_id
        elif mode in ("text_only", "text_then_motion"):
            style_id = text_style_id if text_style_id is not None else motion_style_id
        else:
            # motion_then_text
            confidence = float(motion_info.get("confidence", 0.0)) if isinstance(motion_info, dict) else 0.0
            style_id = text_style_id if (confidence < 0.6 and text_style_id is not None) else motion_style_id
        return int(style_id)

    def _get_single_traj(self, file_idx, batch_idx):
        """Get padded (T, D) dict for a specific trajectory.
        Trims to real (pre-padding) length before padding so that
        windows near the tail of short trajectories see edge-padded
        copies of the last real frame rather than garbage.
        Non-temporal keys (goal_obj_world, fps) are passed through without padding.
        """
        file_data = self.ram_cache[file_idx]

        # Real (non-padded) length for this specific trajectory
        real_T = self.real_traj_lengths[file_idx][batch_idx] if self.real_traj_lengths else None

        pad_start = self.history_size
        target_padded = self.traj_lengths[file_idx]

        def _trim_and_pad(arr):
            # Trim to real length, then pad to target_padded
            if real_T is not None and arr.shape[0] > real_T:
                arr = arr[:real_T]
            pad_end = target_padded - (arr.shape[0] + pad_start)
            pad_end = max(pad_end, 0)
            if arr.ndim == 2:
                return np.pad(arr, ((pad_start, pad_end), (0, 0)), mode='edge')
            else:
                return np.pad(arr, ((0, 0),), mode='edge')

        result = {}
        for k, v in file_data.items():
            if k in self._NON_TEMPORAL_KEYS:
                # Non-temporal — just index batch dim, no padding
                result[k] = v[batch_idx] if hasattr(v, '__getitem__') and np.asarray(v).ndim >= 1 else v
            else:
                result[k] = _trim_and_pad(v[batch_idx])
        return result
        

    def _index_dataset(self):
        """
        Iterate through files to determine valid window start indices.
        Mimics create_dataset.py padding and windowing logic.
        Restricts window starts so each window overlaps real (non-padded) data.
        """
        print("Indexing dataset...")
        for i in range(len(self.ram_cache)):
            try:
                data = self.ram_cache[i]
                
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
                
                # Check for per-trajectory real lengths from metadata
                if 'traj_length' in data:
                    _tl = np.asarray(data['traj_length'])
                    if _tl.ndim == 0:
                        real_lengths = [int(_tl)] * B
                    else:
                        real_lengths = [int(_tl[b]) for b in range(B)]
                elif 'traj_lengths' in data:
                    _tl = np.asarray(data['traj_lengths'])
                    real_lengths = [int(_tl[b]) for b in range(B)]
                else:
                    real_lengths = [T] * B
                self.real_traj_lengths.append(real_lengths)
                
                # Data is already downsampled in _preload_from_buffer
                T_down = T

                pad_start = self.history_size
                # Ensure sufficient padding so that the trailing window can have its history 
                # fully overlapping the end of the real trajectory (future is padding).
                pad_end = self.window_size + self.history_size
            
                T_padded = T_down + pad_start + pad_end
                self.traj_lengths.append(T_padded)
                
                w_size = self.window_size
                
                for b in range(B):
                    # Restrict to real data: window history part must be valid real data.
                    # This allows the very last real states to be used as history conditioning
                    # for a window whose future is entirely padding (stationary target).
                    real_T = real_lengths[b]
                    max_start = min(T_padded - w_size, real_T + pad_start - self.history_size)
                    max_start = max(max_start, 0)
                    
                    if max_start >= 0:
                        starts = np.arange(0, max_start + 1, self.stride)
                        for s in starts:
                            self.indices.append((i, b, s))

            except Exception as e:
                print(f"Error indexing file {i}: {e}")
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
            goal_quat = raw_traj.get('goal_obj_quat', raw_traj['obj'][-1, 3:7])
            if self.task_type == "reorient":
                features["task_params"] = compute_task_params_reorient(
                    current_robot_state=raw_traj['base'][read_start, 3:],
                    current_obj_state=raw_traj['obj'][read_start, :7],
                    desired_obj_z=goal_world[2],
                    desired_obj_quat=goal_quat,
                )
            else:
                features["task_params"], _ = compute_task_params(
                    current_robot_state=raw_traj['base'][read_start, 3:],
                    current_obj_state=raw_traj['obj'][read_start, :3],
                    desired_obj_pos=goal_world,
                    normalize_goal_vec=self.normalize_goal_vec,
                    num_task_params=self.num_task_params,
                    max_goal_dist=self.max_obj_displacement,
                    current_obj_rot=raw_traj['obj'][read_start, 3:7],
                    desired_obj_rot=goal_quat,
                )
        else:
            anchor["final_obj_pos"] = raw_traj['obj'][-1, :3]
            features["task_params"], _ = self._compute_task_params(
                raw_traj['base'], raw_traj['obj'], start_idx=read_start,
            )

        return features, anchor

    def _compute_task_params(self, base_traj, obj_traj, start_idx=0,
                              desired_obj_quat=None, current_obj_quat=None):
        if self.task_type == "reorient":
            des_quat = desired_obj_quat if desired_obj_quat is not None else obj_traj[-1, 3:7]
            tp = compute_task_params_reorient(
                current_robot_state=base_traj[start_idx, 3:],
                current_obj_state=obj_traj[start_idx, :7],
                desired_obj_z=obj_traj[-1, 2],
                desired_obj_quat=des_quat,
            )
            return tp, None
        return compute_task_params(
            current_robot_state=base_traj[start_idx, 3:],
            current_obj_state=obj_traj[start_idx, :3],
            desired_obj_pos=obj_traj[-1, :3],
            normalize_goal_vec=self.normalize_goal_vec,
            num_task_params=self.num_task_params,
            max_goal_dist=self.max_obj_displacement,
            current_obj_rot=current_obj_quat if current_obj_quat is not None else obj_traj[start_idx, 3:7],
            desired_obj_rot=desired_obj_quat if desired_obj_quat is not None else obj_traj[-1, 3:7],
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

    # ────────────────────────────────────────────────────────────────
    # Carry-walk approach-phase augmentation
    # ────────────────────────────────────────────────────────────────
    def _is_carry_walk_trajectory(self, processed: dict, traj_idx: int) -> bool:
        """Return True if the trajectory involves walking while holding the box."""
        # Data may be stored as flat .npz files (e.g. sub3_largebox_003_original.npz),
        # so the meaningful lookup key is the file stem, not the parent folder name.
        def _get_lookup_key(val):
            if val is None:
                return None
            if isinstance(val, np.ndarray):
                val = val.item() if val.ndim == 0 else val[traj_idx]
            return str(val)

        key = None
        fpath = _get_lookup_key(processed.get('source_file_path'))
        if fpath:
            stem = os.path.splitext(os.path.basename(fpath))[0]
            if stem in self._task_text_by_folder:
                key = stem

        if key is None:
            folder = _get_lookup_key(processed.get('source_folder_name'))
            if folder and folder in self._task_text_by_folder:
                key = folder

        if key is None:
            return False

        task_text = self._task_text_by_folder[key].lower()
        has_carry = "carry" in task_text
        has_hold_walk = "hold" in task_text and any(
            d in task_text for d in ["forward", "to right", "to left"]
        )
        return has_carry or has_hold_walk

    def _try_carry_approach_augmentation(self, base, obj, joints, extra_arrays=None):
        """
        Create a synthetic approach trajectory from a carry-while-walking trajectory.

        The robot's base and joint trajectory are kept exactly as recorded.
        The object is frozen at its final resting position (drop location)
        for the entire trajectory, so the sequence reads as: robot walks toward
        a stationary box, then bends down to interact with it.

        Args:
            base:         (T, 7)  world-frame pelvis [x, y, z, qw, qx, qy, qz]
            obj:          (T, 7)  world-frame object [x, y, z, qw, qx, qy, qz]
            joints:       (T, 29) joint positions
            extra_arrays: optional dict of {key: (T, D)} velocity / ee arrays

        Returns:
            dict with (1, T, D) arrays and 'goal_obj_world' (1, 3), or None.
        """
        T = base.shape[0]
        if T < 2:
            return None

        goal_obj_world = obj[-1, :3].copy()
        goal_obj_quat  = obj[-1, 3:7].copy()

        # Freeze object at final drop location for the entire trajectory.
        obj_frozen = np.empty_like(obj)
        obj_frozen[:] = obj[-1]

        aug = {
            'base':           base[np.newaxis].astype(np.float64),
            'joints':         joints[np.newaxis].astype(np.float64),
            'obj':            obj_frozen[np.newaxis].astype(np.float64),
            'goal_obj_world': goal_obj_world[np.newaxis].astype(np.float64),
            'goal_obj_quat':  goal_obj_quat[np.newaxis].astype(np.float64),
        }

        if extra_arrays:
            for k_e, arr in extra_arrays.items():
                if k_e == 'obj_vel':
                    aug[k_e] = np.zeros_like(arr)[np.newaxis].astype(np.float64)
                else:
                    aug[k_e] = arr[np.newaxis].astype(np.float64)

        return aug

    # ────────────────────────────────────────────────────────────────
    # G1 left/right mirror symmetry (C2 sagittal-plane reflection)
    # ────────────────────────────────────────────────────────────────
    def _build_mirror_tables(self):
        """Pre-compute permutation and sign-flip tables for the feature vector
        and the observation (current_state) vector.

        G1 joint ordering (29 DOF, indices within the joints block):
            0-5   left  hip-pitch, hip-roll, hip-yaw, knee, ankle-pitch, ankle-roll
            6-11  right hip-pitch, hip-roll, hip-yaw, knee, ankle-pitch, ankle-roll
            12-14 waist yaw, roll, pitch
            15-21 left  shoulder-pitch, shoulder-roll, shoulder-yaw, elbow,
                        wrist-roll, wrist-pitch, wrist-yaw
            22-28 right shoulder-pitch, shoulder-roll, shoulder-yaw, elbow,
                        wrist-roll, wrist-pitch, wrist-yaw

        Under sagittal-plane reflection (left ↔ right):
          • Left/right limb blocks are swapped.
          • Roll and yaw joints negate (they change sign under reflection).
          • Pitch joints stay the same.
        """
        layout = build_feature_layout()
        D = sum(layout[k] for k in self.feature_order)

        # ── Feature-vector index map ────────────────────────────────
        fidx = 0
        f_map = {}
        for key in self.feature_order:
            dim = layout.get(key, 0)
            f_map[key] = slice(fidx, fidx + dim)
            fidx += dim

        # ── Permutation & sign arrays (default: identity / +1) ─────
        perm = np.arange(D, dtype=np.int64)
        sign = np.ones(D, dtype=np.float32)

        # 1. delta_xy: negate y
        if "delta_xy" in f_map:
            s = f_map["delta_xy"]
            sign[s.start + 1] = -1.0          # delta_y

        # 2. delta_yaw: negate
        if "delta_yaw" in f_map:
            s = f_map["delta_yaw"]
            sign[s.start] = -1.0

        # 3. obj_delta_xy: negate y
        if "obj_delta_xy" in f_map:
            s = f_map["obj_delta_xy"]
            sign[s.start + 1] = -1.0           # obj_delta_y

        # 4. obj_z: invariant (no change needed)

        # 5. joints: swap left↔right, negate roll/yaw
        if "joints" in f_map:
            js = f_map["joints"].start

            # Explicit permutation based on user specification:
            #   [0-5]   (L Leg) <- [6-11]  (R Leg)
            #   [6-11]  (R Leg) <- [0-5]   (L Leg)
            #   [12-14] (Waist) <- [12-14] (Waist)
            #   [15-21] (L Arm) <- [22-28] (R Arm)
            #   [22-28] (R Arm) <- [15-21] (L Arm)
            perm_indices = [
                 6,  7,  8,  9, 10, 11,      # 0-5
                 0,  1,  2,  3,  4,  5,      # 6-11
                12, 13, 14,                  # 12-14
                22, 23, 24, 25, 26, 27, 28,  # 15-21
                15, 16, 17, 18, 19, 20, 21   # 22-28
            ]
            for i, p_idx in enumerate(perm_indices):
                perm[js + i] = js + p_idx

            # Explicit sign flips based on user specification:
            # reflection_Q_js: [[
            #   1, -1, -1, 1, 1, -1,       # L leg
            #   1, -1, -1, 1, 1, -1,       # R leg
            #   -1, -1, 1,                 # Waist
            #   1, -1, -1, 1, -1, 1, -1,   # L arm
            #   1, -1, -1, 1, -1, 1, -1    # R arm
            # ]]
            flip_vec = [
                1, -1, -1, 1, 1, -1,      # 0-5
                1, -1, -1, 1, 1, -1,      # 6-11
               -1, -1,  1,                # 12-14
                1, -1, -1, 1, -1, 1, -1,  # 15-21
                1, -1, -1, 1, -1, 1, -1   # 22-28
            ]
            for i, s_val in enumerate(flip_vec):
                if s_val == -1:
                    sign[js + i] = -1.0

        # 6. body_z: invariant

        # 7. body_rot6d: [r1x, r1y, r1z, r2x, r2y, r2z]
        #    Under S=diag(1,-1,1): → [r1x, -r1y, r1z, -r2x, r2y, -r2z]
        if "body_rot6d" in f_map:
            s = f_map["body_rot6d"]
            sign[s.start + 1] = -1.0  # r1y
            sign[s.start + 3] = -1.0  # r2x
            sign[s.start + 5] = -1.0  # r2z

        # 8. obj_rel_pos: [x, y, z] → [x, -y, z]
        if "obj_rel_pos" in f_map:
            s = f_map["obj_rel_pos"]
            sign[s.start + 1] = -1.0  # y

        # 9. obj_rel_rot6d: same sign pattern as body_rot6d
        if "obj_rel_rot6d" in f_map:
            s = f_map["obj_rel_rot6d"]
            sign[s.start + 1] = -1.0  # r1y
            sign[s.start + 3] = -1.0  # r2x
            sign[s.start + 5] = -1.0  # r2z

        self._mirror_perm = perm
        self._mirror_sign = sign                   # numpy arrays

        # ── Observation-vector tables (joints, body_z, body_rot6d,
        #    obj_rel_pos, obj_rel_rot6d = 45 dims) ─────────────────
        obs_D = self.num_observations  # 45
        obs_perm = np.arange(obs_D, dtype=np.int64)
        obs_sign = np.ones(obs_D, dtype=np.float32)

        # Joints (0-28): reuse explicit values from user spec
        for i, p_idx in enumerate(perm_indices):
            obs_perm[i] = p_idx
        for i, s_val in enumerate(flip_vec):
            if s_val == -1:
                obs_sign[i] = -1.0

        # body_z (29): invariant
        # body_rot6d (30-35)
        obs_sign[31] = -1.0; obs_sign[33] = -1.0; obs_sign[35] = -1.0
        # obj_rel_pos (36-38): negate y
        obs_sign[37] = -1.0
        # obj_rel_rot6d (39-44)
        obs_sign[40] = -1.0; obs_sign[42] = -1.0; obs_sign[44] = -1.0

        self._mirror_obs_perm = obs_perm
        self._mirror_obs_sign = obs_sign

        # Task param sign under L/R reflection depends on task type.
        if self.task_type == "reorient":
            # Layout: [delta_z, delta_roll, delta_yaw]
            # delta_z: invariant; delta_roll: negates; delta_yaw: negates
            _tp_sign_base = np.array([1.0, -1.0, -1.0], dtype=np.float32)
        else:
            # Layout: [goal_dir_x, goal_dir_y, goal_dir_z, goal_dist, (delta_yaw)]
            # y-direction and yaw negate; x, z, dist unchanged.
            _tp_sign_base = np.array([1.0, -1.0, 1.0, 1.0, -1.0], dtype=np.float32)
        self._mirror_task_sign = _tp_sign_base[:self.num_task_params]

    def _build_mirror_norm_bias(self):
        """
        Pre-compute per-feature additive bias needed to correctly apply sign-flips
        in *normalised* space.

        For a sign-flip on feature i (sign[i] == -1) the raw operation is x' = -x.
        Under min-max normalisation: x_n = (x - lo) / (hi - lo)
          → x'_n = (-x - lo) / (hi - lo)
                  = -(x - lo)/(hi - lo) - 2*lo/(hi - lo)
                  = -x_n  +  bias_i
        where  bias_i = -2 * lo_i / (hi_i - lo_i).
        Under mean-std normalisation: x_n = (x - mu) / sigma
          → x'_n = -(x - mu)/sigma - 2*mu/sigma
                  = -x_n  +  bias_i
        where  bias_i = -2 * mu_i / sigma_i.

        For features that are NOT flipped (sign[i] == +1), bias = 0.
        This correction makes the mirror exactly equivalent to working in raw space.
        """
        layout  = build_feature_layout()
        D_feat  = sum(layout[k] for k in self.feature_order)
        D_obs   = self.num_observations

        # ── feature-vector bias ─────────────────────────────────────
        feat_bias = np.zeros(D_feat, dtype=np.float32)
        fidx = 0
        for key in self.feature_order:
            dim = layout.get(key, 0)
            sl  = slice(fidx, fidx + dim)
            if self.normalization_type == "min_max":
                lo = self.stats.get(f"min_{key}")
                hi = self.stats.get(f"max_{key}")
                if lo is not None and hi is not None:
                    lo_np = lo.numpy()
                    hi_np = hi.numpy()
                    denom = hi_np - lo_np
                    denom = np.where(np.abs(denom) < 1e-6, 1.0, denom)
                    feat_bias[sl] = -2.0 * lo_np / denom
            else:   # mean_std
                mu  = self.stats.get(f"mean_{key}")
                sig = self.stats.get(f"std_{key}")
                if mu is not None and sig is not None:
                    mu_np  = mu.numpy()
                    sig_np = sig.numpy()
                    sig_np = np.where(np.abs(sig_np) < 1e-6, 1.0, sig_np)
                    feat_bias[sl] = -2.0 * mu_np / sig_np
            fidx += dim
        # Only apply bias where sign == -1
        feat_bias = np.where(self._mirror_sign == -1.0, feat_bias, 0.0)
        self._mirror_feat_bias = feat_bias.astype(np.float32)

        # ── observation-vector bias ──────────────────────────────────
        # Observation = the last num_observations dims of the feature vector,
        # i.e. feature indices [D_feat - D_obs : D_feat].
        obs_feat_bias = feat_bias[D_feat - D_obs:]
        # However the obs sign table is independent; reconstruct bias from obs sign
        obs_bias = np.zeros(D_obs, dtype=np.float32)
        obs_sign_nz = self._mirror_obs_sign  # (-1 where flipped)
        obs_keys_in_order = [
            k for k in self.feature_order
            if k in ("joints", "body_z", "body_rot6d", "obj_rel_pos", "obj_rel_rot6d")
        ]
        oidx = 0
        for key in obs_keys_in_order:
            dim = layout.get(key, 0)
            sl  = slice(oidx, oidx + dim)
            if self.normalization_type == "min_max":
                lo = self.stats.get(f"min_{key}")
                hi = self.stats.get(f"max_{key}")
                if lo is not None and hi is not None:
                    lo_np = lo.numpy()
                    hi_np = hi.numpy()
                    denom = hi_np - lo_np
                    denom = np.where(np.abs(denom) < 1e-6, 1.0, denom)
                    obs_bias[sl] = np.where(
                        obs_sign_nz[sl] == -1.0,
                        -2.0 * lo_np / denom,
                        0.0
                    )
            else:
                mu  = self.stats.get(f"mean_{key}")
                sig = self.stats.get(f"std_{key}")
                if mu is not None and sig is not None:
                    mu_np  = mu.numpy()
                    sig_np = sig.numpy()
                    sig_np = np.where(np.abs(sig_np) < 1e-6, 1.0, sig_np)
                    obs_bias[sl] = np.where(
                        obs_sign_nz[sl] == -1.0,
                        -2.0 * mu_np / sig_np,
                        0.0
                    )
            oidx += dim
        self._mirror_obs_bias = obs_bias.astype(np.float32)

        # ── task-params bias ────────────────────────────────────────
        n_tp = len(self._mirror_task_sign)
        tp_bias = np.zeros(n_tp, dtype=np.float32)
        if self.normalization_type == "min_max":
            lo = self.stats.get("min_task_params")
            hi = self.stats.get("max_task_params")
            if lo is not None and hi is not None:
                lo_np = lo.numpy()[:n_tp]
                hi_np = hi.numpy()[:n_tp]
                denom = hi_np - lo_np
                denom = np.where(np.abs(denom) < 1e-6, 1.0, denom)
                tp_bias = np.where(
                    self._mirror_task_sign == -1.0,
                    -2.0 * lo_np / denom,
                    0.0
                ).astype(np.float32)
        else:
            mu  = self.stats.get("mean_task_params")
            sig = self.stats.get("std_task_params")
            if mu is not None and sig is not None:
                mu_np  = mu.numpy()[:n_tp]
                sig_np = sig.numpy()[:n_tp]
                sig_np = np.where(np.abs(sig_np) < 1e-6, 1.0, sig_np)
                tp_bias = np.where(
                    self._mirror_task_sign == -1.0,
                    -2.0 * mu_np / sig_np,
                    0.0
                ).astype(np.float32)
        self._mirror_task_bias = tp_bias

    def _apply_mirror_symmetry(
        self,
        window_tensor: torch.Tensor,
        current_state: torch.Tensor,
        task_params: torch.Tensor,
    ):
        """Apply C2 sagittal-plane reflection (left↔right mirror).

        Operates correctly in *normalised* space regardless of whether
        min-max or mean-std normalisation is used.

        For a sign-flip on feature i, the normalised-space operation is:
            x_n' = -x_n + bias_i
        where bias_i = -2 * lo_i / (hi_i - lo_i)  (min-max)
                     = -2 * mu_i / sigma_i          (mean-std)
        bias_i == 0 when the distribution is symmetric around zero, which is
        the ideal case; the explicit correction handles the asymmetric case
        (e.g. body_z, which is always positive).

        Permutation (left↔right joint swap) is exact in any normalised space.

        Returns:
            (window_tensor, current_state, task_params) — mirrored copies.
        """
        # Lazily build norm bias tables the first time (stats must be ready)
        if not hasattr(self, '_mirror_feat_bias'):
            self._build_mirror_norm_bias()

        dev = window_tensor.device
        perm_t      = torch.from_numpy(self._mirror_perm).long().to(dev)
        sign_t      = torch.from_numpy(self._mirror_sign).to(dev)
        feat_bias_t = torch.from_numpy(self._mirror_feat_bias).to(dev)

        # Feature trajectory: (T, D) → permute first, then sign-flip + bias
        window_tensor = window_tensor[:, perm_t] * sign_t + feat_bias_t

        # Current state: (H, obs_dim) → permute + sign-flip + bias
        obs_perm_t = torch.from_numpy(self._mirror_obs_perm).long().to(dev)
        obs_sign_t = torch.from_numpy(self._mirror_obs_sign).to(dev)
        obs_bias_t = torch.from_numpy(self._mirror_obs_bias).to(dev)
        current_state = current_state[:, obs_perm_t] * obs_sign_t + obs_bias_t

        # Task params: negate y component + bias correction
        task_sign_t = torch.from_numpy(self._mirror_task_sign).to(dev)
        task_bias_t = torch.from_numpy(self._mirror_task_bias).to(dev)
        tp_dim = task_params.shape[-1]
        task_params = task_params * task_sign_t[:tp_dim] + task_bias_t[:tp_dim]

        return window_tensor, current_state, task_params


    def __getitem__(self, idx):
        file_idx, batch_idx, t_start = self.indices[idx]
        
        # Use RAM-cached data
        raw_traj = self._get_single_traj(file_idx, batch_idx)
        # Compute features
        features, anchor = self._compute_transform(raw_traj, t_start)

        # Mirror symmetry augmentation (Raw Feature Space)
        if self.mirror_enabled and torch.rand(1).item() < self.mirror_prob:
            self._mirror_raw_features(features)
            
            # Mirror Anchor (must copy to avoid modifying cache)
            anchor = {k: v.copy() for k, v in anchor.items()}
            
            if 'ref_pos' in anchor:
                anchor['ref_pos'][1] *= -1.0
            if 'ref_obj_pos' in anchor:
                anchor['ref_obj_pos'][1] *= -1.0
            if 'final_obj_pos' in anchor:
                anchor['final_obj_pos'][1] *= -1.0
            if 'ref_quat' in anchor:
                anchor['ref_quat'][1] *= -1.0
                anchor['ref_quat'][3] *= -1.0

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
        
        # Future trajectory
        future_states = window_tensor.clone()

        ret = [future_states]

        # Current State 
        if self.state_condition:
            if self.num_observations == self.full_num_observations and self.full_feature_order != self.feature_order:
                full_parts = []
                for key in self.full_feature_order:
                    if key in features:
                        part = torch.from_numpy(features[key]).float()
                        part = self._normalize(key, part)
                        if self.add_noise:
                            history_slice = part[:self.history_size]
                            part[:self.history_size] = self._add_obs_noise(history_slice, key)
                        full_parts.append(part)
                    else:
                        raise ValueError(f"Feature {key} needed but not computed.")
                full_window_tensor = torch.cat(full_parts, dim=-1)
                current_state = full_window_tensor[:self.history_size, (self.full_num_features - self.full_num_observations):].clone()
            else:
                current_state = window_tensor[:self.history_size, (self.num_features-self.num_observations):].clone()
            ret.append(current_state)

        # Task Params
        if self.task_condition:
            task_params = torch.from_numpy(features["task_params"]).float()
            task_params = self._normalize("task_params", task_params)
            if self.add_goal_noise:
                task_params = self._add_obs_noise(task_params, "task_params")
            ret.append(task_params)

        # Style Condition
        if self.style_condition and "style_onehot" in raw_traj:
            style_cond = torch.from_numpy(np.asarray(raw_traj["style_onehot"]).astype(np.float32))
            ret.append(style_cond)

        ret.append(anchor)
        return tuple(ret)


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
                    tp, _ = self._compute_task_params(raw_traj['base'], raw_traj['obj'], start_idx=read_start)
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
    
    def _mirror_raw_features(self, features):
        """
        Apply mirror symmetry to raw feature arrays (in-place modification).
        Mirroring happens BEFORE normalization to avoid statistical bias from 
        normalized distributions of swapped antisymmetric joints.
        """
        # 1. Joints: Swap L/R and Negate Roll/Yaw
        if "joints" in features:
            perm_indices = [
                 6,  7,  8,  9, 10, 11,      # 0-5
                 0,  1,  2,  3,  4,  5,      # 6-11
                12, 13, 14,                  # 12-14
                22, 23, 24, 25, 26, 27, 28,  # 15-21
                15, 16, 17, 18, 19, 20, 21   # 22-28
            ]
            flip_vec = [
                1, -1, -1, 1, 1, -1,      # 0-5
                1, -1, -1, 1, 1, -1,      # 6-11
               -1, -1,  1,                # 12-14
                1, -1, -1, 1, -1, 1, -1,  # 15-21
                1, -1, -1, 1, -1, 1, -1   # 22-28
            ]
            
            # Apply permutation (Advanced indexing returns a copy)
            joints = features["joints"]
            joints = joints[..., perm_indices]
            
            # Apply sign
            sign = np.array(flip_vec, dtype=joints.dtype)
            joints = joints * sign
            
            features["joints"] = joints

        # 2. Simple flips (Negate Y or specific indices)
        # Lists of (key, indices_to_negate)
        # task_params sign flip depends on task type:
        #   pickplace: negate index 1 (y-direction)
        #   reorient:  negate indices 1,2 (delta_roll, delta_yaw both flip under L/R mirror)
        tp_negate = [1, 2] if self.task_type == "reorient" else [1]
        negate_tasks = [
            ("delta_xy", [1]),
            ("delta_yaw", [0]),
            ("obj_delta_xy", [1]),
            ("obj_rel_pos", [1]),
            ("task_params", tp_negate),
            ("body_rot6d", [1, 3, 5]),
            # ("obj_rel_rot6d", [1, 3, 5]),  # Handled separately below with geom offset
        ]
        
        for key, indices in negate_tasks:
            if key in features:
                # In-place negation
                # Ensure we have a copy first if it might be a view? 
                # features are usually copies from compute_sbto_components.
                features[key][..., indices] *= -1.0

        # 3. Object Orientation Mirroring (with Geom Offset Correction)
        # The object BODY frame is mirrored, but we must account for the fixed GEOM offset
        # so that the VISUAL object is mirrored correctly.
        # R_new = M @ (R_rel @ R_off) @ M @ R_off.T
        if "obj_rel_rot6d" in features:
            # Hardcoded offset from unitree_g1/mj_model.xml
            # q_off = [0.00991298, 0.849052, -0.523456, 0.0707591] (w, x, y, z)
            q_off = np.array([0.00991298, 0.849052, -0.523456, 0.0707591], dtype=np.float64)
            
            # Get R_off (3x3)
            # quat_to_rot expects (..., 4)
            R_off = quat_to_rot(q_off[None, :])[0] # (3, 3)
            R_off_T = R_off.T
            
            # Mirror Matrix M = diag(1, -1, 1)
            M = np.diag([1, -1, 1]).astype(np.float64)
            
            # Convert obj_rel_rot6d to R_rel
            obj_rot6d = features["obj_rel_rot6d"] # (..., 6)
            input_shape = obj_rot6d.shape
            obj_rot6d_flat = obj_rot6d.reshape(-1, 6)
            R_rel = rot6d_to_rot(obj_rot6d_flat) # (N, 3, 3)
            
            # Compute R_new
            # R_vis = R_rel @ R_off
            R_vis = np.matmul(R_rel, R_off) 
            # R_vis_mirr = M @ R_vis @ M
            R_vis_mirr = np.matmul(M, np.matmul(R_vis, M))
            
            # R_new = R_vis_mirr @ R_off.T
            R_new = np.matmul(R_vis_mirr, R_off_T)
            
            # Convert back to 6d
            obj_rot6d_new = rot_to_6d(R_new)
            features["obj_rel_rot6d"] = obj_rot6d_new.reshape(input_shape).astype(obj_rot6d.dtype)
