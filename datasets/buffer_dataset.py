import numpy as np
import torch
import os
from torch.utils.data import Dataset
from tqdm import tqdm

from .flexible_dataset import FlexibleWindowDataset
from mjlab.scripts.diffusion_planner.utils.math.sbto_utils import compute_task_params
from mjlab.scripts.diffusion_planner.utils.math.math_tools import yaw_from_quat, yaw_to_rot_matrix

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
        feature_order=None, 
        norm_path=None, 
        calculate_stats=True, 
        noise_cfg=None, 
        add_noise=False,
        add_goal_noise=False
    ):
        
        super().__init__(
            data_root=None,  # No files, data is in buffer # type: ignore
            config=config,
            feature_order=feature_order,
            norm_path=norm_path,
            calculate_stats=calculate_stats,
            noise_cfg=noise_cfg,
            add_noise=add_noise,
            add_goal_noise=add_goal_noise,
        )
        
        self._raw_buffer = data_buffer
        self._task_params = task_params

        # No file paths for buffer dataset

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

    # ------------------------------------------------------------------
    # Buffer helpers
    # ------------------------------------------------------------------
    def get_k(self, key):
        return self.key_mapping.get(key, key)

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
            for k in ['body_pos_w', 'base_xyz_quat', 'joint_pos', 'actuator_pos']:
                if k in buf:
                    v = buf[k]
                    if isinstance(v, (list, tuple)):
                        num_trajs = len(v)
                    elif isinstance(v, np.ndarray) and v.ndim >= 3:
                        num_trajs = v.shape[0]
                    break

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

    def _preload_from_buffer(self):
        """
        Process data buffer into ram_cache format identical to FlexibleWindowDataset.
        Each trajectory becomes a separate entry in ram_cache with shape (1, T, D).
        """
        std_buf = self._standardize_buffer()
        if not std_buf:
            return

        rep_key = list(std_buf.keys())[0]
        num_trajs = len(std_buf[rep_key])
        keys = set(std_buf.keys())

        for i in range(num_trajs):
            processed = {}

            # ---- RL Rollout schema ----
            if 'body_pos_w' in keys or self.get_k('body_pos_w') in keys:
                pos_key = 'body_pos_w' if 'body_pos_w' in keys else self.get_k('body_pos_w')
                quat_key = 'body_quat_w' if 'body_quat_w' in keys else self.get_k('body_quat_w')
                joint_key = 'joint_pos' if 'joint_pos' in keys else self.get_k('joint_pos')
                obj_pos_key = 'object_pos_w' if 'object_pos_w' in keys else self.get_k('object_pos_w')
                obj_quat_key = 'object_quat_w' if 'object_quat_w' in keys else self.get_k('object_quat_w')

                bp = np.asarray(std_buf[pos_key][i])[::self.downsample]
                bq = np.asarray(std_buf[quat_key][i])[::self.downsample]
                if bp.ndim == 3:
                    bp = bp[:, 0, :]
                if bq.ndim == 3:
                    bq = bq[:, 0, :]

                processed['base'] = np.concatenate([bp, bq], axis=-1)[np.newaxis]
                processed['joints'] = np.asarray(std_buf[joint_key][i])[::self.downsample][np.newaxis]

                op = np.asarray(std_buf[obj_pos_key][i])[::self.downsample]
                oq = np.asarray(std_buf[obj_quat_key][i])[::self.downsample]
                processed['obj'] = np.concatenate([op, oq], axis=-1)[np.newaxis]

                if 'ee_rel_pos' in keys:
                    processed['ee_rel_pos'] = np.asarray(std_buf['ee_rel_pos'][i])[::self.downsample][np.newaxis]

            # ---- SBTO schema ----
            elif 'base_xyz_quat' in keys:
                processed['base'] = np.asarray(std_buf['base_xyz_quat'][i])[::self.downsample][np.newaxis]
                processed['joints'] = np.asarray(std_buf['actuator_pos'][i])[::self.downsample][np.newaxis]
                if 'obj_0_xyz_quat' in keys:
                    processed['obj'] = np.asarray(std_buf['obj_0_xyz_quat'][i])[::self.downsample][np.newaxis]

                if 'base_linvel_angvel' in keys:
                    processed['base_vel'] = np.asarray(std_buf['base_linvel_angvel'][i])[::self.downsample][np.newaxis]
                if 'actuator_vel' in keys:
                    processed['joints_vel'] = np.asarray(std_buf['actuator_vel'][i])[::self.downsample][np.newaxis]
                if 'obj_0_linvel_angvel' in keys:
                    processed['obj_vel'] = np.asarray(std_buf['obj_0_linvel_angvel'][i])[::self.downsample][np.newaxis]

                if 'ee_rel_pos' in keys:
                    processed['ee_rel_pos'] = np.asarray(std_buf['ee_rel_pos'][i])[::self.downsample][np.newaxis]
            else:
                raise ValueError(f"Unrecognized buffer schema. Keys: {keys}")

            # ---- External goal (per-trajectory, no time dim) ----
            # The pipeline passes the desired final box position expressed in the
            # coordinate frame of the initial robot pelvis.  Convert to world frame
            # so every window can call compute_task_params(ref, obj, goal_world).
            if self._task_params is not None:
                goal_local = np.asarray(self._task_params[i], dtype=np.float64)

                # Initial pelvis pose & initial object pos (first real frame)
                init_base = processed['base'][0, 0, :]   # (7,) [x,y,z,w,x,y,z]
                init_obj  = processed['obj'][0, 0, :3]    # (3,)

                init_yaw = yaw_from_quat(init_base[3:7])
                R_local_to_world = yaw_to_rot_matrix(init_yaw)

                goal_3d = np.zeros(3, dtype=np.float64)
                n = min(len(goal_local), 3)
                goal_3d[:n] = goal_local[:n]

                goal_world = (R_local_to_world @ goal_3d[:, None])[:, 0] + init_obj[:3]
                processed['goal_obj_world'] = goal_world  # (3,) world frame

            self.ram_cache.append(processed)

        print(f"Loaded {num_trajs} trajectories from buffer into RAM cache.")

    # ------------------------------------------------------------------
    # Indexing  (mirrors FlexibleWindowDataset._index_dataset)
    # ------------------------------------------------------------------
    def _index_dataset(self):
        """Build sliding-window indices from ram_cache (same padding logic as parent)."""
        print("Indexing buffer dataset...")
        self.indices = []
        self.traj_lengths = []

        for file_idx, file_data in enumerate(self.ram_cache):
            if 'base' not in file_data:
                continue

            # file_data['base'] shape is (1, T_down, D)
            T_down = file_data['base'].shape[1]

            pad_start = self.history_size
            pad_end = self.window_size - (T_down + pad_start) % self.window_size + self.history_size
            T_padded = T_down + pad_start + pad_end
            self.traj_lengths.append(T_padded)

            w_size = self.window_size + self.history_size
            max_start = T_padded - w_size

            if max_start >= 0:
                starts = np.arange(0, max_start + 1, self.stride)
                for s in starts:
                    # batch_idx = 0 because each ram_cache entry has N=1
                    self.indices.append((file_idx, 0, int(s)))

        print(f"Indexed {len(self.indices)} windows from buffer.")

    # ------------------------------------------------------------------
    # Override _get_single_traj to pass through non-temporal keys
    # ------------------------------------------------------------------
    def _get_single_traj(self, file_idx, batch_idx):
        """Get padded (T, D) dict. Passes through non-temporal keys like task_params."""
        file_data = self.ram_cache[file_idx]

        pad_start = self.history_size
        pad_end = self.traj_lengths[file_idx] - (file_data["base"][batch_idx].shape[0] + pad_start)

        pad_kwargs = lambda arr: np.pad(arr, ((pad_start, pad_end), (0, 0)) if arr.ndim == 2 else ((0, 0),), mode='edge')

        result = {}
        for k, v in file_data.items():
            if k in ('task_params', 'goal_obj_world'):
                # Non-temporal — pass through as-is
                result[k] = v
            else:
                result[k] = pad_kwargs(v[batch_idx])
        return result

    # ------------------------------------------------------------------
    # Override _compute_task_params so stats use compute_task_params
    # ------------------------------------------------------------------
    def _compute_task_params(self, base_traj, obj_traj, start_idx=0):
        """Delegate to compute_task_params from sbto_utils for consistency."""
        return compute_task_params(
            current_robot_state=base_traj[start_idx],
            current_obj_state=obj_traj[start_idx],
            desired_obj_pos=obj_traj[-1, :3],
            normalize_goal_vec=self.normalize_goal_vec,
            num_task_params=self.num_task_params,
        )

    # ------------------------------------------------------------------
    # Override _compute_transform to use world-frame goal per window
    # ------------------------------------------------------------------
    def _compute_transform(self, raw_traj, t_start):
        """Call parent transform, then recompute task_params with external goal if available."""
        feats, anchor = super()._compute_transform(raw_traj, t_start)

        if 'goal_obj_world' in raw_traj:
            goal_world = raw_traj['goal_obj_world']
            read_start = max(0, t_start)

            feats['task_params'] = compute_task_params(
                current_robot_state=raw_traj['base'][read_start],
                current_obj_state=raw_traj['obj'][read_start],
                desired_obj_pos=goal_world,
                normalize_goal_vec=self.normalize_goal_vec,
                num_task_params=self.num_task_params,
            )
            anchor['final_obj_pos'] = goal_world

        return feats, anchor