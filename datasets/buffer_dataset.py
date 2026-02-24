import numpy as np
import torch
import os
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.flexible_dataset import FlexibleWindowDataset


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
                 feature_order=None,
                 norm_path=None,
                 calculate_stats=True,
                 noise_cfg=None,
                 add_noise=False,
                 add_goal_noise=False):

        # Store raw buffer before any init
        self._raw_buffer = data_buffer

        # ---- Config (mirrors FlexibleWindowDataset.__init__) ----
        self.data_root = None
        self.noise_cfg = noise_cfg or {}
        self.num_observations = config.get("num_observations", 45)
        self.num_features = config.get("num_features", 48)
        self.history_size = config.get("state_history", 4)
        self.window_size = config.get("num_timesteps", 50) // config.get("downsample", 1)
        self.stride = config.get("stride", 1)
        self.downsample = config.get("downsample", 1)
        self.start_timestep = config.get("start_timestep", 0)
        self.normalize_goal_vec = config.get("normalize_goal_vec", True)
        self.num_task_params = config.get("num_task_params", 2)
        self.normalization_type = config.get("normalization_type", "mean_std")
        self.key_mapping = config.get("key_mapping", {})

        self.add_noise = add_noise
        self.add_goal_noise = add_goal_noise

        self.feature_order = feature_order or [
            "delta_xy", "delta_yaw", "obj_delta_xy",
            "joints", "body_z", "body_rot6d",
            "obj_rel_pos", "obj_rel_rot6d",
        ]

        # No file paths for buffer dataset
        self.file_paths = []

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

            # ---- External task_params (per-trajectory, no time dim) ----
            if 'task_params' in keys:
                tp = np.asarray(std_buf['task_params'][i])
                processed['task_params'] = tp  # shape (D,) — not batched over time

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
            if k == 'task_params':
                # Non-temporal — pass through as-is
                result[k] = v
            else:
                result[k] = pad_kwargs(v[batch_idx])
        return result

    # ------------------------------------------------------------------
    # Override _compute_transform to inject external task_params
    # ------------------------------------------------------------------
    def _compute_transform(self, raw_traj, t_start):
        """Call parent transform, then override task_params if provided externally."""
        feats, anchor = super()._compute_transform(raw_traj, t_start)

        if 'task_params' in raw_traj:
            feats['task_params'] = np.asarray(raw_traj['task_params'])

        return feats, anchor