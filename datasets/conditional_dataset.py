import numpy as np
import torch
import os
from torch.utils.data import Dataset


# ================================
#         BASE DATASET
# ================================
class ConditionalStateDataset(Dataset):
    """
    Base class for state-action dataset.
    Supports memory-mapped loading to handle large datasets without OOM.
    """

    def __init__(self, dataset_path, config, state_condition=False, history_condition=False, 
                 task_condition=False, action_condition=False, load_norm=False, norm_path=None):

        # Load data internally
        self.dataset_path = dataset_path
        data = {}
        
        # Handle directory-based dataset (npy + mmap)
        if os.path.isdir(self.dataset_path) or (self.dataset_path.endswith('.npz') and os.path.isdir(self.dataset_path[:-4])):
            real_path = self.dataset_path if os.path.isdir(self.dataset_path) else self.dataset_path[:-4]
            print(f"Loading mmapped data from directory: {real_path}...\n")
            
            for key in ["history", "future", "goal", "current_state", "extra_goals"]:
                p = os.path.join(real_path, f"{key}.npy")
                if os.path.exists(p):
                    data[key] = np.load(p, mmap_mode='r')
        else:
            # Fallback to single file npz
            print(f"Loading data from file: {self.dataset_path}")
            data = np.load(self.dataset_path, allow_pickle=True, mmap_mode='r')

        # Helper for dictionary or NpzFile access
        def get_data(key, default=None):
            if key in data:
                return data[key]
            return default

        self.data_future = get_data("future")
        ref_shape = self.data_future.shape if self.data_future is not None else None
        
        if ref_shape is None:
             raise ValueError("Dataset must contain 'future' key")

        self.num_traj = 1 if config.get("overfit") else ref_shape[0]
        self.start_timestep = config.get("start_timestep", 0)
        self.num_timesteps = config.get("num_timesteps", ref_shape[1])
        self.history_size = config.get("state_history", 1)
        self.hp_overlap = config.get("hp_overlap", 0)
        self.num_features = config.get("num_features", ref_shape[2])
        self.num_observations = config.get("num_observations", self.num_features)   
        self.num_goal_features = config.get("num_task_params", 10)
        self.start_feature = config.get("start_feature", 0)

        # Store raw data (numpy arrays or memmaps)
        self.data_history = get_data("history")
        self.data_current_state = get_data("current_state")
        self.data_extra_goals = get_data("extra_goals")

        # Load SBTO specific keys
        self.data_goal = get_data("goal")
        
        self.state_condition = state_condition
        self.task_condition = task_condition

        # Paths
        self.load_norm = load_norm
        self.norm_path = norm_path
        
        # --- Pre-calculate Normalization Stats ---
        # We calculate min/max once, but do NOT apply it to the whole dataset yet.
        self.stats = {}
        
        if self.data_future is not None:            
            self._init_stats("future", self.data_future, ["min_future", "max_future"])

        if self.data_current_state is not None and self.state_condition:
            self._init_stats("current_state", self.data_current_state, ["min_current_state", "max_current_state"])  
        
        if self.data_goal is not None and self.task_condition:
            self._init_stats("goal", self.data_goal[:, :self.num_goal_features], ["min_goal", "max_goal"])

    def _init_stats(self, name, data_arr, keys):
        """Calculate or load normalization stats without loading full data."""
        print(f"Initializing stats for {name}: shape {data_arr.shape}...")
        min_key, max_key = keys
        
        # Try loading first
        if self.load_norm and self.norm_path and os.path.exists(self.norm_path):
            try:
                stats = np.load(self.norm_path)
                if min_key in stats and max_key in stats:
                    self.stats[min_key] = torch.tensor(stats[min_key], dtype=torch.float32)
                    self.stats[max_key] = torch.tensor(stats[max_key], dtype=torch.float32)
                    return
            except Exception:
                pass # Fallback to calculation

        # Calculate min/max over dataset        
        dims = (0, 1) if data_arr.ndim == 3 else (0,)
        
        # Compute
        # NOTE: If this is too slow/OOM, you must provide a norm_path with pre-computed stats!
        min_val = np.amin(data_arr, axis=dims)
        max_val = np.amax(data_arr, axis=dims)
        
        # Store as tensors
        self.stats[min_key] = torch.from_numpy(min_val).float()
        self.stats[max_key] = torch.from_numpy(max_val).float()

        # Save if path provided
        if self.norm_path:
            os.makedirs(os.path.dirname(self.norm_path), exist_ok=True)
            data_to_save = {}
            if os.path.exists(self.norm_path):
                try:
                    with np.load(self.norm_path) as f:
                        data_to_save = dict(f)
                except: pass
            
            data_to_save[min_key] = min_val
            data_to_save[max_key] = max_val
            np.savez(self.norm_path, **data_to_save)

    # ------------------------
    # Normalization Helper
    # ------------------------
    def _normalize_tensor(self, tensor, min_key, max_key, start_idx, end_idx):
        if min_key not in self.stats or max_key not in self.stats:
            return tensor
        
        min_val = self.stats[min_key][start_idx:end_idx].to(tensor.device)
        max_val = self.stats[max_key][start_idx:end_idx].to(tensor.device)
        
        # Handle broadcasting
        # tensor: (T, C) or (C,)
        # stats: (C,)
        
        return 2 * (tensor - min_val) / (max_val - min_val + 1e-6) - 1

    # ------------------------
    # Dataset API
    # ------------------------
    def __len__(self):
        return self.num_traj

    def __getitem__(self, idx):
        real_idx = idx if self.num_traj > 1 else 0
    
        ret = []
        
        # 1. State (Target)
        if self.data_future is not None:
            # Slice: (T, C)
            x_np = self.data_future[real_idx, :self.num_timesteps, :self.num_features]
            x_t = torch.from_numpy(x_np.copy()).float()
            x_t = self._normalize_tensor(x_t, "min_future", "max_future", 0, self.num_features)
            ret.append(x_t.permute(1, 0)) # (C, T)

        # 2. State condition (Current State)
        if self.data_current_state is not None and self.state_condition:
            current_np = self.data_current_state[real_idx, :self.num_observations]
            current_t = torch.from_numpy(current_np.copy()).float()
            
            # Add extra goals if available (broadcasting/augmentation)
            if self.data_extra_goals is not None:
                extra_goals_t = torch.from_numpy(self.data_extra_goals[real_idx].copy()).float()
                num_goal_choices = extra_goals_t.shape[0] // 4
                selected_goal_idx = np.random.randint(0, num_goal_choices + 1)
                
                # Check limits
                if selected_goal_idx < num_goal_choices:
                    # Specific to 86:89 range mentioned previously
                    start_idx = 86
                    end_idx = 89
                    if current_t.shape[0] >= end_idx:
                         current_t[start_idx:end_idx] = extra_goals_t[selected_goal_idx * 4: selected_goal_idx * 4 + 3]

            current_t = self._normalize_tensor(current_t, "min_current_state", "max_current_state", 0, self.num_observations)
            ret.append(current_t) # (C,)

        # 3. Task Condition (Goal)
        if self.task_condition and self.data_goal is not None:
            goal_np = self.data_goal[real_idx, :self.num_goal_features]
            goal_t = torch.from_numpy(goal_np.copy()).float()
            goal_t = self._normalize_tensor(goal_t, "min_goal", "max_goal", 0, self.num_goal_features)
            ret.append(goal_t) # (C,)

        return tuple(ret)

def denormalize(x, min_val, max_val):
    return ((x + 1) / 2) * (max_val - min_val) + min_val    

def normalize(arr, max_val, min_val):
    """Min-max normalize to [-1, 1]."""
    return (2 * (arr - min_val) / (max_val - min_val + 1e-6) - 1)