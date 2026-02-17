import torch
import numpy as np
import yaml
import os
import time
from typing import List, Dict, Union, Optional
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler
from torch import nn, optim
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from config.configure import load_config, get_data_path, get_save_path, get_norm_path
from models.model import RobotDiffuser
from datasets.buffer_dataset import BufferDataset
from utils.math.sbto_utils import reconstruct_sbto_trajectory
from utils.math.math_tools import yaw_from_quat, yaw_to_rot_matrix
from utils.math.rotation_conversions import quaternion_to_matrix, matrix_to_quaternion


def compute_dataset_weights(dataset, sigma=0.2):
    """
    Computes weights for each sample in the dataset based on the inverse density 
    of its task parameters.
    """
    if not hasattr(dataset, 'indices') or not hasattr(dataset, '_get_single_traj'):
        print("Dataset does not support task density balancing (missing indices).")
        return None

    print("Computing task parameter density weights...")
    
    # 1. Collect Task Params
    # We ideally want unique task params based on trajectory to speed up
    # mapping: (file_idx, batch_idx) -> task_param
    
    unique_traj_keys = set((f, b) for f, b, t in dataset.indices)
    print(f"Found {len(unique_traj_keys)} unique trajectories in {len(dataset)} windows.")
    
    min_tp = dataset.stats.get('min_task_params')
    max_tp = dataset.stats.get('max_task_params')
    
    if min_tp is None:
        print("Warning: Task params stats not found. Density balancing might be inaccurate.")
    
    tp_list = []
    key_list = []
    
    # Pre-fetch if possible to avoid redundant file reads
    # But _get_single_traj uses ram_cache so it's fast
    
    for f, b in tqdm(unique_traj_keys, desc="Extracting Task Params"):
        raw = dataset._get_single_traj(f, b)
        
        # BufferDataset check: if 'obj' is missing
        if 'obj' not in raw and 'task_params' in raw:
             tp_raw = raw['task_params']
        elif 'obj' in raw:
             tp_raw = dataset._compute_task_params(raw['obj'])
        else:
             print(f"Warning: No object data or task params found for {f},{b}")
             continue
        
        # Normalize manually
        if min_tp is not None:
             # Manually normalize: (x - min) / (max - min) * 2 - 1
             # Ensure tensors
             if not torch.is_tensor(tp_raw):
                 tp_raw = torch.tensor(tp_raw, dtype=torch.float32)
             
             mn = min_tp.to(tp_raw.device)
             mx = max_tp.to(tp_raw.device)
             tp_norm = (tp_raw - mn) / (mx - mn + 1e-6) * 2 - 1
        else:
             tp_norm = torch.tensor(tp_raw, dtype=torch.float32)
            
        tp_list.append(tp_norm)
        key_list.append((f, b))
        
    if not tp_list: 
         return None
         
    # Stack (N_traj, D)
    tp_tensor = torch.stack(tp_list).float()
    
    # 2. Compute Density (Simple KDE on CPU)
    # Using Gaussian Kernel: sum(exp(-dist^2 / sigma^2))
    print(f"Computing density (Sigma={sigma})...")
    
    # Pairwise distance matrix (N, N)
    dists = torch.cdist(tp_tensor, tp_tensor) 
    
    # Density ~ sum(kernel)
    # Adding epsilon to avoid division by zero
    density = torch.sum(torch.exp(-(dists ** 2) / (sigma ** 2)), dim=1) + 1e-6
    
    # Weight = 1 / density
    weights_traj = 1.0 / density
    
    # Normalize weights so sum matches length or similar (optional, mainly relative matters)
    # Let's normalize so mean is 1
    weights_traj = weights_traj / weights_traj.mean()
    
    # Map back to dataset indices
    key_to_weight = {k: w.item() for k, w in zip(key_list, weights_traj)}
    
    sample_weights = []
    for f, b, t in dataset.indices:
        if (f, b) in key_to_weight:
             sample_weights.append(key_to_weight[(f, b)])
        else:
             sample_weights.append(1.0) # Default
        
    return torch.tensor(sample_weights).double()


def apply_condition_dropout(cond, dropout_prob: float):
    if cond is None or dropout_prob <= 0.0:
        return cond
    if not torch.is_tensor(cond):
        cond = torch.as_tensor(cond)
    if torch.rand(1).item() < dropout_prob:
        return torch.zeros_like(cond)
    return cond

class MotionGenerator:
    def __init__(self, config_path: str = "config/config.yaml", device: str = None):
        self.config_path = config_path
        
        # Load Config
        with open(config_path, 'r') as file:
            raw_config = yaml.safe_load(file)
        
        self.model_cfg, self.data_cfg, self.training_cfg, self.noise_cfg = load_config(
            config_path, raw_config.get("auto_conf", False)
        )
        
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Initialize Model
        self.diffuser = RobotDiffuser(
            model_config=self.model_cfg, 
            data_config=self.data_cfg,
            training_config=self.training_cfg, 
            noise_scheduler_config=self.noise_cfg,
            mode='inference', # Default to inference, but 'train' or 'inference' just sets up scheduler mainly
            device=self.device
        )
        
        self.dataset = None # To be initialized in fit or loaded

        # Try to load existing stats if available to be ready for inference
        self._setup_dataset_structure(load_stats=True)


    def _setup_dataset_structure(self, load_stats=True):
        """Helper to init a dummy dataset just for stats/normalization logic."""
        norm_path = get_norm_path(self.model_cfg, self.training_cfg, self.data_cfg)
        
        # If we have stats, we can init a dataset without data just for utils
        if load_stats and norm_path and os.path.exists(norm_path):
             # We create a dummy dataset with empty buffer just to load stats
             dummy_buffer = {'body_pos_w': np.zeros((1, 1, 1, 3))}
             self.dataset = BufferDataset(
                data_buffer=dummy_buffer, 
                config=self.data_cfg, 
                calculate_stats=False, 
                norm_path=norm_path,
                noise_cfg={}
            )

    def fit(self, 
            data_source: Union[str, List[Dict]], 
            task_params: Optional[List[Dict]] = None,
            epochs: int = None, 
            save_path: str = None,
            checkpoint: Optional[str] = None):
        """
        Train the model.
        
        Args:
            data_source: Path to data folder (str) or list of trajectory dicts.
            task_params: Optional list of task parameters (e.g. goals) aligned with data_source.
            epochs: Number of epochs to train. Overrides config if provided.
            save_path: Path to save model checkpoints. Overrides config if provided.
            checkpoint: Optional path to a checkpoint to resume from.
        """
        
        if epochs is None:
            epochs = self.training_cfg.get("num_epochs", 100)
        
        # Setup Data
        norm_path = get_norm_path(self.model_cfg, self.training_cfg, self.data_cfg)

        if isinstance(data_source, str):
            raise ValueError("MotionGenerator.fit no longer supports file paths. Please load data into a buffer first.")
        
        self.dataset = BufferDataset(
            data_buffer=data_source,
            config=self.data_cfg, 
            calculate_stats=True, 
            norm_path=norm_path, # will overwrite if provided
            noise_cfg=self.training_cfg.get("state_conditioning_noise_level", {}),
            add_noise=self.training_cfg.get("add_obs_noise", False), 
            add_goal_noise=self.training_cfg.get("add_goal_noise", False)
        )
        # Task Density Balancing
        sampler = None
        shuffle = True
        
        if self.training_cfg.get("balance_task_density", True):
            weights = compute_dataset_weights(self.dataset, sigma=self.training_cfg.get("density_sigma", 0.2))
            if weights is not None:
                sampler = WeightedRandomSampler(weights, len(weights))
                shuffle = False
                print("WeightedRandomSampler activated (Balance Task Density).")

        # Create DataLoader
        train_dataloader = DataLoader(
            self.dataset, 
            batch_size=self.data_cfg["batch_size"], 
            num_workers=4,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=True
        )

        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(
            self.diffuser.model.parameters(), 
            lr=self.training_cfg.get("learning_rate", 1e-4),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        scaler = GradScaler()

        if checkpoint:
            if os.path.exists(checkpoint):
                print(f"Loading diffusion weights from {checkpoint}...")
                ckpt = torch.load(checkpoint, map_location=self.device)
                self.diffuser.model.load_state_dict(ckpt["model"])
                if "optimizer" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer"])
                if "scheduler" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler"])
                if "scaler" in ckpt:
                    scaler.load_state_dict(ckpt["scaler"])
            else:
                print(f"Checkpoint {checkpoint} not found. Starting from scratch.")
        
        self.diffuser.model.train()
        
        state_condition = self.model_cfg.get("state_condition", False)
        task_condition = self.model_cfg.get("task_condition", False)
        dropout_probs = self.training_cfg.get("condition_dropout_prob", {})
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_losses = []
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for step, batch in enumerate(pbar):
                
                if batch[0].shape[0] < self.data_cfg["batch_size"]:
                    # Pad the batch by repeating elements if it's smaller than batch_size
                    current_bs = batch[0].shape[0]
                    target_bs = self.data_cfg["batch_size"]
                    repeats = (target_bs + current_bs - 1) // current_bs
                    
                    new_batch = []
                    for item in batch:
                        if isinstance(item, torch.Tensor):
                            dims = [1] * item.dim()
                            dims[0] = repeats
                            new_batch.append(item.repeat(*dims)[:target_bs])
                    else:
                            new_batch.append(item)
                    batch = new_batch
            
                batch_data = list(batch)
                prediction_target = batch_data[0].to(self.device)
                
                state_cond = None
                task_cond = None
                idx = 1
                
                if state_condition:
                    state_cond = batch_data[idx].to(self.device)
                    state_cond = apply_condition_dropout(state_cond, dropout_probs.get("state", 0.0))
                    idx += 1
                
                if task_condition:
                    task_cond = batch_data[idx].to(self.device)
                    task_cond = apply_condition_dropout(task_cond, dropout_probs.get("task", 0.0))
                    idx += 1
                
                # Construct cond
                cond = []
                if state_cond is not None: cond.append(state_cond)
                if task_cond is not None: cond.append(task_cond)
                
                model_cond = tuple(cond) if len(cond) > 0 else None
                if len(cond) == 1: model_cond = cond[0]
                
                bs, ts, _ = prediction_target.shape
                timesteps = torch.randint(
                    0, self.diffuser.noise_scheduler.config.num_train_timesteps, (bs, ts,), device=self.device
                ).long()
                
                with torch.autocast(device_type="cuda" if self.device.type=="cuda" else "cpu", dtype=torch.float32):
                    diff_output = self.diffuser.model(
                        prediction_target, 
                        model_cond, 
                        timesteps=timesteps, 
                    )
                    pred_loss = diff_output["loss"]
                    loss = pred_loss.mean()

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.diffuser.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                epoch_losses.append(loss.item())
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            scheduler.step()
            mean_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1} Mean Loss: {mean_loss:.5f}")
            
            # Save Checkpoint
            if (epoch + 1) % self.training_cfg.get("save_every", 50) == 0 or epoch == epochs - 1:
                if save_path:
                    # Check if save_path serves as a directory or a specific file prefix
                    is_dir_like = not (save_path.endswith('.pt') or save_path.endswith('.pth'))
                    
                    if is_dir_like:
                        os.makedirs(save_path, exist_ok=True)
                        fpath = os.path.join(save_path, f"model_{epoch+1}.pth")
                    else:
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        base, ext = os.path.splitext(save_path)
                        fpath = f"{base}_{epoch+1}{ext}"
                else:
                    # Default save
                    save_dir = get_save_path(self.model_cfg, self.data_cfg, self.training_cfg)
                    os.makedirs(save_dir, exist_ok=True)
                    fpath = os.path.join(save_dir, f"model_{epoch}.pth")
               
                checkpoint = {
                    "model": self.diffuser.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "scheduler": scheduler.state_dict()
                }
                torch.save(checkpoint, fpath)

    def _interpolate_trajectory(self, trajectory):
        """
        Interpolate trajectory if downsample > 1.
        trajectory: (B, T, D)
        """
        # If dataset not initialized, can't check downsample, but fit() or _setup calls it.
        if self.dataset is None:
             return trajectory

        k = self.dataset.downsample
        if k <= 1:
            return trajectory

        # print(f"Interpolating trajectory with downsample factor {k}...")
        B, T, D = trajectory.shape
        # Original knots at 0, k, 2k, ...
        # If we have T knots, the duration is roughly (T-1)*k. 
        # We target a full reconstruction of length T*k to ensure we recover the trailing frames 
        # (which are usually lost in integer division downsampling).
        original_times = np.arange(T) * k
        target_length = T * k + (k - 1)
        target_times = np.arange(target_length)
        
        new_T = len(target_times)
        interpolated = np.zeros((B, new_T, D))

        # Indices for Slerp
        # Robot Quat: 3:7 (indices 3,4,5,6)
        # Object Quat: 39:43 (indices 39,40,41,42)
        quat_indices = [slice(3, 7), slice(39, 43)]
        
        for b in range(B):
            # 1. Linear Interpolation for all dims
            # Use fill_value to extend last pose to the end of the bin (nearest neighbor for tail)
            f = interp1d(original_times, trajectory[b], axis=0, kind='linear', 
                         fill_value=(trajectory[b][0], trajectory[b][-1]), bounds_error=False)
            interpolated[b] = f(target_times)
            
            # 2. Slerp for Rotation dims
            for sl in quat_indices:
                # Need consistent quaternion sign for Slerp?
                # R.from_quat usually handles, but let's be safe.
                # Only if the quat slice is valid
                if sl.stop <= D:
                     q_vals = trajectory[b, :, sl]
                     rot = R.from_quat(q_vals)
                     slerp = Slerp(original_times, rot)
                     
                     # Clamp times for Slerp to avoid extrapolation error (equivalent to nearest)
                     clamped_times = np.clip(target_times, original_times[0], original_times[-1])
                     
                     interp_q = slerp(clamped_times).as_quat()
                     interpolated[b, :, sl] = interp_q
        
        return interpolated

    def _update_condition(self, robot_world_history, obj_world_history, goal_condition=None):
        """
        Update condition for next autoregressive step.
        """
        if self.dataset is None:
             raise RuntimeError("Dataset not initialized. Call fit() or manually init dataset.")

        B, H, _ = robot_world_history.shape
        next_states = []
        next_anchors = {'ref_pos': [], 'ref_quat': []}

        for b in range(B):
            r_slice = robot_world_history[b]
            o_slice = obj_world_history[b]
            
            raw_chunk = {
                'base': r_slice[:, :7],       
                'joints': r_slice[:, 7:36],
                'obj': o_slice[:, :7]
            }
            
            # Helper to extract batch goal
            tp = None
            if goal_condition is not None:
                if isinstance(goal_condition, (np.ndarray, torch.Tensor)) and len(goal_condition) == B:
                    tp = goal_condition[b]
                else: 
                     # Fallback/Broadcast or dict
                     tp = goal_condition

            feats, new_anch = self.dataset._compute_transform(raw_chunk, t_start=0, task_params=tp)
            
            current_parts = []
            obs_start_idx = self.dataset.num_features - self.dataset.num_observations
            cumulative_dim = 0
            
            for key in self.dataset.feature_order:
                if key in feats:
                    part = torch.from_numpy(feats[key]).float()
                    part = self.dataset._normalize(key, part) 
                    
                    part_dim = part.shape[-1]
                    part_end = cumulative_dim + part_dim
                    
                    if part_end > obs_start_idx:
                        local_start = max(0, obs_start_idx - cumulative_dim)
                        current_parts.append(part[:H, local_start:])
                    
                    cumulative_dim += part_dim
            
            c_state = torch.cat(current_parts, dim=-1)
            next_states.append(c_state)
            next_anchors['ref_pos'].append(new_anch['ref_pos'])
            next_anchors['ref_quat'].append(new_anch['ref_quat'])

        next_state_tens = torch.stack(next_states)
        batched_anchor = {
            'ref_pos': np.stack(next_anchors['ref_pos']),
            'ref_quat': np.stack(next_anchors['ref_quat']),
        }
        return next_state_tens, batched_anchor

    def generate_trajectory(self, 
                            initial_condition: Dict, 
                            goal_condition: Union[Dict, np.ndarray],
                            stitch_steps: int = 1, 
                            num_samples: int = 1):
        """
        Generate trajectory.
        
        Args:
            initial_condition: Dictionary with 'robot' (H, 36) and 'obj' (H, 7) history in world frame.
                               OR indices or whatever dataset expects.
                               However, standard inference logic assumes we start from normalized state tensor.
                               For flexibility, let's assume the user passes world-frame history, 
                               and we use dataset logic to normalize it.
            goal_condition: Goal parameters (e.g. final object pos).
            
        Returns:
            np.ndarray of shape (num_samples, T_total, D)
        """
        if self.dataset is None:
             raise RuntimeError("Dataset not initialized. Call fit() or setup.")

        self.diffuser.model.eval()
        
        # 1. Process Initial Condition
        # We assume initial_condition contains raw history to compute encoded state
        # robot: (H, 36), obj: (H, 7)
    
        r_hist = initial_condition['robot'] # (H, 36)
        o_hist = initial_condition['obj']   # (H, 7)
        
        # If single sample provided, add batch dim
        if r_hist.ndim == 2: r_hist = r_hist[None, ...]
        if o_hist.ndim == 2: o_hist = o_hist[None, ...]
        
        # We first need to "update_condition" to get features from raw history
        curr_state_tens, current_anchors = self._update_condition(r_hist, o_hist, goal_condition=None)
        
        # Now replicate for num_samples
        if curr_state_tens.shape[0] == 1 and num_samples > 1:
            curr_state_tens = curr_state_tens.repeat(num_samples, 1, 1).to(self.device)
        
        # 2. Process Goal (Absolute Frame, matching Training Pipeline)
        
        # Goal Parsing
        if isinstance(goal_condition, (np.ndarray, list)):
             gc = np.array(goal_condition) # (7,) or (1, 7) or (B, 7)
             if gc.ndim == 1: gc = gc[None, :]
             
             # If broadcasting needed
             if gc.shape[0] == 1 and curr_state_tens.shape[0] // num_samples > 1:
                 # Match original batch size B
                 B = curr_state_tens.shape[0] // num_samples
                 gc = np.repeat(gc, B, axis=0)

             task_params = torch.from_numpy(gc).float()

        else:
             # Assume dict, maybe extract?
             raise ValueError("Goal condition format not supported yet")
        
        # Normalize (Absolute) task params
        task_params = self.dataset._normalize("task_params", task_params) # (B, D)
        
        # Replicate for num_samples
        if task_params.shape[0] == 1 and num_samples > 1:
            task_params = task_params.repeat(num_samples, 1)
        
        task_tens = task_params.to(self.device)
        curr_state_tens = curr_state_tens.to(self.device)
        
        # Update anchors for reconstruction
        current_anchors['ref_pos'] = np.repeat(current_anchors['ref_pos'], num_samples, axis=0)
        current_anchors['ref_quat'] = np.repeat(current_anchors['ref_quat'], num_samples, axis=0)
        current_anchors['ref_obj_pos'] = np.repeat(current_anchors['ref_obj_pos'], num_samples, axis=0)
        
        history_size = self.dataset.history_size
        stitched_segments = []
        
        # 3. Autoregressive Loop
        for step in range(stitch_steps):
            # A. Inference
            normalized_sample = self.diffuser.getSample(
                num_trajectories=num_samples,
                state_cond=curr_state_tens,
                goal_cond=task_tens,
                deterministic=self.noise_cfg["deterministic_inference"],
            ) # (B, T, D_out)
            
            # B. Denormalize
            denorm_btc = self.dataset.denormalize_global(normalized_sample)
            future_traj_np = denorm_btc.detach().cpu().numpy()
            
            # C. Reconstruct World Frame
            anchor_arr = np.concatenate([current_anchors['ref_pos'], current_anchors['ref_quat'], current_anchors['ref_obj_pos']], axis=-1)
            
            res = reconstruct_sbto_trajectory(anchor_arr, future_traj_np)
            r_world, o_world = res[0], res[1]
            
            # Store Segment
            # Robot(36) + Object(7)
            segment_world = np.concatenate([r_world[..., :36], o_world[..., :7]], axis=-1)
            stitched_segments.append(segment_world)
            
            # D. Update Condition for next step
            if step < stitch_steps - 1:
                r_hist_new = r_world[:, -history_size:, :]
                o_hist_new = o_world[:, -history_size:, :]
                curr_state_tens, current_anchors = self._update_condition(r_hist_new, o_hist_new)
                curr_state_tens = curr_state_tens.to(self.device)
        
        full_trajectory = np.concatenate(stitched_segments, axis=1)

        # Interpolate if needed (based on dataset downsample config)
        if hasattr(self, 'dataset') and self.dataset is not None:
             full_trajectory = self._interpolate_trajectory(full_trajectory)

        return full_trajectory

