import gc
import torch
import numpy as np
import yaml
import os
import math
import time
from typing import List, Dict, Union, Optional
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler
from torch import nn, optim
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.ndimage import gaussian_filter1d

from config.configure import load_config, get_data_path, get_save_path, get_norm_path
from models.model import RobotDiffuser
from datasets.flexible_dataset import FlexibleWindowDataset
from utils.math.sbto_utils import reconstruct_sbto_trajectory, compute_task_params
from utils.math.math_tools import yaw_from_quat, yaw_to_rot_matrix
from inference import build_inference_waypoints


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
        
        # FlexibleWindowDataset: if 'obj' is missing
        if 'task_params' in raw:
             tp_raw = raw['task_params']
        elif 'obj' in raw and 'base' in raw:
             tp_raw, _ = dataset._compute_task_params(raw['base'], raw['obj'])
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
        """Load normalization stats into a lightweight dataset for inference use."""
        norm_path = get_norm_path(self.model_cfg, self.training_cfg, self.data_cfg)
        
        if load_stats and norm_path and os.path.exists(norm_path):
            # Create a tiny single-step buffer just to bootstrap the dataset object
            # so we can use its normalize/denormalize/feature logic.
            dummy_buffer = {
                'base': np.zeros((1, 2, 7)),
                'joints': np.zeros((1, 2, 29)),
                'obj': np.zeros((1, 2, 7)),
            }
            self.dataset = FlexibleWindowDataset(
                data_buffer=dummy_buffer,
                config=self.data_cfg,
                norm_path=norm_path,
                calculate_stats=False,
                training_cfg={},
            )

    def fit(self, 
        data_source: Union[str, List[Dict]], 
        task_params: Optional[List[Dict]] = None,
        epochs: int = None, 
        save_path: str = None,
        checkpoint: Optional[str] = None,
        calc_stats: bool = True,
    ):

        """
        Train the model.
        
        Args:
            data_source: Path to data folder (str) or list of trajectory dicts.
            task_params: Optional list of task parameters (e.g. goals) aligned with data_source.
            epochs: Number of epochs to train. Overrides config if provided.
            save_path: Path to save model checkpoints. Overrides config if provided.
            checkpoint: Optional path to a checkpoint to resume from.
        """
        self.diffuser.model.train()  # Ensure model is in training mode

        if epochs is None:
            epochs = self.training_cfg.get("num_epochs", 100)
        
        # Setup Data
        norm_path = get_norm_path(self.model_cfg, self.training_cfg, self.data_cfg)

        if isinstance(data_source, str):
            raise ValueError("MotionGenerator.fit no longer supports file paths. Please load data into a buffer first.")
        
        # If epochs is 0, we assume inference mode and load stats.
        calc_stats = calc_stats
        if epochs == 0:
            if norm_path and os.path.exists(norm_path):
                print(f"Epochs=0 detected. Loading normalization stats from {norm_path} instead of recalculating.")
                calc_stats = False
            else:
                print(f"Epochs=0 but no stats found at {norm_path}. Recalculating stats from data_source.")

        # --- Free previous dataset to avoid memory accumulation ---
        if self.dataset is not None:
            del self.dataset
            self.dataset = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.dataset = FlexibleWindowDataset(
            data_buffer=data_source,
            task_params=task_params,
            config=self.data_cfg, 
            calculate_stats=calc_stats, 
            norm_path=norm_path,
            training_cfg=self.training_cfg,
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
        # num_workers=0 avoids forking after CUDA context init
        # (fork + CUDA → Warp error 3). Dataset is in-memory, so no I/O benefit.
        train_dataloader = DataLoader(
            self.dataset, 
            batch_size=self.data_cfg["batch_size"], 
            num_workers=2,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=True,
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
                ckpt = self.diffuser.loadWeights(checkpoint)
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

                # if self.training_cfg.get("batches_per_epoch") and step >= self.training_cfg["batches_per_epoch"]:
                #     break
                
                if batch[0].shape[0] < self.data_cfg["batch_size"]:
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

                state_cond = None
                task_cond = None
            
                batch_data = list(batch)
                prediction_target = batch_data[0].to(self.device)
                
                idx = 1
                if state_condition:
                    state_cond = batch_data[idx].to(self.device)
                    idx += 1
                
                if task_condition:
                    task_cond = batch_data[idx].to(self.device)
                    idx += 1
                
                # Construct cond (matches train.py)
                cond = []
                if state_cond is not None: cond.append(state_cond)
                if task_cond is not None: cond.append(task_cond)
                
                model_cond = tuple(cond) if len(cond) > 0 else None
                if len(cond) == 1: model_cond = cond[0]
                
                bs, ts, _ = prediction_target.shape
                timesteps = None  # Let the model handle timestep sampling
                
                with torch.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu", dtype=torch.float32):
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
                last_ckpt_path = fpath

        # --- Cleanup training objects to release GPU/CPU memory ---
        # Note: optimizer, scheduler, scaler are kept on self for reuse across fit() calls
        del train_dataloader
        if sampler is not None:
            del sampler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return last_ckpt_path if epochs > 0 else checkpoint

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
                     # Trajectory stores quats as (w,x,y,z) but scipy expects (x,y,z,w)
                     q_wxyz = trajectory[b, :, sl]
                     q_xyzw = np.roll(q_wxyz, -1, axis=-1)
                     rot = R.from_quat(q_xyzw)
                     slerp = Slerp(original_times, rot)
                     
                     # Clamp times for Slerp to avoid extrapolation error (equivalent to nearest)
                     clamped_times = np.clip(target_times, original_times[0], original_times[-1])
                     
                     interp_q_xyzw = slerp(clamped_times).as_quat()
                     # Convert back to (w,x,y,z) for trajectory storage
                     interpolated[b, :, sl] = np.roll(interp_q_xyzw, 1, axis=-1)
        
        return interpolated

    def _smooth_trajectory(self, trajectory, sigma=1.0):
        """
        Smooth trajectory using Gaussian filter.
        trajectory: (B, T, D)
        """
        # Apply Gaussian filter along time axis
        smoothed = gaussian_filter1d(trajectory, sigma=sigma, axis=1)
        
        # Renormalize quaternions
        # Robot Quat: 3:7
        # Object Quat: 39:43
        quat_indices = [slice(3, 7), slice(39, 43)]
        D = trajectory.shape[-1]

        for sl in quat_indices:
            if sl.stop <= D:
                q_vals = smoothed[..., sl]
                # Normalize
                norm = np.linalg.norm(q_vals, axis=-1, keepdims=True)
                # Avoid division by zero
                norm = np.maximum(norm, 1e-8)
                smoothed[..., sl] = q_vals / norm
        return smoothed

    def _update_condition(self, robot_world_history, obj_world_history, final_obj_pos=None):
        """
        Update condition for next autoregressive step.
        Mirrors inference.py's update_condition().
        """
        if self.dataset is None:
             raise RuntimeError("Dataset not initialized. Call fit() or manually init dataset.")

        B, H, _ = robot_world_history.shape
        next_states = []
        next_anchors = {'ref_pos': [], 'ref_quat': [], 'ref_obj_pos': [], 'final_obj_pos': []}

        for b in range(B):
            r_slice = robot_world_history[b]  # (H, 36)
            o_slice = obj_world_history[b]    # (H, 7)
            
            raw_chunk = {
                'base': r_slice[:, :7],       
                'joints': r_slice[:, 7:36],
                'obj': o_slice[:, :7]
            }
            
            feats, new_anch = self.dataset._compute_transform(raw_chunk, t_start=0)
            
            if final_obj_pos is not None:
                new_anch['final_obj_pos'] = final_obj_pos[b]
            
            # Assemble Feature Vector (same logic as inference.py)
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
            next_anchors['ref_obj_pos'].append(new_anch['ref_obj_pos'])
            next_anchors['final_obj_pos'].append(new_anch.get('final_obj_pos', np.zeros(3)))

        next_state_tens = torch.stack(next_states)
        batched_anchor = {
            'ref_pos': np.stack(next_anchors['ref_pos']),
            'ref_quat': np.stack(next_anchors['ref_quat']),
            'ref_obj_pos': np.stack(next_anchors['ref_obj_pos']),
            'final_obj_pos': np.stack(next_anchors['final_obj_pos']),
        }
        return next_state_tens, batched_anchor

    def generate_trajectory(self, 
                            initial_condition: Dict, 
                            goal_condition: Union[Dict, np.ndarray],
                            target_traj_length: int = None,
                            stitch_steps: int = None, 
                            num_samples: int = 1,
                            cfg_w: float = 1.0,
                            end_error_threshold: float = 0.1,
                            enable_goal_stop: bool = False,
                            enable_physics_stop: bool = False,
                            use_last_frame_wp: bool = True,
                            arrival_ratio: float = 0.85,
                            lift_height: float = 0.5,
                            no_lower_dist: float = 0.5,
                            lift_start: float = 0.0,
                            lift_end: float = 0.20,
                            walk_start_z: float = 0.80,):
        """
        Generate trajectory via autoregressive diffusion (mirrors inference.py).
        
        Args:
            initial_condition: Dict with 'robot' (B, H, 36) or (H, 36) and 
                               'obj' (B, H, 7) or (H, 7) history in world frame.
            goal_condition: np.ndarray (B, D_task) or (D_task,) — desired task params 
                           (e.g. final object position) in world frame.
            target_traj_length: Desired output trajectory length (pre-interpolation).
                                Used to auto-compute stitch_steps from the model window size.
                                Ignored if stitch_steps is explicitly provided.
            stitch_steps: Number of autoregressive segments. If None, auto-computed from
                          target_traj_length and model window size (minimum 1).
            num_samples: Number of parallel samples **per input condition**.
                         When input is already batched (B > 1), set to 1.
            cfg_w: Classifier-free guidance weight.
            end_error_threshold: Stop stitching early if goal reached within this L2 error
                                 (per trajectory). Only used when enable_goal_stop=True.
            enable_goal_stop: If True, each trajectory independently stops generating and
                              pads once its goal is reached within end_error_threshold.
                              Set to False to always run the full stitch_steps.
            enable_physics_stop: If True, truncate and pad when a physical consistency
                                 violation is detected (ground penetration / position spike).
                                 Set to False to continue generating regardless.
            use_last_frame_wp: If True and model supports inbetweening, add a partial
                               waypoint at the last frame with obj_delta_xy + obj_z.
            arrival_ratio: Object should arrive within this fraction of total time (0-1).
            lift_height: Peak lift height in metres for pick-and-place z profile.
            no_lower_dist: Lower z only when remaining XY distance < this (metres).
            lift_start: Fraction of trajectory where lift begins.
            lift_end: Fraction of trajectory where lift reaches peak.
            walk_start_z: Gate XY walk until z >= this fraction of lift_height.
            
        Returns:
            np.ndarray of shape (B * num_samples, T_total, D) where D = 36 (robot) + 7 (object).
        """
        if self.dataset is None:
             raise RuntimeError("Dataset not initialized. Call fit() or setup.")

        self.diffuser.model.eval()
        
        window_size = self.dataset.window_size  # model output length per segment
        history_size = self.dataset.history_size

        # --- Auto-compute stitch_steps from target trajectory length ---
        if stitch_steps is None:
            if target_traj_length is not None:
                _eff = window_size - 1   # each step contributes (T-1) frames (t=0 always skipped)
                stitch_steps = max(1, math.ceil(target_traj_length / _eff))
            else:
                stitch_steps = 1

        # 1. Process Initial Condition
        r_hist = np.asarray(initial_condition['robot'])  # (B, H, 36) or (H, 36)
        o_hist = np.asarray(initial_condition['obj'])    # (B, H, 7) or (H, 7)
        
        if r_hist.ndim == 2: r_hist = r_hist[None, ...]
        if o_hist.ndim == 2: o_hist = o_hist[None, ...]

        B_input = r_hist.shape[0]

        # Get initial encoded state and anchors
        curr_state_tens, current_anchors = self._update_condition(r_hist, o_hist)
        
        # 2. Process Goal
        #    Pipeline passes the desired final object position expressed in the
        #    coordinate frame of the initial robot pelvis.  Convert to world frame
        #    so compute_task_params receives world-frame inputs.
        if isinstance(goal_condition, (np.ndarray, list)):
             gc_np = np.asarray(goal_condition, dtype=np.float64)  # (D,) or (B, D)
             if gc_np.ndim == 1: gc_np = gc_np[None, :]
             
             # Broadcast single goal to match batch
             if gc_np.shape[0] == 1 and B_input > 1:
                 gc_np = np.repeat(gc_np, B_input, axis=0)
        else:
             raise ValueError("Goal condition format not supported yet. Pass np.ndarray or list.")
        
        # Convert from initial-pelvis-local frame to world frame.
        # The goal encodes relative object *displacement* (final − initial)
        # expressed in the robot's yaw-aligned local frame.  To recover the
        # desired final object world position we rotate the displacement back
        # into the world frame and add the *initial object position* (NOT the
        # robot position).
        init_base = r_hist[:, 0, :7]    # (B, 7) initial pelvis [x,y,z, qw,qx,qy,qz]
        init_obj_pos = o_hist[:, 0, :3]  # (B, 3) initial object position (world)
        goal_world = np.zeros((B_input, 3), dtype=np.float64)
        for b in range(B_input):
            yaw = yaw_from_quat(init_base[b, 3:7])
            R_local_to_world = yaw_to_rot_matrix(yaw)
            goal_3d = np.zeros(3, dtype=np.float64)
            n = min(gc_np.shape[-1], 3)
            goal_3d[:n] = gc_np[b, :n]
            goal_world[b] = (R_local_to_world @ goal_3d[:, None])[:, 0] + init_obj_pos[b]

        # 3. Replicate for num_samples (when generating multiple samples per condition)
        if num_samples > 1 and curr_state_tens.shape[0] == 1:
            curr_state_tens = curr_state_tens.repeat(num_samples, 1, 1)
            for k in current_anchors:
                current_anchors[k] = np.repeat(current_anchors[k], num_samples, axis=0)
            goal_world = np.repeat(goal_world, num_samples, axis=0)
        
        curr_state_tens = curr_state_tens.to(self.device)
        effective_batch = curr_state_tens.shape[0]

        # Physics-check thresholds
        _FLOOR_Z        = -0.1   # m — pelvis/object z below this → ground penetration
        _MAX_ROBOT_STEP =  0.1   # m/frame — robot pelvis XY spike threshold
        _MAX_OBJ_STEP   =  0.2   # m/frame — object XY spike threshold
        # Seed "previous position" trackers from the reference frame of the initial condition
        _prev_robot_xyz = current_anchors['ref_pos'][:, :3].copy()    # (B, 3)
        _prev_obj_xyz   = current_anchors['ref_obj_pos'][:, :3].copy() # (B, 3)

        # Per-trajectory goal-stop state
        _goal_reached     = np.zeros(effective_batch, dtype=bool)
        _goal_last_frame  = np.zeros((effective_batch, 43), dtype=np.float64)

        # Episode-start object z — absolute baseline for the z-waypoint target.
        _rest_obj_z = float(current_anchors['ref_obj_pos'][0, 2])

        stitched_segments = []

        # 4. Autoregressive Loop
        with torch.no_grad():
            for step in range(stitch_steps):
                # A. Compute per-step task params (world-frame goal → local displacement)
                task_cond_np, actual_dist = compute_task_params(
                    current_robot_state=current_anchors['ref_quat'],
                    current_obj_state=current_anchors['ref_obj_pos'],
                    desired_obj_pos=goal_world,
                    normalize_goal_vec=self.data_cfg.get("normalize_goal_vec", True),
                    num_task_params=self.data_cfg.get("num_task_params", 3),
                    max_goal_dist=self.dataset.max_obj_displacement,
                )
                # task_actual uses the real (unclipped) distance for waypoint planning
                task_actual = task_cond_np.copy()
                if actual_dist is not None:
                    task_actual[..., -1:] = actual_dist

                task_cond = self.dataset._normalize(
                    'task_params',
                    torch.from_numpy(task_cond_np.astype(np.float32)).to(self.device),
                )

                # B. Build waypoints (full keyframe at t=0 + optional last-frame partial wp)
                wv, wm = None, None
                if getattr(self.diffuser.model, 'inbetweening_enabled', False):
                    remaining = max(stitch_steps - step, 1)
                    _current_obj_z = torch.from_numpy(
                        current_anchors['ref_obj_pos'][:, 2].astype(np.float32)
                    )
                    wv, wm = build_inference_waypoints(
                        curr_state_tens, task_actual, self.dataset,
                        self.data_cfg['num_features'], window_size,
                        use_last_frame_wp=use_last_frame_wp,
                        stitch_steps=stitch_steps,
                        remaining_steps=remaining,
                        arrival_ratio=arrival_ratio,
                        lift_height=lift_height,
                        no_lower_dist=no_lower_dist,
                        lift_start=lift_start,
                        lift_end=lift_end,
                        walk_start_z=walk_start_z,
                        current_obj_z=_current_obj_z,
                        rest_obj_z=_rest_obj_z,
                    )

                # C. Generate normalized sample
                normalized_sample = self.diffuser.getSample(
                    num_trajectories=effective_batch,
                    state_cond=curr_state_tens,
                    goal_cond=task_cond,
                    deterministic=self.noise_cfg.get("deterministic_inference", False),
                    cfg_w=cfg_w,
                    waypoint_values=wv,
                    waypoint_mask=wm,
                )  # (B, T, D_out)
                
                # D. Denormalize
                denorm_btc = self.dataset.denormalize_global(normalized_sample)
                future_traj_np = denorm_btc.detach().cpu().numpy()
                
                # E. Reconstruct World Frame from SBTO features
                #    final_obj_pos in anchor must be in world frame for reconstruction
                current_anchors['final_obj_pos'] = goal_world

                anchor_arr = np.concatenate([
                    current_anchors['ref_pos'], 
                    current_anchors['ref_quat'], 
                    current_anchors['ref_obj_pos'],
                    current_anchors['final_obj_pos'],
                ], axis=-1)
                
                res = reconstruct_sbto_trajectory(
                    anchor_arr, future_traj_np, 
                    inpaint=self.diffuser.model_cfg.get("inpaint", False)
                )
                r_world, o_world = res[0], res[1]

                # Always skip t=0 (anchor frame) — matches inference.py
                r_world = r_world[:, 1:, :]
                o_world = o_world[:, 1:, :]
                
                # Store segment: Robot(36) + Object(7)
                segment_world = np.concatenate([r_world[..., :36], o_world[..., :7]], axis=-1)

                # ── Physical-consistency check ─────────────────────────────────────────
                _rxy   = segment_world[:, :, :2]      # robot pelvis XY  (B, T, 2)
                _rz    = segment_world[:, :, 2]       # robot pelvis Z   (B, T)
                _oxy   = segment_world[:, :, 36:38]   # object XY        (B, T, 2)
                _oz    = segment_world[:, :, 38]      # object Z         (B, T)
                _rxy_p = np.concatenate([_prev_robot_xyz[:, None, :2], _rxy[:, :-1, :]], axis=1)
                _oxy_p = np.concatenate([_prev_obj_xyz[:, None, :2],   _oxy[:, :-1, :]], axis=1)
                _floor   = (_rz < _FLOOR_Z) | (_oz < _FLOOR_Z)
                _rspike  = np.linalg.norm(_rxy - _rxy_p, axis=-1) > _MAX_ROBOT_STEP
                _ospike  = np.linalg.norm(_oxy - _oxy_p, axis=-1) > _MAX_OBJ_STEP
                _bad_t   = np.where((_floor | _rspike | _ospike).any(axis=0))[0]
                _phys_stop = enable_physics_stop and _bad_t.size > 0
                if _phys_stop:
                    _t_bad   = int(_bad_t[0])
                    _t_clamp = max(_t_bad, 1)
                    _last_ok = segment_world[:, _t_clamp - 1 : _t_clamp, :]
                    segment_world = segment_world.copy()
                    segment_world[:, _t_clamp:, :] = _last_ok
                    print(f"[physics] step {step+1}: violation at frame {_t_bad} "
                          f"(floor={bool(_floor.any(0)[_t_bad])}, "
                          f"robot_spike={bool(_rspike.any(0)[_t_bad])}, "
                          f"obj_spike={bool(_ospike.any(0)[_t_bad])}). Truncating and halting.")
                # ──────────────────────────────────────────────────────────────────────

                # ── Per-trajectory goal-stop ───────────────────────────────────────────
                if enable_goal_stop:
                    # 1. Overwrite segments for trajectories already at goal
                    if _goal_reached.any():
                        _seg_T = segment_world.shape[1]
                        segment_world = segment_world.copy()
                        for _di in np.where(_goal_reached)[0]:
                            segment_world[_di] = np.tile(_goal_last_frame[_di], (_seg_T, 1))

                    # 2. Detect trajectories newly reaching goal
                    _obj_end  = segment_world[:, -1, 36:39]                    # (B, 3)
                    _goal_err = np.linalg.norm(_obj_end - goal_world, axis=-1) # (B,)
                    _newly    = (~_goal_reached) & (_goal_err < end_error_threshold)
                    if _newly.any():
                        for _ni in np.where(_newly)[0]:
                            _goal_last_frame[_ni] = segment_world[_ni, -1, :]
                        _goal_reached |= _newly
                        print(f"[goal] step {step+1}: sample {np.where(_newly)[0].tolist()} "
                              f"reached goal (err={_goal_err[_newly].round(4).tolist()})")
                # ──────────────────────────────────────────────────────────────────────

                stitched_segments.append(segment_world)

                _prev_robot_xyz = segment_world[:, -1, :3].copy()
                _prev_obj_xyz   = segment_world[:, -1, 36:39].copy()

                if _phys_stop:
                    _steps_to_pad = stitch_steps - step - 1
                    if _steps_to_pad > 0:
                        _last_frame = segment_world[:, -1:, :]
                        _seg_T      = segment_world.shape[1]
                        _pad_seg    = np.repeat(_last_frame, _seg_T, axis=1)
                        for _ in range(_steps_to_pad):
                            stitched_segments.append(_pad_seg)
                    break

                # ── Early-exit when all samples have reached their goal ────────────────
                if enable_goal_stop and _goal_reached.all():
                    _steps_to_pad = stitch_steps - step - 1
                    if _steps_to_pad > 0:
                        _seg_T   = segment_world.shape[1]
                        _all_pad = np.stack([
                            np.tile(_goal_last_frame[_i], (_seg_T, 1))
                            for _i in range(effective_batch)
                        ])
                        for _ in range(_steps_to_pad):
                            stitched_segments.append(_all_pad.copy())
                        print(f"[goal] All {effective_batch} samples reached goal after "
                              f"step {step+1}. Padding {_steps_to_pad} remaining steps.")
                    break
                # ──────────────────────────────────────────────────────────────────────

                # F. Update Condition for next step
                if step < stitch_steps - 1:
                    r_hist_new = r_world[:, -history_size:, :]
                    o_hist_new = o_world[:, -history_size:, :]
                    curr_state_tens, current_anchors = self._update_condition(
                        r_hist_new, o_hist_new, 
                        final_obj_pos=goal_world,
                    )
                    curr_state_tens = curr_state_tens.to(self.device)
        
        full_trajectory = np.concatenate(stitched_segments, axis=1)

        # Trim or pad to exactly target_traj_length frames
        if target_traj_length is not None:
            T_actual = full_trajectory.shape[1]
            if T_actual > target_traj_length:
                full_trajectory = full_trajectory[:, :target_traj_length, :]
            elif T_actual < target_traj_length:
                deficit = target_traj_length - T_actual
                pad = np.repeat(full_trajectory[:, -1:, :], deficit, axis=1)
                full_trajectory = np.concatenate([full_trajectory, pad], axis=1)

        # # Interpolate if needed (based on dataset downsample config)
        # if self.dataset is not None:
        #      full_trajectory = self._interpolate_trajectory(full_trajectory)

        # Smooth trajectory
        full_trajectory = self._smooth_trajectory(full_trajectory, sigma=2.0)

        return full_trajectory

