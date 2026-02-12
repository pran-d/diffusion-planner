"""
Script for autoregressive (stitched) diffusion rollout.

Responsibilities:
- Parse CLI args
- Load config, data, model
- Autoregressively generate multiple trajectory segments
- Handle optional text + goal conditioning (including custom goals)
- Reconstruct absolute SBTO trajectories
- Visualize stitched result in MuJoCo

"""

import argparse
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml

from config.configure import load_config, get_data_path, get_norm_path
from models.model import RobotDiffuser
from datasets.flexible_dataset import FlexibleWindowDataset
from utils.visualize.visualize import MjVisualizer, DiffusionOverlayVisualizer
from utils.math.sbto_utils import reconstruct_sbto_trajectory

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser("Autoregressive diffusion stitching")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--traj_num", type=int, default=0)
    parser.add_argument("--stitch_steps", type=int, default=3)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--custom_goal", type=float, nargs=7, default=None)
    parser.add_argument("--task_height", type=float, default=None, help="Desired height of the object")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--headless", action="store_true", help="Run without visualization")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--cfg_w", type=float, default=0.0, help="Classifier-free guidance weight")
    parser.add_argument("--return_all", action="store_true", help="Return all generated trajectories instead of just the best one")
    parser.add_argument("--type", type=str, default="clean",
                        help="Type of current state to use: clean, noised, home, zero, copy_pos, random, random_swap")
    parser.add_argument("--render_segments", action="store_true", help="Render each segment during stitching")
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def load_env_and_data():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model_cfg, data_cfg, training_cfg, noise_cfg = load_config("config/config.yaml")

    data_path = get_data_path(data_cfg)
    norm_path = get_norm_path(model_cfg, training_cfg, data_cfg)
    
    print("Loading dataset...")
    # Initialize dataset (loads stats internally)
    dataset = FlexibleWindowDataset(
        data_root=data_path, 
        config=data_cfg, 
        norm_path=norm_path,
        calculate_stats=False,
        noise_cfg=noise_cfg.get("state_conditioning_noise_level", {}) # Apply same noise if needed
    )

    return device, model_cfg, data_cfg, training_cfg, noise_cfg, dataset

# -----------------------------------------------------------------------------
# Main stitching logic
# -----------------------------------------------------------------------------

def autoregressive_rollout(args, diffuser, dataset, model_cfg, data_cfg, noise_cfg, visualizer):
    num_T = data_cfg["num_timesteps"]
    history_size = data_cfg["state_history"]
    
    generated_segments = []

    # Stitch Loop
    for i in range(args.stitch_steps):
        print(f"Segment {i+1}/{args.stitch_steps}")

        # Determine indices/batch
        indices = [args.traj_num + i * num_T] * args.batch_size

        # Load initial batch via Dataset
        curr_states_list = []
        goal_cond_list = []   
        anchors_list = []
        
        print(f"Loading initial conditions for indices: {indices}")
        for idx in indices:
            _, curr, task, anch = dataset[idx]
            curr_states_list.append(curr)
            goal_cond_list.append(task)
            anchors_list.append(anch)

        task_parameter = anchors_list[0]['task_params'] 

        # current_state: (B, 1, F_obs)
        curr_state_tens = torch.stack(curr_states_list).to(diffuser.device)
        
        # goal condition: (B, 1) or whatever `task` is
        goal_cond_tens = torch.stack(goal_cond_list).to(diffuser.device)
        
        # Anchor: dictionary of arrays
        batched_anchor = {
            'ref_pos': np.stack([a['ref_pos'] for a in anchors_list]),
            'ref_quat': np.stack([a['ref_quat'] for a in anchors_list]),
        }
    
        # Determine if we are generating or using GT
        if not args.generate:
            # --- GROUND TRUTH MODE ---            

            gt_futures = []
            gt_anchors_pos = []
            gt_anchors_quat = []
            
            for b_idx, d_idx in enumerate(indices):
                # dataset[d_idx] -> (future, current, task, anchor)
                future_tens, _, _, anchor = dataset[d_idx]
                gt_futures.append(future_tens)
                gt_anchors_pos.append(anchor['ref_pos'])
                gt_anchors_quat.append(anchor['ref_quat'])
                                
            # Stack (B, T, C)
            future_tensor = torch.stack(gt_futures).to(diffuser.device) 
            
            # Denormalize
            denorm_tensor = dataset.denormalize_global(future_tensor)
            future_traj = denorm_tensor.cpu().numpy()
            
            # Anchor Array
            anchor_pos = np.stack(gt_anchors_pos)
            anchor_quat = np.stack(gt_anchors_quat)
            anchor_array = np.concatenate([anchor_pos, anchor_quat], axis=-1)
            
            # Reconstruct
            robot_world, obj_world, _, _ = reconstruct_sbto_trajectory(
                base_pose_world=anchor_array,
                future_traj=future_traj
            )
            
        else:
            # --- GENERATION MODE ---
            if i == 0:
                 diffuser.loadWeights(args.epoch)

            # getSample returns normalized features (B, C, T)
            normalized_sample = diffuser.getSample(
                num_trajectories=args.batch_size,
                state_cond=curr_state_tens, 
                goal_cond=goal_cond_tens,
                deterministic=False,
                cfg_w=args.cfg_w
            ) 
            
            # 1. Transpose to (B, T, C) -- Sample returns (B, T, C) so no transpose needed
            # normalized_sample is (B, T, C)
            
            # 2. Denormalize
            denorm_tensor = dataset.denormalize_global(normalized_sample)
            future_traj = denorm_tensor.cpu().numpy()
            
            # 3. Anchor Construction
            anchor_array = np.concatenate([batched_anchor['ref_pos'], batched_anchor['ref_quat']], axis=-1)

            # 4. Reconstruct
            robot_world, obj_world, _, _ = reconstruct_sbto_trajectory(
                base_pose_world=anchor_array,
                future_traj=future_traj
            )

        # --- Common Post-Processing / Storage ---
        # Concatenate robot [36] and object [7] for 'stitched' result
        segment = np.concatenate([robot_world[..., :36], obj_world[..., :7]], axis=-1)
        generated_segments.append(segment)

        if args.generate:
            # 3. Predict Next State (only needed for generation loop)
            ''' 
            We need to construct the next 'current_state' from the end of the generated segment.
            This involves:
              a. Taking last 'history_size' frames of world-frame trajectory.
              b. Converting them to SBTO relative features (new anchor is T-1 of that history).
              c. Normalizing and Noising again.
            '''
            
            new_curr_states = []
            new_anchors = {'ref_pos': [], 'ref_quat': []}
            
            # Processing per item in batch using dataset methods
            for b in range(args.batch_size):
                T_gen = robot_world.shape[1]
                extract_len = min(history_size, T_gen)
                
                # Slices (last H frames)
                r_slice = robot_world[b, -extract_len:] # (H, 36+)
                o_slice = obj_world[b, -extract_len:]   # (H, 7+)
                
                # Construct raw dict for dataset helper
                raw_chunk = {
                    'base': r_slice[:, :7],       # pos, quat
                    'joints': r_slice[:, 7:36],   # joints
                    'obj': o_slice[:, :7]
                }
                
                # Compute SBTO features for this chunks                
                feats, new_anch = dataset._compute_transform(raw_chunk, t_start=0)
                
                # Assemble Feature Vector (normalize -> cat -> noise)
                current_parts = []
                obs_start_idx = dataset.num_features - dataset.num_observations
                cumulative_dim = 0
                
                for key in dataset.feature_order:
                    if key in feats:
                        part = torch.from_numpy(feats[key]).float()
                        part = dataset._normalize(key, part) # Normalize
                        
                        part_dim = part.shape[-1]
                        part_end = cumulative_dim + part_dim
                        
                        # Filter for observation features (current state)
                        if part_end > obs_start_idx:
                            local_start = max(0, obs_start_idx - cumulative_dim)
                            current_parts.append(part[:history_size, local_start:])
                        
                        cumulative_dim += part_dim
                
                if not current_parts:
                     raise ValueError("No observation features found during rollout update!")

                c_state = torch.cat(current_parts, dim=-1) # (H, F_obs)
                
                new_curr_states.append(c_state)
                new_anchors['ref_pos'].append(new_anch['ref_pos'])
                new_anchors['ref_quat'].append(new_anch['ref_quat'])
            # Update loop variables
            curr_state_tens = torch.stack(new_curr_states).to(diffuser.device)
            batched_anchor = {
                'ref_pos': np.stack(new_anchors['ref_pos']),
                'ref_quat': np.stack(new_anchors['ref_quat']),
            }

    # -----------------------------------------------
    # Post-Processing / Return
    # -----------------------------------------------
    if not generated_segments:
        return None, None, None, None
        
    stitched = np.concatenate(generated_segments, axis=1) # (B, Total_T, D)
    
    # Select best SINGLE trajectory
    best_idx = 0    
    if args.generate and args.batch_size > 1:
        best_distance = float('inf')
        goal_obj_height = args.task_height if args.task_height is not None else task_parameter
        for b in range(0, args.batch_size):
            final_obj_height = stitched[b, -1, 36 + 2] # obj z-pos is at index 36+2
            if abs(final_obj_height - goal_obj_height) < best_distance:
                best_idx = b
                best_distance = abs(final_obj_height - goal_obj_height)
        print(f"Selected trajectory {best_idx} with final object height {stitched[best_idx, -1, 36 + 2].item():.3f} (distance to goal: {best_distance.item():.3f})")
    
    stitched = stitched[best_idx]
    
    return stitched, None


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device, model_cfg, data_cfg, training_cfg, noise_cfg, dataset = load_env_and_data()

    xml_path = "./mj_model.xml"

    visualizer = None
    if not args.headless:
        visualizer = MjVisualizer(xml_path, close_on_enter=False)

    diffuser = RobotDiffuser(
        model_config=model_cfg,
        data_config=data_cfg,
        training_config=training_cfg,
        noise_scheduler_config=noise_cfg,
        mode="infer",
        device=device,
    )

    stitched, guidance_vec = autoregressive_rollout(
        args, diffuser, dataset, model_cfg, data_cfg, noise_cfg, visualizer
    )

    if args.save_path:
        # Append traj_num to avoid overwrite if looping
        base, ext = os.path.splitext(args.save_path)
        path_with_idx = f"{base}_{args.traj_num}{ext}"
        
        if path_with_idx.endswith(".npy"):
            np.save(path_with_idx, stitched)
        elif path_with_idx.endswith(".npz"):
            to_save = stitched
            if to_save.ndim == 3: 
                to_save = to_save[0]
                
            save_dict = {
                'body_pos_w': to_save[:, 0:3][:, None, :],   # (T, 1, 3) 
                'body_quat_w': to_save[:, 3:7][:, None, :],  # (T, 1, 4)
                'joint_pos': to_save[:, 7:36],               # (T, 29)
                'object_pos_w': to_save[:, 36:39],           # (T, 3)
                'object_quat_w': to_save[:, 39:43]           # (T, 4)
            }
            
            np.savez(path_with_idx, **save_dict)
        print(f"Saved stitched trajectory to {path_with_idx}")

    # Single Trajectory
    if stitched.ndim == 3: stitched = stitched[0] # Should be (T, D) if single? Or handled by visualize_trajectory check?
    
    t = np.linspace(
        0,
        stitched.shape[0] * (visualizer.mj_model.opt.timestep if visualizer else 0.02),
        stitched.shape[0],
    )

    if visualizer is not None:
        visualizer.visualize_trajectory(
            t,
            stitched,
            repeat=True,
            guidance_vec=guidance_vec,
        )

    # Optional plotting
    if False:
        fig, axes = plt.subplots(8, 4, figsize=(30, 30), sharex=True)
        axes = axes.flatten()
        for i in range(min(stitched.shape[1], len(axes))):
            axes[i].plot(stitched[:, i])
            axes[i].set_title(f"Field {i}")
        plt.show()
