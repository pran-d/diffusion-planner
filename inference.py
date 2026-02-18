import torch
import numpy as np
import yaml
import os
import argparse
import time
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from config.configure import load_config, get_data_path, get_norm_path
from models.model import RobotDiffuser
from datasets.flexible_dataset import FlexibleWindowDataset, yaw_to_rot_matrix, yaw_from_quat
from utils.math.sbto_utils import reconstruct_sbto_trajectory

def compute_task_params(current_robot_state, current_obj_state, desired_obj_pos):
    """
    Computes the task parameters (local object displacement) for the diffusion model.

    Args:
        current_robot_state (np.ndarray): Shape (7,) or (3,) [x, y, z, qx, qy, qz, qw]
                                          representing the robot base pose.
        current_obj_state (np.ndarray): Shape (3,) or (7,) [x, y, z, ...]
                                        representing the current object position.
        desired_obj_pos (np.ndarray): Shape (3,) [x, y, z] 
                                      representing the GOAL object position in world frame.

    Returns:
        np.ndarray: Shape (2,) [delta_x_local, delta_y_local] normalized if needed.
    """
    # 1. Extract Positions
    curr_obj_pos = current_obj_state[:3]  # We only care about X, Y for displacement
    goal_obj_pos = desired_obj_pos[:3]
    
    # 2. Calculate World Displacement
    world_delta = goal_obj_pos - curr_obj_pos
    
    # 3. Extract Robot Yaw
    if len(current_robot_state) >= 7:
        quat = current_robot_state[3:7]
    else:
        quat = current_robot_state
    R_ref_inv = yaw_to_rot_matrix(-yaw_from_quat(quat))

    # 4. Rotate into Robot Frame (Global -> Local)
    local_delta = (R_ref_inv @ world_delta[..., None])[..., 0]

    local_delta_norm = np.linalg.norm(local_delta)
    
    if local_delta_norm > 1e-6:
        local_delta = local_delta / local_delta_norm
    
    return local_delta[..., :3]

def update_condition(dataset, robot_world_history, obj_world_history, final_obj_pos=None):
    """
    Update condition for next autoregressive step.
    Extracts last history window and re-computes relative SBTO features.
    """
    B, H, _ = robot_world_history.shape
    next_states = []
    next_anchors = {'ref_pos': [], 'ref_quat': [], 'ref_obj_pos': [], 'final_obj_pos': []}

    for b in range(B):
        # Extract robot (base+joints) and object
        r_slice = robot_world_history[b] # (H, 36)
        o_slice = obj_world_history[b]   # (H, 7)
        
        raw_chunk = {
            'base': r_slice[:, :7],       
            'joints': r_slice[:, 7:36],
            'obj': o_slice[:, :7]
        }
        
        # Compute SBTO feats relative to new start (index 0 of chunk)
        feats, new_anch = dataset._compute_transform(raw_chunk, t_start=0)
        
        if final_obj_pos is not None:
            new_anch['final_obj_pos'] = final_obj_pos[b]
        
        # Assemble Feature Vector
        current_parts = []
        obs_start_idx = dataset.num_features - dataset.num_observations
        cumulative_dim = 0
        
        for key in dataset.feature_order:
            if key in feats:
                part = torch.from_numpy(feats[key]).float()
                part = dataset._normalize(key, part) 
                
                part_dim = part.shape[-1]
                part_end = cumulative_dim + part_dim
                
                # Filter for observation features
                if part_end > obs_start_idx:
                    local_start = max(0, obs_start_idx - cumulative_dim)
                    current_parts.append(part[:H, local_start:])
                
                cumulative_dim += part_dim
        
        c_state = torch.cat(current_parts, dim=-1)
        next_states.append(c_state)
        next_anchors['ref_pos'].append(new_anch['ref_pos'])
        next_anchors['ref_quat'].append(new_anch['ref_quat'])
        next_anchors['ref_obj_pos'].append(new_anch['ref_obj_pos'])
        next_anchors['final_obj_pos'].append(new_anch['final_obj_pos'])
        
    next_state_tens = torch.stack(next_states)

    batched_anchor = {
        'ref_pos': np.stack(next_anchors['ref_pos']),
        'ref_quat': np.stack(next_anchors['ref_quat']),
        'ref_obj_pos': np.stack(next_anchors['ref_obj_pos']),
        'final_obj_pos': np.stack(next_anchors['final_obj_pos']),
    }
    return next_state_tens, batched_anchor


def run_visualization(stitched_trajs, xml_path, goal_vectors=None, final_obj_pos=None):
    from utils.visualize.visualize import MjVisualizer
    
    vis = MjVisualizer(xml_path, close_on_enter=False)
    print("Optimization Complete. Visualizing first sample...")
    print("Controls: SPACE=Pause, ARROWS=Step, ESC=Exit")
    
    # Use first sample (num_samples, T, D) -> (T, D)
    if stitched_trajs.ndim == 3:
        traj = stitched_trajs[0] 
    else:
        traj = stitched_trajs

    T_steps = traj.shape[0]
    t = np.arange(T_steps) * 0.01

    vis.visualize_trajectory(t=t, x_traj=traj, repeat=True, guidance_vec=goal_vectors, goal_pos=final_obj_pos)

    vis.close()

def main():
    parser = argparse.ArgumentParser(description="Clean Inference & Stitching Pipeline")
    parser.add_argument("--epoch", type=str, required=True, help="Checkpoint epoch or path")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--stitch_steps", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="results/inference.npy")
    parser.add_argument("--sample_idx", type=int, default=0, help="Initial condition index (Overridden if traj_idx is set)")
    parser.add_argument("--traj_idx", type=int, default=None, help="Trajectory (file) index")
    parser.add_argument("--batch_idx", type=int, default=0, help="Batch index within file")
    parser.add_argument("--start_time", type=int, default=0, help="Window start timestep")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference (cuda or cpu)")
    parser.add_argument("--cfg_w", type=float, default=1.0, help="Classifier-free guidance weight")
    parser.add_argument("--task_params", nargs="+", type=float, default=None, help="Custom task parameters (e.g., --task_params 0.5 -0.2)")
    parser.add_argument("--visualize_dataset", action="store_true", help="Whether to visualize the original dataset trajectory instead of the generated one")
    parser.add_argument("--action_horizon", type=int, default=None, help="Number of future steps to visualize/control (for dataset visualization)")
    # Guidance arguments
    parser.add_argument("--guidance_wt", type=float, default=0.0, help="Test-time gradient guidance strength")
    parser.add_argument("--guidance_goal", nargs="+", type=float, default=None, help="Target values for guidance (normalized)")
    parser.add_argument("--guidance_indices", nargs="+", type=int, default=None, help="Indices of the state vector to apply guidance on")
    
    args = parser.parse_args()

    # 1. Load Config
    config_path = "config/config.yaml"
    with open(config_path, 'r') as file:
        raw_config = yaml.safe_load(file)
    model_cfg, data_cfg, training_cfg, noise_cfg = load_config(config_path, raw_config.get("auto_conf", False))
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 2. Setup Dataset & Stats
    data_path = get_data_path(data_cfg)
    norm_path = get_norm_path(model_cfg, training_cfg, data_cfg)
    calculate_stats = True
    if norm_path and os.path.exists(norm_path):
        calculate_stats = False

    dataset = FlexibleWindowDataset(
        data_root=data_path, config=data_cfg, 
        calculate_stats=calculate_stats, norm_path=norm_path,
        noise_cfg={}
    )

    # 3. Model
    diffuser = RobotDiffuser(
        model_config=model_cfg, data_config=data_cfg,
        training_config=training_cfg, noise_scheduler_config=noise_cfg,
        mode='inference', device=device
    )
    
    if os.path.exists(args.epoch):
        diffuser.load_weights_from_file(args.epoch)
    else:
        diffuser.loadWeights(int(args.epoch))

    # 4. Prepare Initial Condition
    if args.traj_idx is not None:
        target = (args.traj_idx, args.batch_idx, args.start_time)
        try:
            args.sample_idx = dataset.indices.index(target)
            print(f"Mapped {target} -> Sample {args.sample_idx}")
        except ValueError:
            print(f"Error: Target {target} not found in dataset indices.")
            raise ValueError(f"Target indices {target} (Trajectory {args.traj_idx}, Batch {args.batch_idx}, Start {args.start_time}) not present in valid window list.")

    print(f"Loading initial condition (Sample {args.sample_idx})...")

    # Capture start meta for task updates
    current_file_idx, current_batch_idx, current_start_time = dataset.indices[args.sample_idx]

    # Build index map for faster access
    index_map = {idx_tuple: i for i, idx_tuple in enumerate(dataset.indices)}

    fut_traj, curr_state, task_params, anchor = dataset[args.sample_idx]
    
    history_size = dataset.history_size
    stitched_segments = []
    goal_vectors = []

    # Override task params if provided
    if args.task_params is not None:
        print(f"Overriding task params with: {args.task_params}")
        anchor["final_obj_pos"] = torch.tensor(args.task_params, dtype=torch.float32)

    current_anchors = {
        'ref_pos': np.tile(anchor['ref_pos'][None], (args.num_samples, 1)),
        'ref_quat': np.tile(anchor['ref_quat'][None], (args.num_samples, 1)),
        'ref_obj_pos': np.tile(anchor['ref_obj_pos'][None], (args.num_samples, 1)),
        'final_obj_pos': np.tile(anchor['final_obj_pos'][None], (args.num_samples, 1)),
    }

    curr_state_tens = curr_state.unsqueeze(0).repeat(args.num_samples, 1, 1)

    # 5. Autoregressive Loop
    for step in range(args.stitch_steps):
        print(f"Generating segment {step+1}/{args.stitch_steps}...")

        tp_init = compute_task_params(
            current_robot_state=current_anchors['ref_quat'], 
            current_obj_state=current_anchors['ref_obj_pos'], 
            desired_obj_pos=current_anchors["final_obj_pos"]
        ) 
        task_params = dataset._normalize("task_params", tp_init)
        task_tens = task_params.repeat(args.num_samples, 1)
        
        # A. Inference
        if not args.visualize_dataset:
            normalized_sample = diffuser.getSample(
                num_trajectories=args.num_samples,
                state_cond=curr_state_tens.to(device),
                goal_cond=task_tens.to(device),
                deterministic=True,
                cfg_w=args.cfg_w,
                guidance_wt=args.guidance_wt,
                guidance_goal=args.guidance_goal,
            )
        else:
            normalized_sample = fut_traj.unsqueeze(0)
        
        # B. Denormalize
        denorm_btc = dataset.denormalize_global(normalized_sample)
        future_traj_np = denorm_btc.cpu().numpy()
        
        # C. Reconstruct World Frame
        anchor_arr = np.concatenate([
            current_anchors['ref_pos'], 
            current_anchors['ref_quat'], 
            current_anchors['ref_obj_pos'],
            current_anchors['final_obj_pos']
        ], axis=-1)

        # Assuming reconstruct_sbto_trajectory returns (robot, object, ...)
        res = reconstruct_sbto_trajectory(anchor_arr, future_traj_np, inpaint=diffuser.model_cfg.get("inpaint", False))
        r_world, o_world = res[0], res[1]

        if args.action_horizon is not None:
            r_world = r_world[:, :args.action_horizon, :]
            o_world = o_world[:, :args.action_horizon, :]
        
        # Store Segment
        # Robot(36) + Object(7)
        segment_world = np.concatenate([r_world[..., :36], o_world[..., :7]], axis=-1)
        stitched_segments.append(segment_world)
    
        # Denormalize task params before transforming to global frame
        task_denorm = dataset._denormalize("task_params", task_tens) # (B, 2)
        # task_denorm_3d = torch.cat([task_denorm, torch.zeros_like(task_denorm[:, :1])], dim=1)[..., None] # (B, 3)
        task_denorm_3d = task_denorm[..., None]
        goal_vec_global = (yaw_to_rot_matrix(yaw_from_quat(current_anchors['ref_quat'])) @ task_denorm_3d.cpu().numpy())[..., :3, 0] # (B, 3)        
        goal_vectors.append(goal_vec_global.repeat(segment_world.shape[1], 0)) # (B, 3)

        # Desired Displacement (From Ground Truth Full Trajectory)                    
        err = np.linalg.norm(current_anchors["final_obj_pos"] - segment_world[:, -1, 36:39])
        if err < 0.25:
            print(f"Segment {step+1} successfully reached the goal (Error: {err:.4f}).")
            break

        # D. Update Condition
        if step < args.stitch_steps - 1:
            r_hist = r_world[:, -history_size:, :]
            o_hist = o_world[:, -history_size:, :]
            curr_state_tens, current_anchors = update_condition(dataset, r_hist, o_hist, final_obj_pos=current_anchors['final_obj_pos'])

            if  args.visualize_dataset: 
                # Update Task Params for next window
                next_start = current_start_time + (step + 1) * dataset.window_size
                target_key = (current_file_idx, current_batch_idx, next_start)
                if target_key in index_map:
                    next_idx = index_map[target_key]
                    fut_traj, _, next_task, _ = dataset[next_idx]
                    task_tens = next_task.unsqueeze(0).repeat(args.num_samples, 1).to(device)
                    print(f"Updated task params for step {step+1} to sample {next_idx} (Start T={next_start})")
                else:
                    print(f"Warning: Could not find next window starting at {next_start}. Reusing previous task params.")


    # 6. Finalize
    full_trajectory = np.concatenate(stitched_segments, axis=1) # (B, T, D)
    goal_vectors = np.concatenate(goal_vectors, axis=0) # (B, task_dim)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    np.save(args.save_path, full_trajectory)
    print(f"Stitched trajectory saved to {args.save_path} (Shape: {full_trajectory.shape})")
    
    # Object indices: 36:39 (pos), 39:43 (quat)
    if full_trajectory.shape[-1] >= 39:
        # Achieved Displacement
        start_obj = full_trajectory[0, 0, 36:39]
        end_obj = full_trajectory[0, -1, 36:39]
        achieved_displacement = (end_obj - start_obj)
        desired_displacement = current_anchors["final_obj_pos"] - current_anchors["ref_obj_pos"]
        try:            
            print("-" * 30)
            print(f"Goal: Full Trajectory Displacement")
            print(f"Desired (GT) Delta XY: {desired_displacement[:2]}")
            print(f"Achieved (Gen) Delta XY: {achieved_displacement[:2]}")
            err = np.linalg.norm(desired_displacement - achieved_displacement)
            print(f"L2 Error: {err:.4f}")
            print("-" * 30)
            
        except Exception as e:
            print(f"Could not load ground truth for comparison: {e}")
            # Fallback to previous method if needed, but it's likely wrong for windows
            pass

    # 7. Visualize
    xml_path = "mj_model.xml" 
    if not os.path.exists(xml_path):
            xml_path = os.path.join(data_path, "mj_model.xml")
            
    if os.path.exists(xml_path):
        run_visualization(full_trajectory, xml_path, goal_vectors, anchor["final_obj_pos"])
    else:
        print("Could not find mj_model.xml for visualization.")

if __name__ == "__main__":
    main()
