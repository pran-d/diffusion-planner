import torch
import numpy as np
import yaml
import os
import argparse
import time
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from config.configure import load_config, get_data_path, get_save_path, get_norm_path
from models.model import RobotDiffuser
from datasets.flexible_dataset import FlexibleWindowDataset
from utils.math.sbto_utils import reconstruct_sbto_trajectory

def update_condition(dataset, robot_world_history, obj_world_history):
    """
    Update condition for next autoregressive step.
    Extracts last history window and re-computes relative SBTO features.
    """
    B, H, _ = robot_world_history.shape
    next_states = []
    next_anchors = {'ref_pos': [], 'ref_quat': []}

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

    next_state_tens = torch.stack(next_states)
    batched_anchor = {
        'ref_pos': np.stack(next_anchors['ref_pos']),
        'ref_quat': np.stack(next_anchors['ref_quat']),
    }
    return next_state_tens, batched_anchor

def interpolate_trajectory(trajectory, downsample_factor):
    """
    Interpolate trajectory if downsample > 1.
    trajectory: (B, T, D)
    """
    k = downsample_factor
    if k <= 1:
        return trajectory

    print(f"Interpolating trajectory with downsample factor {k}...")
    
    # Ensure B, T, D shape
    is_batched = True
    if trajectory.ndim == 2:
        trajectory = trajectory[None, ...]
        is_batched = False
        
    B, T, D = trajectory.shape
    
    # Original knots at 0, k, 2k, ...
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
        f = interp1d(original_times, trajectory[b], axis=0, kind='linear', 
                     fill_value=(trajectory[b][0], trajectory[b][-1]), bounds_error=False)
        interpolated[b] = f(target_times)
        
        # 2. Slerp for Rotation dims
        for sl in quat_indices:
            # Only if the quat slice is valid
            if sl.stop <= D:
                 q_vals = trajectory[b, :, sl]
                 rot = R.from_quat(q_vals)
                 slerp = Slerp(original_times, rot)
                 
                 # Clamp times for Slerp to avoid extrapolation error
                 clamped_times = np.clip(target_times, original_times[0], original_times[-1])
                 
                 interp_q = slerp(clamped_times).as_quat()
                 interpolated[b, :, sl] = interp_q
    
    if not is_batched:
        interpolated = interpolated[0]
        
    return interpolated

def run_visualization(stitched_trajs, xml_path):
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

    vis.visualize_trajectory(t=t, x_traj=traj, repeat=True)

    vis.close()

def main():
    parser = argparse.ArgumentParser(description="Clean Inference & Stitching Pipeline")
    parser.add_argument("--epoch", type=str, required=True, help="Checkpoint epoch or path")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--stitch_steps", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="results/inference.npy")
    parser.add_argument("--sample_idx", type=int, default=0, help="Initial condition index")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference (cuda or cpu)")
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
    print(f"Loading initial condition (Sample {args.sample_idx})...")

    _, curr_state, task_params, anchor = dataset[args.sample_idx]
    
    curr_state_tens = curr_state.unsqueeze(0).repeat(args.num_samples, 1, 1).to(device)
    task_tens = task_params.unsqueeze(0).repeat(args.num_samples, 1).to(device)
    
    current_anchors = {
        'ref_pos': np.tile(anchor['ref_pos'][None], (args.num_samples, 1)),
        'ref_quat': np.tile(anchor['ref_quat'][None], (args.num_samples, 1))
    }
    
    history_size = dataset.history_size
    stitched_segments = []

    # 5. Autoregressive Loop
    for step in range(args.stitch_steps):
        print(f"Generating segment {step+1}/{args.stitch_steps}...")
        
        # A. Inference
        normalized_sample = diffuser.getSample(
            num_trajectories=args.num_samples,
            state_cond=curr_state_tens,
            goal_cond=task_tens,
            deterministic=True
        )
        
        # B. Denormalize
        samples_btc = normalized_sample
        tensor_btc = torch.from_numpy(samples_btc).float().to(device)
        denorm_btc = dataset.denormalize_global(tensor_btc)
        future_traj_np = denorm_btc.cpu().numpy()
        
        # C. Reconstruct World Frame
        anchor_arr = np.concatenate([current_anchors['ref_pos'], current_anchors['ref_quat']], axis=-1)
        # Assuming reconstruct_sbto_trajectory returns (robot, object, ...)
        res = reconstruct_sbto_trajectory(anchor_arr, future_traj_np)
        r_world, o_world = res[0], res[1] 
        
        # Store Segment
        # Robot(36) + Object(7)
        segment_world = np.concatenate([r_world[..., :36], o_world[..., :7]], axis=-1)
        stitched_segments.append(segment_world)
        
        # D. Update Condition
        if step < args.stitch_steps - 1:
            r_hist = r_world[:, -history_size:, :]
            o_hist = o_world[:, -history_size:, :]
            curr_state_tens, current_anchors = update_condition(dataset, r_hist, o_hist)
            curr_state_tens = curr_state_tens.to(device)

    # 6. Finalize
    full_trajectory = np.concatenate(stitched_segments, axis=1)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    np.save(args.save_path, full_trajectory)
    print(f"Stitched trajectory saved to {args.save_path} (Shape: {full_trajectory.shape})")

    if dataset.downsample > 1:
         full_trajectory = interpolate_trajectory(full_trajectory, dataset.downsample)


    # 7. Visualize
    if args.visualize:
        xml_path = "mj_model.xml" 
        if not os.path.exists(xml_path):
             xml_path = os.path.join(data_path, "mj_model.xml")
             
        if os.path.exists(xml_path):
            run_visualization(full_trajectory, xml_path)
        else:
            print("Could not find mj_model.xml for visualization.")

if __name__ == "__main__":
    main()
