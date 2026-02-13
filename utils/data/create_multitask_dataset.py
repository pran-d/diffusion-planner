import numpy as np
import torch
import argparse
import os
import yaml
import shutil
import gc
from tqdm import tqdm
from utils.math.sbto_utils import compute_sbto_features


# ======================
# --- Helper Functions
# ======================
def load_tasks(task_list_path):
    with open(task_list_path, 'r') as f:
        tasks = yaml.safe_load(f)
    if not tasks:
        raise ValueError(f"No tasks found in {task_list_path}")
    return tasks


import glob

# ======================
# --- Helper Functions
# ======================
def load_trajectory(file_path, downsample_factor=1):
    if not os.path.exists(file_path):
        return None

    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
    # helper to ensure (N, T, D)
    def ensure_batched(arr):
        if arr.ndim == 2:
            return arr[None, ...]
        return arr

    # 1. RL Rollout Schema
    if 'body_pos_w' in data:
        # data['body_pos_w'] is (N, T, K, 3) or (T, K, 3)
        body_pos = ensure_batched(data['body_pos_w'].copy())
        body_quat = ensure_batched(data['body_quat_w'].copy())

        # If data was (T, K, 3), ensure_batched makes it (1, T, K, 3). 
        # But if it was (N, T, K, 3), it stays same.
        # Wait, ensure_batched above checks ndim==2. 
        # body_pos dims: (N, T, K, 3) -> 4 dims. (T, K, 3) -> 3 dims.
        
        if data['body_pos_w'].ndim == 3:
             # (T, K, 3) -> (1, T, K, 3)
             body_pos = data['body_pos_w'][None, ...]
             body_quat = data['body_quat_w'][None, ...]
             joints = data['joint_pos'][None, ...]
             obj_pos = data['object_pos_w'][None, ...]
             obj_quat = data['object_quat_w'][None, ...]
        else:
             body_pos = data['body_pos_w']
             body_quat = data['body_quat_w']
             joints = data['joint_pos']
             obj_pos = data['object_pos_w']
             obj_quat = data['object_quat_w']

        # Construct base: (N, T, 7) using K=0
        base = np.concatenate([body_pos[..., 0, :], body_quat[..., 0, :]], axis=-1)
        
        # Construct obj: (N, T, 7)
        obj = np.concatenate([obj_pos, obj_quat], axis=-1)
        
        # Joints: (N, T, D)
        # joints already loaded

        # Placeholder velocities (if not present)
        base_vel = np.zeros_like(base)
        joint_vel = np.zeros_like(joints)
        obj_vel = np.zeros_like(obj)
        
        # Apply downsample
        if downsample_factor > 1:
            base = base[:, ::downsample_factor]
            joints = joints[:, ::downsample_factor]
            base_vel = base_vel[:, ::downsample_factor]
            joint_vel = joint_vel[:, ::downsample_factor]
            obj = obj[:, ::downsample_factor]
            obj_vel = obj_vel[:, ::downsample_factor]

        return base, joints, base_vel, joint_vel, obj, obj_vel

    # 2. SBTO OmniRetarget Dataset Schema
    elif 'base_xyz_quat' in data:
        base = data["base_xyz_quat"]
        joints = data["actuator_pos"]
        obj = data["obj_0_xyz_quat"]
        
        # Handle velocities if available
        base_vel = data.get("base_linvel_angvel", np.zeros_like(base))
        joint_vel = data.get("actuator_vel", np.zeros_like(joints))
        obj_vel = data.get("obj_0_linvel_angvel", np.zeros_like(obj))

        # Ensure batch dim
        # If (T, D), make (1, T, D)
        if base.ndim == 2:
            base = base[None, ...]
            joints = joints[None, ...]
            obj = obj[None, ...]
            base_vel = base_vel[None, ...]
            joint_vel = joint_vel[None, ...]
            obj_vel = obj_vel[None, ...]

        # Downsample
        if downsample_factor > 1:
            base = base[:, ::downsample_factor]
            joints = joints[:, ::downsample_factor]
            base_vel = base_vel[:, ::downsample_factor]
            joint_vel = joint_vel[:, ::downsample_factor]
            obj = obj[:, ::downsample_factor]
            obj_vel = obj_vel[:, ::downsample_factor]

        return base, joints, base_vel, joint_vel, obj, obj_vel

    return None



def create_windows(base, joints, obj, base_vel=None, joint_vel=None, obj_vel=None,
                   window_size=50, stride=1, save_velocities=False, multiple_goals=False):
    """
    Create sliding windows with optional padding:
    """
    N, T, D_base = base.shape
    if T < 1:
        return None

    # Pad with repeated first timestep at the start and last timestep at the end
    pad_start = 1
    pad_end = 1
    pad_kwargs = lambda arr: np.pad(arr, ((0,0),(pad_start, pad_end),(0,0)), mode='edge')

    base = pad_kwargs(base)
    joints = pad_kwargs(joints)
    obj = pad_kwargs(obj)
    if save_velocities:
        base_vel = pad_kwargs(base_vel)
        joint_vel = pad_kwargs(joint_vel)
        obj_vel = pad_kwargs(obj_vel)

    # Recompute length after padding
    T_padded = base.shape[1]

    w_base, w_joints, w_obj = [], [], []
    w_base_vel, w_joint_vel, w_obj_vel = [], [], []
    w_additional_goals = []
    
    for i in range(N):
        starts = np.arange(0, T_padded - window_size + 1, stride)
        for s in starts:
            e = s + window_size
            w_base.append(base[i, s:e])
            w_joints.append(joints[i, s:e])
            w_obj.append(obj[i, s:e])
            if save_velocities:
                w_base_vel.append(base_vel[i, s:e])
                w_joint_vel.append(joint_vel[i, s:e])
                w_obj_vel.append(obj_vel[i, s:e])
            if multiple_goals:
                # Shape (2, 7) -> we will stack these later per window
                goals_for_window = []
                goals_for_window.append(obj[i, min(s + 10 * window_size, T_padded - 1)])
                w_additional_goals.append(np.stack(goals_for_window)) # List of (2,7) arrays

    if not w_base:
        return None

    return {
        "base": np.stack(w_base),
        "joints": np.stack(w_joints),
        "obj": np.stack(w_obj),
        "base_vel": np.stack(w_base_vel) if save_velocities else None,
        "joint_vel": np.stack(w_joint_vel) if save_velocities else None,
        "obj_vel": np.stack(w_obj_vel) if save_velocities else None,
        "additional_goals": np.stack(w_additional_goals) if multiple_goals else None,
    }


def compute_features(windowed_data, history_size=4, save_velocities=False, goal_in_curr_state=True):
    return compute_sbto_features(
        windowed_data["base"], windowed_data["joints"], windowed_data["obj"],
        history_size,
        base_vel=windowed_data.get("base_vel"),
        joints_vel=windowed_data.get("joint_vel"),
        obj_vel=windowed_data.get("obj_vel"),
        additional_goals=windowed_data.get("additional_goals"),
        save_velocities=save_velocities,
        goal_in_curr_state=goal_in_curr_state,
    )


def save_chunk(temp_dir, task_idx, obs_hist, obs_future, guidance, current_state, base_pose_world, extra_goals=None):
    chunk_path = os.path.join(temp_dir, f"chunk_{task_idx}.npz") 
    np.savez(
        chunk_path, 
        history=obs_hist, 
        future=obs_future, 
        goal=guidance, 
        current_state=current_state,
        base_pose_world=base_pose_world,
        extra_goals=extra_goals,
    )
    return chunk_path


# ======================
# --- Main Processing
# ======================

def process_multi_task_dataset(
    task_list_path,
    dataset_root,
    output_path,
    downsample_factor=1,
    history_size=4,
    window_size=50,
    window_stride=1,
    shuffle=False,
    save_velocities=False,
    multiple_goals=False,
    goal_in_curr_state=False,
):
    # Flexible file gathering logic
    files = []
    
    # 1. Try task list
    if task_list_path and os.path.exists(task_list_path):
        print(f"Loading task list from {task_list_path}")
        try:
            tasks_list = load_tasks(task_list_path)
             # Support both list of strings or dict
            if isinstance(tasks_list, dict):
                 if 'tasks' in tasks_list: tasks_list = tasks_list['tasks']
                 else: tasks_list = list(tasks_list.keys())
            
            for t in tasks_list:
                # Try constructed path
                p = os.path.join(dataset_root, t, "top_trajectories.npz")
                if os.path.exists(p):
                    files.append(p)
                else:
                    # Try original "top_trajectories.npz" as fallback or direct path
                    p2 = os.path.join(dataset_root, t)
                    if os.path.exists(p2) and p2.endswith(".npz"):
                        files.append(p2)
        except Exception as e:
            print(f"Error reading task list: {e}")
            
    # 2. Fallback to glob
    if not files:
        print(f"No files found via task list. Globbing {dataset_root}...")
        files = glob.glob(os.path.join(dataset_root, "**/*.npz"), recursive=True)
        files = sorted(files)

    if not files:
        print(f"No .npz files found in {dataset_root}!")
        return

    temp_dir = os.path.join(os.path.dirname(output_path), "temp_chunks")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    print(f"Processing {len(files)} files...")
    processed_count = 0

    for file_idx, file_path in enumerate(tqdm(files)):
        try:
            data = load_trajectory(file_path, downsample_factor)
            if data is None:
                continue
            base, joints, base_vel, joint_vel, obj, obj_vel = data

            windows = create_windows(
                base, joints, obj, base_vel, joint_vel, obj_vel,
                window_size, window_stride, save_velocities, multiple_goals
            )
            if windows is None:
                continue

            obs_hist, obs_future, guidance, current_state, base_pose_world, extra_goals = compute_features(
                windows, history_size, save_velocities, goal_in_curr_state=goal_in_curr_state
            )

            save_chunk(temp_dir, file_idx, obs_hist, obs_future, guidance, current_state, base_pose_world, extra_goals)
            processed_count += 1

            del data, windows, obs_hist, obs_future, guidance, current_state, base_pose_world, extra_goals
            gc.collect()
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if processed_count == 0:
        print("No data processed.")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return

    merge_chunks(temp_dir, output_path, shuffle, args.multiple_goals)
    print("Done.")


# ======================
# --- Merge and Stats
# ======================

def merge_chunks(temp_dir, output_path, shuffle=False, multiple_goals=False):
    chunk_files = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".npz")])
    if shuffle:
        np.random.shuffle(chunk_files)

    # Determine shapes
    with np.load(chunk_files[0]) as first_chunk:
        shape_hist = first_chunk['history'].shape[1:]
        shape_fut = first_chunk['future'].shape[1:]
        shape_goal = first_chunk['goal'].shape[1:]
        shape_current = first_chunk['current_state'].shape[1:]
        shape_base_pose = first_chunk['base_pose_world'].shape[1:]
        if multiple_goals:
            shape_extra_goals = first_chunk['extra_goals'].shape[1:] 

    total_samples = sum(np.load(cf)['history'].shape[0] for cf in chunk_files)

    # Allocate memmaps
    final_hist = np.memmap(os.path.join(temp_dir, 'final_hist.dat'), dtype='float32', mode='w+', shape=(total_samples, *shape_hist))
    final_fut = np.memmap(os.path.join(temp_dir, 'final_fut.dat'), dtype='float32', mode='w+', shape=(total_samples, *shape_fut))
    final_goal = np.memmap(os.path.join(temp_dir, 'final_goal.dat'), dtype='float32', mode='w+', shape=(total_samples, *shape_goal))
    final_current = np.memmap(os.path.join(temp_dir, 'final_current.dat'), dtype='float32', mode='w+', shape=(total_samples, *shape_current))
    final_base_pose = np.memmap(os.path.join(temp_dir, 'final_base_pose.dat'), dtype='float32', mode='w+', shape=(total_samples, *shape_base_pose))
    if multiple_goals:
        final_extra_goals = np.memmap(os.path.join(temp_dir, 'final_extra_goals.dat'), dtype='float32', mode='w+', shape=(total_samples, *shape_extra_goals))

    current_idx = 0
    for cf in tqdm(chunk_files, desc="Merging"):
        with np.load(cf) as data:
            n = data['history'].shape[0]
            final_hist[current_idx:current_idx+n] = data['history']
            final_fut[current_idx:current_idx+n] = data['future']
            final_goal[current_idx:current_idx+n] = data['goal']
            final_current[current_idx:current_idx+n] = data['current_state']
            final_base_pose[current_idx:current_idx+n] = data['base_pose_world']
            if multiple_goals:
                final_extra_goals[current_idx:current_idx+n] = data['extra_goals']
            current_idx += n

    # Save to output
    save_dir = output_path[:-4] if output_path.endswith('.npz') else output_path
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "history.npy"), final_hist)
    np.save(os.path.join(save_dir, "future.npy"), final_fut)
    np.save(os.path.join(save_dir, "goal.npy"), final_goal)
    np.save(os.path.join(save_dir, "current_state.npy"), final_current)
    np.save(os.path.join(save_dir, "base_pose_world.npy"), final_base_pose)
    if multiple_goals:
        np.save(os.path.join(save_dir, "extra_goals.npy"), final_extra_goals)

    # Cleanup
    del final_hist, final_fut, final_goal, final_current, final_base_pose
    if multiple_goals:
        del final_extra_goals
    gc.collect()
    shutil.rmtree(temp_dir)
    
    print(f"Done. Saved to directory: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_list", type=str, default="test_datasets/chosen_tasks.yml", help="Path to YAML list of tasks")
    parser.add_argument("--dataset_root", type=str, default="test_datasets/SBTO_OmniRetarget_Dataset", help="Root folder of datasets")
    parser.add_argument("--output", type=str, required=True, help="Path to output .npz file")
    
    # Processing args
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--save_velocities", action="store_true", help="Save velocities in the dataset")
    parser.add_argument("--multiple_goals", action="store_true", help="Use multiple goals for SBTO features")
    parser.add_argument("--goal_in_curr_state", action="store_true", help="Include goal in current state for SBTO features")

    args = parser.parse_args()
    
    process_multi_task_dataset(
        task_list_path=args.task_list,
        dataset_root=args.dataset_root,
        output_path=args.output,
        downsample_factor=args.downsample,
        history_size=args.history,
        window_size=args.window,
        window_stride=args.stride,
        shuffle=args.shuffle,
        save_velocities=args.save_velocities,
        multiple_goals=args.multiple_goals,
        goal_in_curr_state=args.goal_in_curr_state, 
    )

# python -m utils.create_multitask_dataset \
# --task_list test_datasets/chosen_tasks.yml \
# --dataset_root test_datasets/SBTO_OmniRetarget_Dataset \
# --output datasets/xml_29dof/pick_drop_dataset.npz \
# --history 3 --window 18 --save_velocities