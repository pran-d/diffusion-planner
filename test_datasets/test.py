import numpy as np
import sys
import os
import json
from scipy.spatial.transform import Rotation as R

import yaml

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simple_diffusion.utils.visualize import MjVisualizer

def visualize_dataset_trajectory(data, traj_idx, xml_path, viz=None):
    """
    Visualizes a trajectory from the dataset using MjVisualizer.
    
    Args:
        data: Dictionary containing the dataset arrays
        traj_idx: Index of the trajectory to visualize
        xml_path: Path to the MuJoCo XML model file
        viz: Existing MjVisualizer instance (optional)
    """
    # Check for different data formats
    if 'x' in data:
        # Processed dataset format: (N, T, 43)
        # 43 = 7 (base) + 29 (joints) + 7 (object)
        qpos = data['x'][traj_idx]
        
        # Create dummy qvel (zeros) since it's not in the processed dataset
        # nv = 6 (base) + 29 (joints) + 6 (object) = 41
        T = qpos.shape[0]
        nv = 41
        qvel = np.zeros((T, nv))
        
        x_traj = np.concatenate([qpos, qvel], axis=-1)
        
    elif 'base_xyz_quat' in data:
        # Original dataset format
        # Extract data for the specific trajectory
        # Shape: (T, D)
        base_pos = data['base_xyz_quat'][traj_idx]      # (T, 7)
        joint_pos = data['actuator_pos'][traj_idx]      # (T, 29)
        obj_pos = data['obj_0_xyz_quat'][traj_idx]      # (T, 7)
        
        base_vel = data['base_linvel_angvel'][traj_idx] # (T, 6)
        joint_vel = data['actuator_vel'][traj_idx]      # (T, 29)
        obj_vel = data['obj_0_linvel_angvel'][traj_idx] # (T, 6)
        
        # Concatenate to form qpos and qvel
        # Order: Base -> Joints -> Object (Standard MuJoCo convention for scene composition)
        # Note: Ensure this order matches your XML model structure
        qpos = np.concatenate([base_pos, joint_pos, obj_pos], axis=-1)
        qvel = np.concatenate([base_vel, joint_vel, obj_vel], axis=-1)
        
        # Concatenate qpos and qvel for MjVisualizer
        # MjVisualizer expects x_traj where each step is [qpos, qvel]
        x_traj = np.concatenate([qpos, qvel], axis=-1)
    else:
        raise ValueError("Unknown dataset format. Keys found: " + str(data.keys()))
    
    print(f"Visualizing trajectory {traj_idx}")
    print(f"Trajectory length: {x_traj.shape[0]}")
    print(f"State dimension: {x_traj.shape[1]} (nq={qpos.shape[1]}, nv={qvel.shape[1]})")

    pos0 = base_pos[0, :3].copy()
    quat0_mj = base_pos[0, 3:].copy() # w, x, y, z
                    
    # Convert to scipy [x, y, z, w] for rotation calc
    quat0_scipy = quat0_mj[[1, 2, 3, 0]]
                    
    # Extract Yaw at T=0
    r0 = R.from_quat(quat0_scipy)
    yaw0 = r0.as_euler('xyz')[2]
    
    # 2. Create Inverse Transform (Translate -pos0, Rotate -yaw0 around Z)
    r_inv = R.from_euler('z', -yaw0)
    r = R.from_euler('z', yaw0)

    # Helpers
    def transform_pos(r, pos_seq):
        # Apply rotation to relative position
        return r.apply(pos_seq - pos0)
    
    def untransform_pos(r, pos_seq):
        # Reverse rotation and translation
        return r.inv().apply(pos_seq) + pos0
        
    def transform_quat(r, quat_seq_mj):
        # mj(wxyz) -> scipy(xyzw)
        q_scipy = quat_seq_mj[:, [1, 2, 3, 0]]
        r_curr = R.from_quat(q_scipy)
        # Rotate: R_new = R_inv * R_curr
        r_new = r * r_curr
        # scipy(xyzw) -> mj(wxyz)
        q_new_scipy = r_new.as_quat()
        return q_new_scipy[:, [3, 0, 1, 2]]
    
    print("Pre-transform robot state:", base_pos[0])
    print("Pre-transform box state: ", obj_pos[0])

    # 3. Apply Transform
    # Base
    base_pos[:, :3] = transform_pos(r_inv, base_pos[:, :3])
    base_pos[:, 3:] = transform_quat(r_inv, base_pos[:, 3:])
    
    # Object
    obj_pos[:, :3] = transform_pos(r_inv, obj_pos[:, :3])
    obj_pos[:, 3:] = transform_quat(r_inv, obj_pos[:, 3:])

    print("Transformed robot state:", base_pos[0])
    print("Transformed box state: ", obj_pos[0])

    base_pos[:, :3] = untransform_pos(r_inv, base_pos[:, :3])
    base_pos[:, 3:] = transform_quat(r, base_pos[:, 3:])

    obj_pos[:, :3] = untransform_pos(r_inv, obj_pos[:, :3])
    obj_pos[:, 3:] = transform_quat(r, obj_pos[:, 3:])

    print("Re-Transformed robot state:", base_pos[0])
    print("Re-Transformed box state: ", obj_pos[0])
    
    # Create visualizer
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        return

    if viz is None:
        viz = MjVisualizer(xml_path, False)
    
    # Check if model dimensions match data
    if viz.mj_model.nq != qpos.shape[1]:
        print(f"Warning: Model nq ({viz.mj_model.nq}) does not match data qpos dim ({qpos.shape[1]})")
    
    # Create dummy time array
    T = x_traj.shape[0]
    t = np.arange(T) * 0.01 # Assuming 50Hz
    
    print("Starting visualization...")
    print("Controls: Space to Pause/Resume, Left/Right Arrows to Step, Enter to Next")
    viz.visualize_trajectory(t, x_traj, repeat=True)
    return viz

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to a specific .npz file to visualize")
    parser.add_argument("--xml", type=str, default="./test_datasets/mj_model.xml", help="Path to MuJoCo XML model")
    args = parser.parse_args()

    tasks = []
    with open("./test_datasets/tasks.yml", 'r') as file:
        for line in file:
            line = json.loads(line)
            if(("[pick]" in line["task"] or "[place]" in line["task"]) and "[kick]" not in line["task"]):
                tasks.append(line["original_folder"])
                print(line["task"])
    # with open("./test_datasets/chosen_tasks.yml", 'w') as file:
    #     yaml.dump(tasks, file)
    # sys.exit(0)

    # Use absolute path or relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.file:
        # Single file mode
        dataset_path = args.file
        xml_path = args.xml
        
        if not os.path.exists(dataset_path):
            print(f"Error: File {dataset_path} not found.")
            sys.exit(1)
            
        print(f"Loading dataset from {dataset_path}...")
        data = np.load(dataset_path, allow_pickle=True)
        
        # Print keys and shapes for verification
        for k in data.keys():
            print(f"{k}: {data[k].shape}")
            
        viz = None
        # Visualize all trajectories in the file
        num_trajs = 0
        if 'x' in data: num_trajs = data['x'].shape[0]
        elif 'base_xyz_quat' in data: num_trajs = data['base_xyz_quat'].shape[0]
        
        print(f"Found {num_trajs} trajectories.")
        
        try:
            for i in range(num_trajs):
                viz = visualize_dataset_trajectory(data, i, xml_path, viz=viz)
        except KeyboardInterrupt:
            print("\nExiting...")
            if viz: viz.close()
            sys.exit(0)
            
    else:
        # Directory iteration mode (original behavior)
        dataset_root = os.path.join(script_dir, "SBTO_OmniRetarget_Dataset")

        if not os.path.exists(dataset_root):
            print(f"Dataset root directory {dataset_root} not found.")
            sys.exit(1)

        # List all subdirectories
        subdirs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
        subdirs.sort()

        if not subdirs:
            print(f"No subdirectories found in {dataset_root}")
            sys.exit(1)

        print(f"Found {len(subdirs)} dataset folders.")

        viz = None
        last_xml_path = None

        for subdir in subdirs:
            if subdir not in tasks:
                print(f"Skipping {subdir}: not in filtered tasks.")
                continue

            folder_path = os.path.join(dataset_root, subdir)
            dataset_path = os.path.join(folder_path, "top_trajectories.npz")
            xml_path = args.xml

            print(xml_path)

            if not os.path.exists(dataset_path):
                print(f"Skipping {subdir}: best_samples.npz not found.")
                continue
            
            if not os.path.exists(xml_path):
                print(f"Skipping {subdir}: mj_model.xml not found.")
                continue

            print(f"\n{'='*50}")
            print(f"Processing dataset folder: {subdir}")
            print(f"{'='*50}")

            try:
                print(f"Loading dataset from {dataset_path}...")
                data = np.load(dataset_path, allow_pickle=True)
                
                # Print keys and shapes for verification
                for k in data.keys():
                    print(f"{k}: {data[k].shape}")
                    
                # Visualize the first trajectory using the local XML
                print(f"Using model: {xml_path}")
                print("Press Enter to move to the next dataset, Ctrl+C to exit...")
                
                # If XML path changed, we need a new visualizer
                if last_xml_path != xml_path:
                    if viz is not None:
                        viz.close()
                        viz = None
                    last_xml_path = xml_path

                viz = visualize_dataset_trajectory(data, 0, xml_path, viz=viz)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)
            except Exception as e:
                print(f"Error processing {subdir}: {e}")
                continue
            
    if viz is not None:
        viz.close()
    print("\nAll datasets processed.")