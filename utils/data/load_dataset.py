import os
import glob
import yaml
from tqdm import tqdm
import numpy as np

def preload_dataset(config: dict, data_root: str) -> list:
    """
    Resolves .npz file paths based on config and data_root, 
    and preloads them directly into RAM.
    """
    task_list_path = config.get("task_list_path", None)
    file_paths = []
    
    # --- 1. Resolve File Paths ---
    if task_list_path:
        # 1a. Absolute or CWD relative
        if not os.path.exists(task_list_path):
            # 1b. Relative to data_root (usually includes train_path)
            possible_1 = os.path.join(data_root, task_list_path)
            if os.path.exists(possible_1):
                task_list_path = possible_1
            else:
                # 1c. Relative to base dir_path (if separate)
                base_dir = config.get("dir_path", "")
                possible_2 = os.path.join(base_dir, task_list_path)
                if os.path.exists(possible_2):
                    task_list_path = possible_2
                    
    if task_list_path and os.path.exists(task_list_path):
        print(f"Loading task list from {task_list_path}")
        with open(task_list_path, 'r') as f:
            tasks = yaml.safe_load(f)
        
        if isinstance(tasks, dict):
            if 'tasks' in tasks: 
                tasks = tasks['tasks']
            else: 
                tasks = list(tasks.keys())
            
        for t in tasks:
            file_names = ["best_trajectory_rand.npz"]
            # file_names = ["top_trajectories.npz", "motion.npz"]
            
            file_found = False
            for file_name in file_names:
                p = os.path.join(data_root, t, file_name)
                if os.path.exists(p):
                    file_paths.append(p)
                    file_found = True
                    break
            
            if not file_found:
                print(f"Warning: Could not find any valid .npz files for task '{t}' in {os.path.join(data_root, t)}.")
                
    elif data_root.endswith(".npz") and os.path.isfile(data_root):
        # Direct .npz file path
        file_paths = [data_root]
    else:
        if task_list_path:
            print(f"Warning: task_list_path '{task_list_path}' configured but not found. Falling back to glob.")
        file_paths = sorted(glob.glob(os.path.join(data_root, "**/*.npz"), recursive=True))

    if not file_paths:
        print(f"Warning: No .npz files found for data_root: {data_root}")
        return []

    key_mapping = config.get("key_mapping", {})
    start_timestep = config.get("start_timestep", 0)
    downsample = config.get("downsample", 1)
    feature_order = config.get("feature_order", None)
    
    # --- 2. Preload Data into RAM ---
    print(f"Preloading {len(file_paths)} files into RAM...")
    ram_cache = []
    for fpath in tqdm(file_paths, desc="Loading Data"):
        # Make sure _load_and_process_file is accessible in this scope
        ram_cache.append(_load_and_process_file(fpath, key_mapping, start_timestep=start_timestep, downsample=downsample, feature_order=feature_order))
        
    return ram_cache

def get_k(key_mapping, key):
    return key_mapping.get(key, key)

def _load_and_process_file(fpath, key_mapping, start_timestep=0, downsample=1, feature_order=None):
        """
        Loads file and returns dict of (N, T, D) arrays with downsampling applied.
        """
        processed = {}
        try:
            with np.load(fpath, allow_pickle=True) as data:
                # RL Rollout Schema
                if 'body_pos_w' in data:
                    is_batched = data['body_pos_w'].ndim == 4
                    # Helper to get (N, T, D)
                    def extract(key):
                        arr = data[key] # (N, T, ...) or (T, ...)
                        if not is_batched:
                            arr = arr[None, ...] # Make (1, T, ...)
                        return arr[:, start_timestep::downsample]
                    
                    # Base (Merge pos+quat)
                    body_pos = extract('body_pos_w') 
                    body_quat = extract('body_quat_w')
                    # Expect (N, T, K, 3). Take K=0
                    # If (N, T, D), then just take it.
                    if body_pos.ndim == 4:
                         processed['base'] = np.concatenate([body_pos[:, :, 0, :], body_quat[:, :, 0, :]], axis=-1)
                    else:
                         # Assume already (N, T, 3) 
                         processed['base'] = np.concatenate([body_pos, body_quat], axis=-1)

                    processed['joints'] = extract('joint_pos')
                    processed['obj'] = np.concatenate([extract('object_pos_w'), extract('object_quat_w')], axis=-1)

                    if 'ee_rel_pos' in data:
                        processed['ee_rel_pos'] = extract('ee_rel_pos')
                    
                elif get_k(key_mapping, 'body_pos_w') in data and get_k(key_mapping, 'body_quat_w') in data:
                    # Same as above but with key mapping
                    is_batched = data[get_k(key_mapping, 'body_pos_w')].ndim == 3

                    def extract(key):
                        real_key = get_k(key_mapping, key)
                        arr = data[real_key]
                        if not is_batched:
                            arr = arr[None, ...]
                        return arr[:, ::downsample]

                    body_pos = extract(get_k(key_mapping, 'body_pos_w'))
                    body_quat = extract(get_k(key_mapping, 'body_quat_w'))
                    processed['base'] = np.concatenate([body_pos, body_quat], axis=-1)
                    processed['joints'] = extract(get_k(key_mapping, 'joint_pos'))
                    processed['obj'] = np.concatenate([extract(get_k(key_mapping, 'object_pos_w')), extract(get_k(key_mapping, 'object_quat_w'))], axis=-1)
                    
                    if 'ee_rel_pos' in data:
                         processed['ee_rel_pos'] = extract(get_k(key_mapping, 'ee_rel_pos'))
                    
                elif 'base_xyz_quat' in data:
                    # SBTO Schema
                    is_batched = data['base_xyz_quat'].ndim == 3
                     # Helper
                    def extract(key):
                        arr = data[key]
                        if not is_batched:
                            arr = arr[None, ...]
                        return arr[:, ::downsample]

                    processed['base'] = extract('base_xyz_quat')
                    processed['joints'] = extract('actuator_pos')
                    processed['obj'] = extract('obj_0_xyz_quat')
                    
                    if 'base_linvel_angvel' in data:
                        processed['base_vel'] = extract('base_linvel_angvel')
                    if 'actuator_vel' in data:
                        processed['joints_vel'] = extract('actuator_vel')
                    if 'obj_0_linvel_angvel' in data:
                        processed['obj_vel'] = extract('obj_0_linvel_angvel')

                    if 'ee_rel_pos' in data:
                        processed['ee_rel_pos'] = extract('ee_rel_pos')
                
                # Metadata
                if get_k(key_mapping, 'fps') in data:
                    val = data[get_k(key_mapping, 'fps')]
                    fps_val = val.item() if val.ndim == 0 else val[0]
                    B_dim = processed['base'].shape[0]
                    processed['fps'] = np.repeat(fps_val, B_dim)

                # Pre-compute ee_rel_pos via FK if not in file.
                if feature_order and 'ee_rel_pos' in feature_order and 'ee_rel_pos' not in processed and 'joints' in processed:
                    try:
                        import mujoco
                        from utils.math.sbto_utils import compute_fk_batched
                        mj_model = mujoco.MjModel.from_xml_path("./mj_model.xml")
                        mj_data = mujoco.MjData(mj_model)
                        # processed['joints'] is (N, T, 29)
                        processed['ee_rel_pos'] = compute_fk_batched(
                            model=mj_model,
                            data=mj_data,
                            qpos_batch=processed['joints'],
                            base_name="pelvis",
                            ee_names=["left_wrist_roll_link", "right_wrist_roll_link"],
                            joint_offset=7,
                        )
                    except Exception as e:
                        print(f"Warning: Could not pre-compute ee_rel_pos for {fpath}: {e}")
                    
        except Exception as e:
            print(f"Failed to load {fpath}: {e}")
            # Return dummy or handle error? indices should prevent access if validation worked.
            # But indices() only reads header. 
            pass 
            
        return processed