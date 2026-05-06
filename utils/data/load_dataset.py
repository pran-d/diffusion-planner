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
    configured_file_names = config.get("file_names", None)
    if isinstance(configured_file_names, str):
        configured_file_names = [configured_file_names]
    elif configured_file_names is not None:
        configured_file_names = [str(x) for x in configured_file_names]

    # Backward-compatible default for task-list mode.
    default_task_file_names = ["best_trajectory.npz"]
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
            file_found = False
            
            # Check if t exists as an npz file directly
            p_exact = os.path.join(data_root, t)
            p_npz = os.path.join(data_root, f"{t}.npz")
            
            if os.path.isfile(p_exact) and p_exact.endswith('.npz'):
                file_paths.append(p_exact)
                file_found = True
            elif os.path.isfile(p_npz):
                file_paths.append(p_npz)
                file_found = True
            elif os.path.isdir(p_exact):
                file_names = configured_file_names or default_task_file_names
                # e.g. ["top_trajectories.npz", "motion.npz"]
                for file_name in file_names:
                    p = os.path.join(p_exact, file_name)
                    if os.path.exists(p):
                        file_paths.append(p)
                        file_found = True
                        break
            
            if not file_found:
                print(f"Warning: Could not find any valid .npz files for task '{t}' in {p_exact}.")
                
    elif data_root.endswith(".npz") and os.path.isfile(data_root):
        # Direct .npz file path
        file_paths = [data_root]
    else:
        if task_list_path:
            print(f"Warning: task_list_path '{task_list_path}' configured but not found. Falling back to glob.")

        if configured_file_names:
            # No task list: still honor configured file names across all subdirectories.
            for file_name in configured_file_names:
                file_paths.extend(glob.glob(os.path.join(data_root, "**", file_name), recursive=True))
            # Deduplicate while keeping deterministic ordering.
            file_paths = sorted(set(file_paths))
        else:
            file_paths = sorted(glob.glob(os.path.join(data_root, "**/*.npz"), recursive=True))

    if not file_paths:
        print(f"Warning: No .npz files found for data_root: {data_root}")
        return []
    
    # --- 2. Preload Data into RAM ---
    print(f"Preloading {len(file_paths)} files into RAM...")
    ram_cache = []
    for fpath in tqdm(file_paths, desc="Loading Data"):
        ram_cache.append(_load_npz_raw(fpath))
        
    return ram_cache


def _load_npz_raw(fpath):
    """Load a .npz file and return a plain dict of numpy arrays."""
    try:
        data = dict(np.load(fpath, allow_pickle=True))
        data.setdefault("source_file_path", fpath)
        data.setdefault("source_folder_name", os.path.basename(os.path.dirname(fpath)))
        return data
    except Exception as e:
        print(f"Failed to load {fpath}: {e}")
        return {}