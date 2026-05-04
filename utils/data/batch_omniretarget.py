import os
import glob
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Per-key augmentation config: (std, special_handling)
# Stds are set ~2-3x above the observed variance noise floor.
# ---------------------------------------------------------------------------
AUG_CONFIG = {
    "qpos": {
        "robot_quat": {"dims": slice(0, 4), "std": 0.002, "renormalize": True},
        "robot_xyz":  {"dims": slice(4, 7), "std": 0.002},
        "joints":     {"dims": slice(7, 36), "std": 0.01},
        "obj_quat":   {"dims": slice(36, 40), "std": 0.001, "renormalize": True},
        "obj_xyz":    {"dims": slice(40, 43), "std": 0.001},
    },
    "base": {
        "xyz":  {"dims": slice(0, 3), "std": 0.02},   # position: 2x above floor (~0.013)
        "quat": {"dims": slice(3, 7), "std": 0.008,   # quaternion: small perturbation
                 "renormalize": True},
    },
    "joints": {
        "all":  {"dims": slice(None), "std": 0.05},   # 2x above floor (~0.047) -> radians
    },
    "obj": {
        "xyz":  {"dims": slice(0, 3), "std": 0.005},  # near-zero variance, keep tiny
        "quat": {"dims": slice(3, 7), "std": 0.002,
                 "renormalize": True},
    },
}

def smooth_noise(shape: tuple, std: float, cutoff_ratio: float = 0.05) -> np.ndarray:
    """
    Generate temporally smooth noise by filtering white noise in frequency domain.
    
    cutoff_ratio: fraction of frequencies to keep (0.05 = only lowest 5%).
                  Lower = smoother. Tune this to control jitter.
    """
    N, T, D = shape
    white = np.random.randn(N, T, D)
    
    freqs     = np.fft.rfft(white, axis=1)           # (N, T//2+1, D)
    n_keep    = max(1, int(cutoff_ratio * (T // 2)))
    freqs[:, n_keep:, :] = 0.0                       # zero out high frequencies
    smooth    = np.fft.irfft(freqs, n=T, axis=1)     # back to (N, T, D)
    
    # Rescale to desired std (filtering changes the amplitude)
    current_std = smooth.std(axis=1, keepdims=True).clip(1e-8)
    smooth = smooth / current_std * std
    
    return smooth


def _to_3d(arr, B):
    if arr is None:
        return None
    if arr.ndim == 3:
        return arr
    if arr.ndim == 2:
        return np.tile(arr[np.newaxis], (B, 1, 1))
    raise ValueError(f"Unsupported array ndim for conversion to 3D: {arr.ndim}")


def build_qpos_from_sbto(data: dict):
    """
    If dataset uses SBTO key naming (actuator_pos, base_xyz_quat, obj_0_xyz_quat),
    synthesize a `qpos` array with layout expected by AUG_CONFIG:
      [robot_quat(4), robot_xyz(3), joints(29), obj_quat(4), obj_xyz(3)] -> 43 dims

    Returns None if required keys are missing.
    Handles both (T,D) and (B,T,D) arrays.
    """
    required = ("actuator_pos", "base_xyz_quat", "obj_0_xyz_quat")
    if not all(k in data for k in required):
        return None

    a = data["actuator_pos"]
    b = data["base_xyz_quat"]
    o = data["obj_0_xyz_quat"]

    # detect batched form
    batched = any(x.ndim == 3 for x in (a, b, o))

    if not batched:
        # expect shapes (T, D)
        robot_quat = b[:, 3:7]
        robot_xyz = b[:, 0:3]
        joints = a
        obj_quat = o[:, 3:7]
        obj_xyz = o[:, 0:3]
        return np.concatenate([robot_quat, robot_xyz, joints, obj_quat, obj_xyz], axis=-1)

    # batched: make all 3D with same B
    B = next(x.shape[0] for x in (a, b, o) if x.ndim == 3)
    a3 = _to_3d(a, B)
    b3 = _to_3d(b, B)
    o3 = _to_3d(o, B)

    robot_quat = b3[:, :, 3:7]
    robot_xyz = b3[:, :, 0:3]
    joints = a3
    obj_quat = o3[:, :, 3:7]
    obj_xyz = o3[:, :, 0:3]

    return np.concatenate([robot_quat, robot_xyz, joints, obj_quat, obj_xyz], axis=-1)


def _normalize_file_names(file_names):
    if file_names is None:
        return None
    if isinstance(file_names, str):
        return [file_names]
    return [str(name) for name in file_names]


def _load_task_list(task_list_path):
    with open(task_list_path, "r") as f:
        tasks = yaml.safe_load(f)
    if isinstance(tasks, dict):
        tasks = tasks["tasks"] if "tasks" in tasks else list(tasks.keys())
    if isinstance(tasks, str):
        tasks = [tasks]
    return [str(task) for task in tasks]


def _load_paths_config(paths_config):
    if not paths_config or not os.path.exists(paths_config):
        return {}
    with open(paths_config, "r") as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("paths", raw)


def _resolve_optional_path(path_value, paths_config, input_dir):
    if not path_value:
        return path_value
    if os.path.exists(path_value):
        return path_value

    config_dir = os.path.dirname(os.path.abspath(paths_config)) if paths_config else ""
    if config_dir:
        candidate = os.path.join(config_dir, path_value)
        if os.path.exists(candidate):
            return candidate

    candidate = os.path.join(input_dir, path_value)
    if os.path.exists(candidate):
        return candidate

    return path_value


def resolve_npz_files(input_dir, task_list_path=None, file_names=None):
    """
    Resolve input .npz files using the same conventions as the training loader.

    Priority:
      1) If task_list_path exists, look for the configured file name(s) inside
         each task folder under input_dir.
      2) Else, if file_names is provided, search recursively for those names.
      3) Else, fall back to all .npz files recursively.
    """
    file_names = _normalize_file_names(file_names)

    if task_list_path and os.path.exists(task_list_path):
        npz_files = []
        for task in _load_task_list(task_list_path):
            task_dir = os.path.join(input_dir, task)
            
            # Check if task is actually an .npz file
            candidate_npz = task_dir if task_dir.endswith(".npz") else f"{task_dir}.npz"
            if os.path.exists(candidate_npz) and os.path.isfile(candidate_npz):
                npz_files.append(candidate_npz)
                continue
                
            if not os.path.isdir(task_dir):
                print(f"Warning: task folder or .npz file not found: {task_dir}")
                continue

            names_to_try = file_names or ["best_trajectory.npz"]
            for file_name in names_to_try:
                candidate = os.path.join(task_dir, file_name)
                if os.path.exists(candidate):
                    npz_files.append(candidate)
                    break
            else:
                print(f"Warning: no configured .npz file found in {task_dir}")

        return sorted(npz_files)

    if file_names:
        npz_files = []
        for file_name in file_names:
            npz_files.extend(glob.glob(os.path.join(input_dir, "**", file_name), recursive=True))
        return sorted(set(npz_files))

    return sorted(glob.glob(os.path.join(input_dir, "**", "*.npz"), recursive=True))

def augment_array(key: str, arr: np.ndarray, N: int,
                  cutoff_ratio: float = 0.05) -> np.ndarray:
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, D   = arr.shape
    copies = np.tile(arr[np.newaxis], (N, 1, 1)).copy()

    cfg = AUG_CONFIG.get(key)

    if cfg is not None:
        for part, spec in cfg.items():
            sl  = spec["dims"]
            std = spec["std"]
            seg = copies[:, :, sl]
            seg_D = seg.shape[-1]

            noise = smooth_noise((N, T, seg_D), std=std, cutoff_ratio=cutoff_ratio)
            seg   = seg + noise

            if spec.get("renormalize"):
                norms = np.linalg.norm(seg, axis=-1, keepdims=True)
                norms = np.where(norms < 1e-8, 1.0, norms)
                seg   = seg / norms

            copies[:, :, sl] = seg

    if squeeze:
        copies = copies[:, :, 0]

    return copies


def process_omniretarget_data(input_dir, output_dir, N=60, task_list_path=None,
                              file_names=None, cutoff_ratio=0.05):
    """
    Reads .npz files in input_dir, augments each trajectory to a batch of N
    using per-field noise scaled to observed dataset variance.

    File discovery follows the dataset loader convention:
      - If task_list_path is set, search inside each task folder for the
        configured file name(s).
      - Otherwise, search recursively for the configured file name(s).
      - If no file names are configured, fall back to all .npz files.

    Outputs are written relative to input_dir so task folders stay separated.
    """
    npz_files = resolve_npz_files(input_dir, task_list_path=task_list_path, file_names=file_names)

    if not npz_files:
        print("No .npz files found!")
        return

    os.makedirs(output_dir, exist_ok=True)

    for file in npz_files:
        data = dict(np.load(file, allow_pickle=True))
        # Detect SBTO-style datasets and synthesize canonical keys
        try:
            qpos = build_qpos_from_sbto(data)
            if qpos is not None and 'qpos' not in data:
                data['qpos'] = qpos
                print("  Synthesized 'qpos' from SBTO keys")
        except Exception as e:
            print(f"  Warning: failed to synthesize qpos: {e}")

        # Provide a `base` key if SBTO uses base_xyz_quat (order: xyz, quat)
        if 'base' not in data and 'base_xyz_quat' in data:
            data['base'] = data['base_xyz_quat']

        batched_data = {}
        print(f"\nProcessing {os.path.basename(file)} ...")

        for k, arr in data.items():
            if not isinstance(arr, np.ndarray) or arr.ndim < 1:
                batched_data[k] = arr  # pass through scalars/metadata (fps etc.)
                continue

            if arr.ndim in [1, 2]:
                # Single trajectory (T,) or (T, D) → augment to (N, T) or (N, T, D)
                batched_data[k] = augment_array(k, arr, N, cutoff_ratio)

            elif arr.ndim == 3:
                # Already batched (B, T, D) — augment each and concat up to N
                existing_N = arr.shape[0]
                if existing_N >= N:
                    batched_data[k] = arr[:N]
                else:
                    # Augment the mean trajectory to fill up to N
                    mean_traj = arr.mean(axis=0)           # (T, D)
                    needed    = N - existing_N
                    aug       = augment_array(k, mean_traj, needed, cutoff_ratio)  # (needed, T, D)
                    batched_data[k] = np.concatenate([arr, aug], axis=0)

            else:
                print(f"  Skipping '{k}': unexpected shape {arr.shape}")
                batched_data[k] = arr

            if isinstance(batched_data[k], np.ndarray):
                print(f"  {k:20s}: {arr.shape} → {batched_data[k].shape}")

        rel_path = os.path.relpath(file, input_dir)
        output_file = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.savez_compressed(output_file, **batched_data)
        print(f"Saved → {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  default="/scratch/project/eu-26-32/diffusion-planner/test_datasets/OmniRetarget_Dataset/robot-object/")
    parser.add_argument("--output_dir", default="/scratch/project/eu-26-32/diffusion-planner/test_datasets/batched_omniretarget")
    parser.add_argument("--paths_config", default="./config/paths.yaml",
                        help="Path to config/paths.yaml used to resolve file_names and task_list_path.")
    parser.add_argument("--task_list",  default=None,
                        help="Path to a .yml task list file to filter processed files.")
    parser.add_argument("--file_names", nargs="*", default=None,
                        help="File name(s) to search for inside each task folder; overrides paths.yaml.")
    parser.add_argument("--N",          type=int, default=60,
                        help="Number of augmented sequences per trajectory.")
    parser.add_argument("--cutoff_ratio", type=float, default=0.05,
                    help="Frequency cutoff for smooth noise (lower = smoother). "
                         "0.05 = very smooth, 0.2 = moderate, 0.5 = nearly white noise.")
    args = parser.parse_args()

    paths_cfg = _load_paths_config(args.paths_config)
    resolved_task_list = args.task_list or paths_cfg.get("task_list_path")
    resolved_task_list = _resolve_optional_path(resolved_task_list, args.paths_config, args.input_dir)
    resolved_file_names = args.file_names if args.file_names else paths_cfg.get("file_names")

    process_omniretarget_data(args.input_dir, args.output_dir,
                              N=args.N, task_list_path=resolved_task_list,
                              file_names=resolved_file_names, cutoff_ratio=args.cutoff_ratio)