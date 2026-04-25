import numpy as np
from config.configure import get_data_path, get_norm_path, load_config
from datasets.flexible_dataset import FlexibleWindowDataset
from utils.data.load_dataset import preload_dataset

def diagnose_mirror_signs(dataset, n_samples=50):
    """
    For each sample, compare the original joints at ref_idx 
    with what a perfect mirror should look like.
    
    A perfect mirror of sample A should produce joints that are 
    statistically consistent with other samples in the dataset.
    We check sign conventions by looking at the mean of each 
    joint after swapping — it should match the mean of the 
    opposite limb's joints.
    """
    from utils.math.sbto_utils import build_feature_layout
    
    left_arm  = list(range(15, 22))  # indices within joints block
    right_arm = list(range(22, 29))
    
    # Collect raw (unnormalized) joint values at ref_idx
    raw_joints = []
    for i in range(min(n_samples, len(dataset))):
        file_idx, batch_idx, t_start = dataset.indices[i]
        raw = dataset._get_single_traj(file_idx, batch_idx)
        features, _ = dataset._compute_transform(raw, t_start)
        j = features['joints'][dataset.history_size - 1]  # ref_idx frame
        raw_joints.append(j)
    
    raw_joints = np.array(raw_joints)  # (N, 29)
    
    print("=== Arm joint analysis (raw, at ref_idx) ===")
    print(f"{'Joint':<30} {'Left mean':>12} {'Right mean':>12} {'Ratio L/R':>12}")
    print("-" * 70)
    
    arm_names = [
        'shoulder_pitch', 'shoulder_roll', 'shoulder_yaw',
        'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw'
    ]
    
    for i, name in enumerate(arm_names):
        l_mean = raw_joints[:, 15 + i].mean()
        r_mean = raw_joints[:, 22 + i].mean()
        ratio = l_mean / r_mean if abs(r_mean) > 1e-4 else float('nan')
        print(f"{name:<30} {l_mean:>12.4f} {r_mean:>12.4f} {ratio:>12.3f}")
    
    print()
    print("=== After swap only (no sign flips) ===")
    print("Left slot gets right values, right slot gets left values.")
    print("If means match original distribution, swap-only is correct.")
    print()
    
    # After swap: left slot = right arm values, right slot = left arm values
    # For a symmetric robot, mean(left_joint_i) should ≈ mean(right_joint_i)
    # If they differ in sign, that joint needs a sign flip
    for i, name in enumerate(arm_names):
        l_mean = raw_joints[:, 15 + i].mean()
        r_mean = raw_joints[:, 22 + i].mean()
        # After swap: left slot has r_mean, right slot has l_mean
        # Expected after perfect mirror: should match original distribution
        # i.e. left slot should have ~l_mean and right slot should have ~r_mean
        # So if r_mean ≈ -l_mean, we need a sign flip
        # If r_mean ≈ l_mean, no sign flip needed
        needs_flip = abs(l_mean + r_mean) < abs(l_mean - r_mean)
        print(f"{name:<30} needs_sign_flip={needs_flip}  "
              f"(l+r={l_mean+r_mean:.4f} vs |l-r|={abs(l_mean-r_mean):.4f})")

model_cfg, data_cfg, training_cfg, noise_cfg = load_config("config/config.yaml")

data_path = get_data_path(data_cfg)
norm_path = get_norm_path(model_cfg, training_cfg, data_cfg)
data_buffer = preload_dataset(data_cfg, data_path)

dataset = FlexibleWindowDataset(
    data_buffer=data_buffer, 
    config=data_cfg, 
    norm_path=norm_path,
    calculate_stats=False,
    training_cfg={}
)
diagnose_mirror_signs(dataset)