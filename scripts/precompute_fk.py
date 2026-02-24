
import os
import glob
import numpy as np
import mujoco
from tqdm import tqdm
import argparse

# Import the FK function
# Assuming utils is in the python path or relative
import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))

from utils.math.sbto_utils import compute_fk_batched    
from config.configure import load_config

def precompute_fk(data_root, model_path, config_path=None):
    print(f"Loading MuJoCo model from {model_path}")
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    key_mapping = {}
    if config_path and os.path.exists(config_path):
        print(f"Loading config from {config_path}")
        # load_config returns (model_cfg, data_cfg, training_cfg, noise_sched_cfg)
        try:
            _, data_cfg, _, _ = load_config(config_path)
            key_mapping = data_cfg.get("key_mapping", {})
            print(f"Using key mapping: {key_mapping}")
        except Exception as e:
            print(f"Failed to load config: {e}")
            
    def get_k(key):
        return key_mapping.get(key, key)

    # Find all npz files
    file_paths = sorted(glob.glob(os.path.join(data_root, "**/*.npz"), recursive=True))
    print(f"Found {len(file_paths)} files.")

    for fpath in tqdm(file_paths, desc="Processing files"):
        try:
            force_recompute = False # Set to True if you want to overwrite
            
            with np.load(fpath, allow_pickle=True) as data:
                # Check if already computed
                if 'ee_rel_pos' in data and not force_recompute:
                    continue
                
                # Logic from FlexibleWindowDataset._load_and_process_file
                joints = None
                
                # Suffixes often used in key mapping if not default
                # But here we try to detect based on presence
                
                # Schema 1: RL Rollout Schema
                if 'body_pos_w' in data:
                     # Standard RL Rollout
                    joints = data['joint_pos']

                # Schema 2: Key mapped or variations (if simple check fails)
                elif get_k('body_pos_w') in data:
                    joints = data[get_k('joint_pos')]

                elif 'actuator_pos' in data and 'base_xyz_quat' in data:
                     # Schema 3: SBTO Schema
                     joints = data['actuator_pos']
                     
                # Fallback / Direct checks if schema detection failed but keys exist
                if joints is None:
                    if get_k('joint_pos') in data: # Re-check with mapped key
                        joints = data[get_k('joint_pos')]
                    elif 'joint_pos' in data:
                        joints = data['joint_pos']
                    elif 'actuator_pos' in data:
                        joints = data['actuator_pos']
                
                if joints is None:
                    print(f"Skipping {fpath}: no joint position data found.")
                    continue
                
                # Standardize shape if needed?
                # compute_fk_batched handles (..., N)
                
                # Verify standard robot shape (7 base + N joints)
                # But here we only pass joints to FK. 
                # joints should be (..., 29) for H1 usually.
                
                ee_rel_pos = compute_fk_batched(
                    model=mj_model,
                    data=mj_data,
                    qpos_batch=joints,
                    base_name="pelvis",
                    ee_names=["left_wrist_roll_link", "right_wrist_roll_link"],
                    joint_offset=7
                )
                
                # Save back to file
                # Use dict(data) to get a mutable dictionary from the NpzFile
                new_data = dict(data)
                new_data['ee_rel_pos'] = ee_rel_pos
                
                np.savez_compressed(fpath, **new_data)
                
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--model_path", type=str, default="./mj_model.xml", help="Path to MuJoCo XML model")
    parser.add_argument("--config", type=str, required=False, help="Path to config.yaml")
    
    args = parser.parse_args()
    
    precompute_fk(args.data_root, args.model_path, args.config)
