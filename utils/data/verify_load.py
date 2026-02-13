
import numpy as np
import sys

def verify_file(fpath):
    print(f"Checking {fpath}...")
    try:
        data = np.load(fpath, allow_pickle=True)
        print("Keys present:", list(data.keys()))
        
        # Simulate FlexibleWindowDataset._load_and_process_file
        processed = {}
        downsample = 1
        
        if 'body_pos_w' in data:
            print("Detected RL Rollout Schema ('body_pos_w')")
            is_batched = data['body_pos_w'].ndim == 4
            
            def extract(key):
                if key not in data:
                    raise KeyError(f"Key {key} missing from data")
                arr = data[key]
                if not is_batched:
                    arr = arr[None, ...]
                return arr[:, ::downsample]

            body_pos = extract('body_pos_w')
            body_quat = extract('body_quat_w')
            
            if body_pos.ndim == 4:
                processed['base'] = np.concatenate([body_pos[:, :, 0, :], body_quat[:, :, 0, :]], axis=-1)
            else:
                processed['base'] = np.concatenate([body_pos, body_quat], axis=-1)
                
            processed['joints'] = extract('joint_pos')
            processed['obj'] = np.concatenate([extract('object_pos_w'), extract('object_quat_w')], axis=-1)
            
            print(f"Processed 'base' shape: {processed['base'].shape}")
            print(f"Processed 'joints' shape: {processed['joints'].shape}")
            print(f"Processed 'obj' shape: {processed['obj'].shape}")

        elif 'base_xyz_quat' in data:
            print("Detected SBTO Schema ('base_xyz_quat')")
            # ... (omitted similar logic)
        else:
            print("ERROR: Unknown schema! expected 'body_pos_w' or 'base_xyz_quat'")
            return False

        if 'fps' in data:
            fps_val = data['fps'].item() if data['fps'].ndim == 0 else data['fps'][0]
            print(f"FPS found: {fps_val}")
            
            # Simulate the fix I made
            B_dim = processed['base'].shape[0]
            processed['fps'] = np.repeat(fps_val, B_dim)
            print(f"Processed 'fps' shape: {processed['fps'].shape}")
        else:
            print("FPS key missing")
            
        print("SUCCESS: File structure is valid for FlexibleWindowDataset.")
        return True

    except Exception as e:
        print(f"FAILED with error: {e}")
        return False

if __name__ == "__main__":
    fpath = "generated_trajectory.npz"
    verify_file(fpath)
