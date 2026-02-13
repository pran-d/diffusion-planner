
import numpy as np
import os

def convert_npy_to_npz(npy_path, npz_path, fps=50):
    if not os.path.exists(npy_path):
        print(f"Error: {npy_path} does not exist.")
        return

    print(f"Loading {npy_path}...")
    data = np.load(npy_path)
    
    # Check shape
    # Expected shape from stitch.py is (T, 43) or (1, T, 43) or similar
    if data.ndim == 3:
        # Assuming (Batch, Time, Dim) -> Take first batch
        print(f"Input shape: {data.shape}. Taking first batch.")
        data = data[0]
    
    if data.ndim != 2:
        print(f"Error: Expected 2D array (T, D), got {data.shape}")
        return

    T, D = data.shape
    # D must be >= 43 for the indices we use
    if D < 43:
        print(f"Error: Expected at least 43 dimensions, got {D}")
        return

    print(f"Processing trajectory of length {T}...")

    # Construct dictionary based on stitch.py logic
    # Indices:
    # 0:3   -> body_pos
    # 3:7   -> body_quat
    # 7:36  -> joint_pos (29 joints)
    # 36:39 -> object_pos
    # 39:43 -> object_quat

    save_dict = {
        'body_pos_w': data[:, 0:3][:, None, :],   # (T, 1, 3)
        'body_quat_w': data[:, 3:7][:, None, :],  # (T, 1, 4)
        'joint_pos': data[:, 7:36],               # (T, 29)
        'object_pos_w': data[:, 36:39],           # (T, 3)
        'object_quat_w': data[:, 39:43],          # (T, 4)
        'fps': fps
    }

    print(f"Saving to {npz_path} with FPS {fps}...")
    np.savez(npz_path, **save_dict)
    print("Done.")

if __name__ == "__main__":
    input_file = "./generated_trajectory.npy"
    output_file = "./generated_trajectory.npz"
    convert_npy_to_npz(input_file, output_file, fps=50)
