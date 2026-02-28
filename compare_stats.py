import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

def extract_index(filepath):
    """Extracts the integer index from the filename for proper sorting."""
    filename = os.path.basename(filepath)
    match = re.search(r'norm_stats_(\d+)\.npz', filename)
    return int(match.group(1)) if match else -1

def main():
    folder_path = "/home/pranish/Downloads/dfot_ts10_f50_trajectory_goal/" # <-- UPDATE THIS PATH
    
    # 1. Find and sort all relevant .npz files
    search_pattern = os.path.join(folder_path, "norm_stats_*.npz")
    file_paths = glob.glob(search_pattern)
    
    if not file_paths:
        print(f"No files matching 'norm_stats_*.npz' found in {folder_path}")
        return

    # Sort numerically based on the index in the filename
    file_paths = sorted(file_paths, key=extract_index)
    indices = [extract_index(f) for f in file_paths]
    
    print(f"Found {len(file_paths)} files. Indices: {indices}")

    # 2. Load data and track history for each key
    stats_history = {}
    
    for path in file_paths:
        data = np.load(path)
        for key in data.files:
            if key not in stats_history:
                stats_history[key] = []
            stats_history[key].append(data[key])
            
    # 3. Compare and Plot the changes
    print("\n--- Stat Changes (L2 Norm of difference from baseline) ---")
    
    # Create a figure to plot the trends
    num_keys = len(stats_history.keys())
    fig, axes = plt.subplots(num_keys, 1, figsize=(10, 4 * num_keys))
    if num_keys == 1:
        axes = [axes]
        
    for ax, (key, history) in zip(axes, stats_history.items()):
        # Convert history list to a 2D numpy array: (num_files, feature_dim)
        history_arr = np.array(history) 
        
        # Baseline is the first file (index 0)
        baseline = history_arr[0]
        
        # Calculate how much the stats have drifted from the baseline
        # Using Mean Absolute Error (MAE) or L2 Norm across features
        drift = np.linalg.norm(history_arr - baseline, axis=1)
        
        # Print tabular summary
        print(f"Key: '{key}' | Shape: {baseline.shape}")
        print(f"  Max drift from baseline: {np.max(drift):.6f}")
        
        # Plotting
        ax.plot(indices, drift, marker='o', linestyle='-', linewidth=2)
        ax.set_title(f"Drift of '{key}' relative to norm_stats_{indices[0]}")
        ax.set_xlabel("Stat File Index")
        ax.set_ylabel("L2 Norm Difference")
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("stat_drift_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()