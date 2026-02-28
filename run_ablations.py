import os
import shutil
import subprocess
import time
import sys
import glob

# =============================================================================
# ABLATION STUDIES CONFIGURATION
# =============================================================================
# Define the root folder containing your ablation configs.
# The script will automatically find all .yaml files in this folder and its subfolders.
ABLATIONS_FOLDER = "config/ablations/masking/"

# The main configuration file that will be overwritten during training
MAIN_CONFIG_FILE = "config/config.yaml"

# =============================================================================
# SCRIPT LOGIC
# =============================================================================

def main():
    if not os.path.isdir(ABLATIONS_FOLDER):
        print(f"ERROR: The directory '{ABLATIONS_FOLDER}' does not exist.")
        sys.exit(1)

    # Recursively find all .yaml files in the ablations folder
    search_pattern = os.path.join(ABLATIONS_FOLDER, "**/*.yaml")
    ablation_files = glob.glob(search_pattern, recursive=True)
    
    # Sort them alphabetically so they run in a predictable order
    ablation_files = sorted(ablation_files)

    # Filter out the main config file just in case it is inside the search folder
    ablation_files = [f for f in ablation_files if os.path.abspath(f) != os.path.abspath(MAIN_CONFIG_FILE)]

    if not ablation_files:
        print(f"Warning: No '.yaml' files found in '{ABLATIONS_FOLDER}'.")
        sys.exit(0)

    # 1. Backup Current Config
    original_config = MAIN_CONFIG_FILE
    backup_config = f"{MAIN_CONFIG_FILE}.bak"
    
    if os.path.exists(original_config):
        print(f"Backing up {original_config} -> {backup_config}")
        shutil.copy(original_config, backup_config)

    total_start = time.time()

    try:
        for i, cfg_path in enumerate(ablation_files):
            # Extract folder/file name for a cleaner display comment
            relative_name = os.path.relpath(cfg_path, ABLATIONS_FOLDER)
            
            print("\n" + "#" * 80)
            print(f"ABLATION {i+1}/{len(ablation_files)}")
            print(f"Target: {relative_name}")
            print(f"Source Config: {cfg_path}")
            print("#" * 80 + "\n")

            # 2. Overwrite Main Config
            print(f"Copying {cfg_path} to {original_config}...")
            shutil.copy(cfg_path, original_config)

            # 3. Run Training
            print("Starting training process...")
            cmd = ["python", "train.py"]
            
            try:
                # Using subprocess.run to wait for completion
                start_t = time.time()
                subprocess.run(cmd, check=True)
                end_t = time.time()
                
                print(f"[SUCCESS] Ablation {i+1} completed in {(end_t - start_t)/60:.2f} mins.")

            except subprocess.CalledProcessError as e:
                print(f"[FAILURE] Ablation {i+1} failed with exit code {e.returncode}.")
                # Buffer time before starting the next one if it fails
                time.sleep(5)

    except KeyboardInterrupt:
        print("\n[STOPPED] Execution interrupted by user.")

    finally:
        # 4. Restore Original Config
        if os.path.exists(backup_config):
            print(f"\nRestoring original config from {backup_config}...")
            shutil.copy(backup_config, original_config)
            os.remove(backup_config)

        total_time = time.time() - total_start
        print(f"\nAll tasks finished. Total time: {total_time/3600:.2f} hours.")

if __name__ == "__main__":
    main()