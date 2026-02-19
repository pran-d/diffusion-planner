import os
import shutil
import subprocess
import time
import sys

# =============================================================================
# ABLATION STUDIES CONFIGURATION
# =============================================================================
# Define your planned ablations here.
# format: ( "path/to/specific_config.yaml", "Description of what this ablation tests" )

ABLATIONS = [

    (
        "config/ablations/timesteps/config_1.yaml",
        "1 time-step in future"
    ),

    (
        "config/ablations/timesteps/config_5.yaml",
        "5 time-steps in future"
    ),

    (
        "config/ablations/timesteps/config_10.yaml",
        "10 time-steps in future"
    ),

    (
        "config/ablations/timesteps/config_20.yaml",
        "20 time-steps in future"
    ),

]
# =============================================================================
# SCRIPT LOGIC
# =============================================================================

def main():
    if not ABLATIONS:
        print("Warning: No ablations defined in 'ABLATIONS' list inside run_ablations.py.")
        print("Please edit the file and uncomment/add entries.")
        sys.exit(0)

    # 1. Backup Current Config
    original_config = "config/config.yaml"
    backup_config = "config/config.yaml.bak"
    
    if os.path.exists(original_config):
        print(f"Backing up {original_config} -> {backup_config}")
        shutil.copy(original_config, backup_config)

    try:
        total_start = time.time()
        
        for i, (cfg_path, comment) in enumerate(ABLATIONS):
            print("\n" + "#" * 80)
            print(f"ABLATION {i+1}/{len(ABLATIONS)}")
            print(f"Comment: {comment}")
            print(f"Source Config: {cfg_path}")
            print("#" * 80 + "\n")

            if not os.path.exists(cfg_path):
                print(f"ERROR: Config file not found at {cfg_path}")
                print("Skipping...")
                continue

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
                # Decide here if you want to break or continue. Usually continue for batch jobs.
                # buffer time
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
