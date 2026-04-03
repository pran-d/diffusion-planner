import os
import shutil
import subprocess
import time
import sys
import glob
import re
import copy
import tempfile
import yaml

# =============================================================================
# ABLATION STUDIES CONFIGURATION
# =============================================================================
# Define the root folder containing your ablation configs.
# The script will automatically find all .yaml files in this folder and its subfolders.
ABLATIONS_FOLDER = "config/ablations/mar_12/"

# The main configuration file that will be overwritten during training
MAIN_CONFIG_FILE = "config/config.yaml"

# =============================================================================
# HELPERS
# =============================================================================

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` and return a new dict."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def derive_suffix_from_filename(cfg_path: str) -> str | None:
    """
    Extract the suffix encoded in the config filename.

    Convention: the file is named  config[_<suffix>].yaml
      - config.yaml              → suffix = ""   (bare name, no suffix)
      - config_increased_noise.yaml → suffix = "_increased_noise"
      - config_keeppartial3.yaml    → suffix = "_keeppartial3"

    Returns the suffix string (may be empty), or None if the filename does not
    match the expected  config[_...].yaml  pattern.
    """
    stem = os.path.splitext(os.path.basename(cfg_path))[0]   # e.g. "config_keeppartial3"
    m = re.fullmatch(r'config(_.*)?', stem)
    if m is None:
        return None
    return m.group(1) or ""   # group(1) is None when there is no underscore part


def build_merged_temp_config(base_cfg: dict, override_cfg_path: str, suffix: str | None) -> str:
    """
    Merge the ablation override YAML into base config and optionally force
    ``training.suffix`` from filename-derived ``suffix``.

    This enables ablation files to only define changed values.
    """
    with open(override_cfg_path, "r") as f:
        override = yaml.safe_load(f)

    if override is None:
        override = {}
    if not isinstance(override, dict):
        print("  [warn] Override config is not a YAML mapping; treating as empty override.")
        override = {}

    data = deep_merge(base_cfg, override)

    if suffix is not None:
        training_section = data.setdefault("training", {})
        old_suffix = training_section.get("suffix", "<not set>")
        training_section["suffix"] = suffix

        if old_suffix != suffix:
            print(f"  [patch] training.suffix: '{old_suffix}' → '{suffix}'")
        else:
            print(f"  [patch] training.suffix already matches: '{suffix}' (no change)")

    fd, tmp_path = tempfile.mkstemp(prefix="abl_train_cfg_", suffix=".yaml")
    os.close(fd)
    with open(tmp_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return tmp_path


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

    # Skip any .patched.yaml files left over from a previous interrupted run
    ablation_files = [f for f in ablation_files if not f.endswith(".patched.yaml")]
    # Skip temp merged files created by this runner (if any exist from interruptions)
    ablation_files = [f for f in ablation_files if not os.path.basename(f).startswith("abl_train_cfg_")]

    if not ablation_files:
        print(f"Warning: No '.yaml' files found in '{ABLATIONS_FOLDER}'.")
        sys.exit(0)

    # 1. Backup Current Config
    original_config = MAIN_CONFIG_FILE
    backup_config = f"{MAIN_CONFIG_FILE}.bak"
    
    if os.path.exists(original_config):
        print(f"Backing up {original_config} -> {backup_config}")
        shutil.copy(original_config, backup_config)
    else:
        print(f"ERROR: Main config file '{original_config}' does not exist.")
        sys.exit(1)

    with open(backup_config, "r") as f:
        base_config_data = yaml.safe_load(f)
    if not isinstance(base_config_data, dict):
        print(f"ERROR: Base config '{backup_config}' is not a YAML mapping.")
        sys.exit(1)

    total_start = time.time()

    try:
        for i, cfg_path in enumerate(ablation_files):
            # Extract folder/file name for a cleaner display comment
            relative_name = os.path.relpath(cfg_path, ABLATIONS_FOLDER)

            # Derive suffix from filename and patch it into a temporary copy
            suffix = derive_suffix_from_filename(cfg_path)
            patched_path = None

            print("\n" + "#" * 80)
            print(f"ABLATION {i+1}/{len(ablation_files)}")
            print(f"Target: {relative_name}")
            print(f"Source Config: {cfg_path}")
            if suffix is None:
                print(f"  [warn] Filename does not match 'config[_...].yaml' — "
                      f"using training.suffix from merged config.")
            else:
                print(f"  Derived suffix from filename: '{suffix}'")
            patched_path = build_merged_temp_config(base_config_data, cfg_path, suffix)
            active_cfg = patched_path
            print("#" * 80 + "\n")

            try:
                # 2. Overwrite Main Config with (possibly patched) ablation config
                print(f"Copying {active_cfg} to {original_config}...")
                shutil.copy(active_cfg, original_config)

                # 3. Run Training
                print("Starting training process...")
                cmd = ["python", "train.py"]
                
                start_t = time.time()
                subprocess.run(cmd, check=True)
                end_t = time.time()
                
                print(f"[SUCCESS] Ablation {i+1} completed in {(end_t - start_t)/60:.2f} mins.")

            except subprocess.CalledProcessError as e:
                print(f"[FAILURE] Ablation {i+1} failed with exit code {e.returncode}.")
                time.sleep(5)
            finally:
                # Clean up the temp patched file
                if patched_path and os.path.exists(patched_path):
                    os.remove(patched_path)

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
