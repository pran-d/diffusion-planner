"""WandB utilities."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Sequence

import wandb
from tqdm import tqdm


def download_motions_from_wandb(
  wandb_entity: str,
  wandb_project: str,
  cache_dir: Path | str | None = None,
  artifact_type: str = "motions",
) -> Path:
  """Download motion artifacts from wandb registry.

  Downloads all artifacts of the specified type from all runs in the wandb project
  and organizes them in a directory structure compatible with MultiMotionLoader.

  Args:
      wandb_entity: Wandb entity/username (e.g., "ATARITUM")
      wandb_project: Wandb project name (e.g., "sbto-v2-motions")
      cache_dir: Directory to cache downloaded artifacts. If None, uses ~/.cache/mjlab/wandb_motions
      artifact_type: Type of artifact to download (default: "motions")

  Returns:
      Path to the directory containing downloaded motions organized as:
      cache_dir/{project}/motions/{run_name}/{artifact_name}/motion.npz
  """
  # Set up cache directory
  if cache_dir is None:
    cache_dir = Path.home() / ".cache" / "mjlab" / "wandb_motions"
  else:
    cache_dir = Path(cache_dir)

  project_cache_dir = cache_dir / wandb_project
  motions_dir = project_cache_dir / "motions"
  motions_dir.mkdir(parents=True, exist_ok=True)

  # Load cached version metadata
  metadata_file = project_cache_dir / "artifact_versions.json"
  cached_versions = {}
  if metadata_file.exists():
    try:
      with open(metadata_file, "r") as f:
        cached_versions = json.load(f)
    except Exception:
      cached_versions = {}

  # Check what's already in cache
  cached_motions = {}
  if motions_dir.exists():
    for motion_file in motions_dir.rglob("motion.npz"):
      # Get relative path from motions_dir to determine artifact name
      rel_path = motion_file.relative_to(motions_dir)
      # For registry artifacts: artifact_name/motion.npz
      # For run artifacts: run_name/artifact_name/motion.npz
      if len(rel_path.parts) == 2:
        artifact_name = rel_path.parts[0]
        cached_motions[artifact_name] = motion_file
      elif len(rel_path.parts) == 3:
        # Run-based artifacts: run_name/artifact_name/motion.npz
        run_name, artifact_name = rel_path.parts[0], rel_path.parts[1]
        key = f"{run_name}/{artifact_name}"
        cached_motions[key] = motion_file

  # Initialize wandb API
  if wandb is None:
    raise ImportError("wandb is required but not available")
  api = wandb.Api()

  downloaded_count = 0
  skipped_count = 0

  # Try to get artifacts from registry first
  try:
    # Get all runs from the project and collect artifact names from them
    print(f"Collecting artifacts from {wandb_entity}/{wandb_project}...")
    runs = api.runs(f"{wandb_entity}/{wandb_project}")
    artifact_names = set()

    # Convert to list to get total count for progress bar
    runs_list = list(runs)
    for run in tqdm(runs_list, desc="Scanning runs", unit="run"):
      # Check both used_artifacts and logged_artifacts
      for artifact in list(run.used_artifacts()) + list(run.logged_artifacts()):
        if artifact.type == artifact_type:
          artifact_names.add(artifact.name)

    if not artifact_names:
      print(f"No artifacts found in {wandb_entity}/{wandb_project}")
      return motions_dir

    print(f"Found {len(artifact_names)} artifacts to check")
    # Get latest version of each artifact and download if needed
    pbar = tqdm(sorted(artifact_names), desc="Downloading artifacts", unit="artifact")
    for artifact_name in pbar:
      pbar.set_postfix(
        {
          "current": artifact_name[:30],
          "downloaded": downloaded_count,
          "skipped": skipped_count,
        }
      )

      # Get the latest version of this artifact
      try:
        latest_artifact = api.artifact(
          f"{wandb_entity}/{wandb_project}/{artifact_name}:latest"
        )
      except Exception as e:
        # If latest alias doesn't exist, try to get any version
        try:
          latest_artifact = api.artifact(
            f"{wandb_entity}/{wandb_project}/{artifact_name}"
          )
        except Exception:
          print(f"Warning: Could not fetch artifact {artifact_name}: {e}")
          continue

      # Get version info for comparison
      latest_version = latest_artifact.version
      # Convert updated_at to string (handles both datetime and string types)
      updated_at = latest_artifact.updated_at
      try:
        # Try to call isoformat if it's a datetime object
        latest_updated = updated_at.isoformat()  # type: ignore
      except (AttributeError, TypeError):
        # Fall back to string conversion
        latest_updated = str(updated_at)

      # Use artifact name as the trajectory name (since it's the collection name)
      # Create directory: {artifact_name}
      artifact_dir = motions_dir / artifact_name
      motion_file = artifact_dir / "motion.npz"

      # Check if we need to re-download
      cached_version = cached_versions.get(artifact_name, {})
      cached_version_str = cached_version.get("version")
      cached_updated = cached_version.get("updated_at")

      # Re-download if version changed or if file doesn't exist
      needs_download = (
        cached_version_str != latest_version
        or cached_updated != latest_updated
        or not motion_file.exists()
      )

      if not needs_download:
        skipped_count += 1
        pbar.set_postfix(
          {
            "current": artifact_name[:30],
            "downloaded": downloaded_count,
            "skipped": skipped_count,
          }
        )
        continue

      try:
        # Remove old version if it exists
        if artifact_dir.exists():
          shutil.rmtree(artifact_dir)

        # Download artifact to temporary directory
        artifact_dir.mkdir(parents=True, exist_ok=True)
        temp_download_dir = artifact_dir / "temp"
        temp_download_dir.mkdir(exist_ok=True)

        # Download with progress indication
        artifact_path = latest_artifact.download(root=str(temp_download_dir))

        # Check if motion.npz is in the artifact root or a subdirectory
        artifact_path_obj = Path(artifact_path)
        motion_file_candidate = artifact_path_obj / "motion.npz"

        found_motion = False
        if motion_file_candidate.exists():
          # Move to expected location
          shutil.move(str(motion_file_candidate), str(motion_file))
          found_motion = True
        else:
          # Look for motion.npz in subdirectories
          for motion_candidate in artifact_path_obj.rglob("motion.npz"):
            # Move to expected location
            shutil.move(str(motion_candidate), str(motion_file))
            found_motion = True
            break

        if not found_motion:
          # Clean up empty directory
          if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
          continue

        # Clean up temporary download directory
        if temp_download_dir.exists():
          shutil.rmtree(temp_download_dir)

        # Save version metadata
        cached_versions[artifact_name] = {
          "version": latest_version,
          "updated_at": latest_updated,
        }

        downloaded_count += 1
        pbar.set_postfix(
          {
            "current": artifact_name[:30],
            "downloaded": downloaded_count,
            "skipped": skipped_count,
          }
        )

      except Exception as e:
        # Clean up on error
        if artifact_dir.exists():
          shutil.rmtree(artifact_dir)
        print(f"Warning: Error downloading {artifact_name}: {e}")
        continue

  except Exception as e:
    print(f"Warning: Could not access registry: {e}")

  # Save version metadata
  if cached_versions:
    try:
      with open(metadata_file, "w") as f:
        json.dump(cached_versions, f, indent=2)
    except Exception as e:
      print(f"Warning: Could not save version metadata: {e}")

  updated_count = downloaded_count
  print(
    f"\nDownloaded/updated {updated_count} artifacts with latest versions, {skipped_count} already up-to-date"
  )

  # Verify we have at least one motion file
  motion_files = list(motions_dir.rglob("motion.npz"))
  print(f"Total number of motions: {len(motion_files)}")
  if not motion_files:
    raise RuntimeError(
      f"No motion.npz files found in {motions_dir}. "
      f"Please check that artifacts are properly uploaded to wandb."
    )

  # Clean up old versions after downloading
  cleanup_old_versions(motions_dir)

  return motions_dir


def cleanup_old_versions(motions_dir: Path | str) -> dict[str, int]:
  """Remove old versions of artifacts, keeping only the latest version.

  This function scans the motions directory and identifies folders with version tags
  (e.g., `:v0`, `:v1`, `:v2`) in their names. For each base artifact name, it keeps
  only the folder with the highest version number and deletes the others.

  Args:
      motions_dir: Directory containing motion artifacts organized as:
                  motions_dir/{artifact_name}/motion.npz or
                  motions_dir/{run_name}/{artifact_name}/motion.npz

  Returns:
      Dictionary mapping base artifact names to the number of old versions deleted
  """
  motions_dir = Path(motions_dir)
  if not motions_dir.exists():
    return {}

  # Pattern to match version tags like :v0, :v1, :v2, etc.
  version_pattern = re.compile(r":v(\d+)$")

  # Group folders by base name (without version tag)
  versioned_folders: dict[str, list[tuple[Path, int]]] = {}

  # Scan all directories that contain motion.npz files
  for motion_file in motions_dir.rglob("motion.npz"):
    rel_path = motion_file.relative_to(motions_dir)
    folder_path = motion_file.parent

    if len(rel_path.parts) == 2:
      # Format: artifact_name/motion.npz
      folder_name = rel_path.parts[0]
      match = version_pattern.search(folder_name)
      if match:
        version_num = int(match.group(1))
        base_name = folder_name[: match.start()]
        if base_name not in versioned_folders:
          versioned_folders[base_name] = []
        versioned_folders[base_name].append((folder_path, version_num))
    elif len(rel_path.parts) == 3:
      # Format: run_name/artifact_name/motion.npz
      run_name, artifact_name = rel_path.parts[0], rel_path.parts[1]
      match = version_pattern.search(artifact_name)
      if match:
        version_num = int(match.group(1))
        base_name = artifact_name[: match.start()]
        # Use run_name/base_name as the key to group by both
        key = f"{run_name}/{base_name}"
        if key not in versioned_folders:
          versioned_folders[key] = []
        versioned_folders[key].append((folder_path, version_num))

  deleted_count = {}
  total_deleted = 0

  # For each base name, keep only the highest version
  for base_name, folders in versioned_folders.items():
    if len(folders) <= 1:
      continue  # Only one version, nothing to clean up

    # Sort by version number (descending)
    folders.sort(key=lambda x: x[1], reverse=True)
    latest_folder, latest_version = folders[0]

    # Delete all older versions
    deleted = 0
    for folder_path, _version_num in folders[1:]:
      try:
        if folder_path.exists():
          shutil.rmtree(folder_path)
          deleted += 1
          total_deleted += 1
      except Exception as e:
        print(f"Warning: Could not delete {folder_path}: {e}")

    if deleted > 0:
      deleted_count[base_name] = deleted
      print(
        f"Cleaned up {deleted} old version(s) of '{base_name}', kept v{latest_version}"
      )

  if deleted_count:
    print(f"\nTotal: Removed {total_deleted} old version(s), kept latest versions")
  else:
    print("No old versions to clean up")

  return deleted_count


def get_wandb_motion_cache_dir(
  wandb_entity: str,
  wandb_project: str,
  cache_dir: Path | str | None = None,
) -> Path:
  """Get the cache directory path for wandb motions without downloading.

  This is useful for checking if motions are already cached or for setting
  motion_dir in configs before initialization.

  Args:
      wandb_entity: Wandb entity/username (e.g., "ATARITUM")
      wandb_project: Wandb project name (e.g., "sbto_v1")
      cache_dir: Base cache directory. If None, uses ~/.cache/mjlab/wandb_motions

  Returns:
      Path to the motions directory in cache (may not exist yet)
  """
  if cache_dir is None:
    cache_dir = Path.home() / ".cache" / "mjlab" / "wandb_motions"
  else:
    cache_dir = Path(cache_dir)

  project_cache_dir = cache_dir / wandb_project
  motions_dir = project_cache_dir / "motions"
  return motions_dir


def get_wandb_entity_and_project(
  wandb_entity_project: str,
) -> tuple[str | None, str | None]:
  """Get wandb entity and project from combined format."""
  if wandb_entity_project:
    parts = wandb_entity_project.split("/", 1)
    if len(parts) == 2:
      return parts[0], parts[1]
    else:
      raise ValueError(
        f"wandb_entity_project must be in format 'entity/project', got: {wandb_entity_project}"
      )
  return None, None


def add_wandb_tags(tags: Sequence[str]) -> None:
  """Add tags to the current wandb run.

  Note: This function stores tags in wandb.config._wandb_tags if the run is not yet
  initialized, allowing them to be retrieved later. If the run is already initialized,
  tags are added directly.
  """
  if not tags:
    return

  try:
    import wandb

    if wandb.run is not None:
      existing_tags = list(wandb.run.tags) if wandb.run.tags else []
      new_tags = list(set(existing_tags + list(tags)))
      wandb.run.tags = new_tags
    else:
      # Store tags to be added when run is initialized.
      # This is a workaround for lazy wandb initialization in rsl_rl 3.1.0.
      current_tags = os.environ.get("WANDB_TAGS", "")
      all_tags = set(current_tags.split(",") if current_tags else [])
      all_tags.update(tags)
      os.environ["WANDB_TAGS"] = ",".join(sorted(all_tags))
  except ImportError:
    pass
