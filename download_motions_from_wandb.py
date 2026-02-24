"""Download all motion artifacts from a wandb project.

This script downloads all motion artifacts from a wandb project and caches them
locally for use with MultiMotionLoader.

Example usage:
    # Download all motions from a wandb project:
    python scripts/download_motions_from_wandb.py \
        --entity ATARITUM \
        --project sbto_v1
"""

from __future__ import annotations

from pathlib import Path

import tyro

from wandb_utils import download_motions_from_wandb


def main(
  entity: str,
  project: str,
  cache_dir: str | None = None,
  artifact_type: str = "motions",
) -> None:
  """Download all motion artifacts from a wandb project.

  Args:
    entity: Wandb entity/username (e.g., 'ATARITUM')
    project: Wandb project name (e.g., 'sbto_v1')
    cache_dir: Directory to cache downloaded artifacts.
      If not specified, uses ~/.cache/mjlab/wandb_motions
    artifact_type: Type of artifact to download (default: 'motions')
  """

  print(f"Downloading motions from wandb: {entity}/{project}")
  print(f"Artifact type: {artifact_type}")
  if cache_dir:
    print(f"Cache directory: {cache_dir}")
  else:
    print("Using default cache directory: ~/.cache/mjlab/wandb_motions")

  cache_dir_path = Path(cache_dir) if cache_dir else None

  try:
    motions_dir = download_motions_from_wandb(
      wandb_entity=entity,
      wandb_project=project,
      cache_dir=cache_dir_path,
      artifact_type=artifact_type,
    )
    print(f"\n✓ Successfully downloaded motions to: {motions_dir}")
  except Exception as e:
    print(f"\n✗ Error downloading motions: {e}")
    raise


if __name__ == "__main__":
  tyro.cli(main, description=__doc__)
