#!/usr/bin/env bash
#SBATCH --job-name=diffusion-ablations
#SBATCH --account=eu-26-32
#SBATCH --partition=qgpu
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/ablations_%j.out
#SBATCH --error=logs/ablations_%j.out

set -euo pipefail

mkdir -p logs

# Usage:
#   sbatch scripts/cluster/run_ablations_cluster.sh
#   sbatch scripts/cluster/run_ablations_cluster.sh ablations/training_cluster1 config/config.yaml
ABLATIONS_FOLDER="${1:-ablations/training_cluster1}"
MAIN_CONFIG="${2:-config/config.yaml}"
START_INDEX="${3:-0}"
END_INDEX="${4:--1}"

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

if command -v uv >/dev/null 2>&1; then
  PY_RUNNER=(uv run python)
elif command -v python >/dev/null 2>&1; then
  PY_RUNNER=(python)
elif command -v python3 >/dev/null 2>&1; then
  PY_RUNNER=(python3)
else
  echo "ERROR: no Python interpreter found (tried uv/python/python3)." >&2
  exit 1
fi

"${PY_RUNNER[@]}" -u run_ablations.py \
  --ablations_folder "$ABLATIONS_FOLDER" \
  --main_config "$MAIN_CONFIG" \
  --start_index "$START_INDEX" \
  --end_index "$END_INDEX"
