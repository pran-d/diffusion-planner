#!/usr/bin/bash
#SBATCH --job-name diffusion-ablations
#SBATCH --account p201250            # e.g. p200XXXX
#SBATCH --partition gpu
#SBATCH --qos default
#SBATCH --time 2:00:00
#SBATCH --nodes 1
#SBATCH --hint=nomultithread
#SBATCH --gpus 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --output logs/meluxina_ablations_%j.out
#SBATCH --error logs/meluxina_ablations_%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# Usage:
#   sbatch run_ablations_meluxina.sh
#   sbatch run_ablations_meluxina.sh ablations/training_cluster1/masking_modes config/config.yaml
ABLATIONS_FOLDER="${1:-ablations/training_cluster1/masking_modes}"
MAIN_CONFIG="${2:-config/config.yaml}"

if ! command -v module >/dev/null 2>&1; then
  set +u
  [[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh
  [[ -f /usr/share/Modules/init/bash ]] && source /usr/share/Modules/init/bash
  set -u
fi

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load Python/3.11.3-GCCcore-12.3.0 || true
  module load CUDA/12.1.1 || true
else
  echo "[info] Environment Modules not available; skipping module load."
fi

if command -v uv >/dev/null 2>&1; then
  uv run python -u run_ablations.py \
    --ablations_folder "$ABLATIONS_FOLDER" \
    --main_config "$MAIN_CONFIG"
else
  python -u run_ablations.py \
    --ablations_folder "$ABLATIONS_FOLDER" \
    --main_config "$MAIN_CONFIG"
fi
