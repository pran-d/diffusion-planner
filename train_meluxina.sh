#!/usr/bin/bash
#SBATCH --job-name diffusion-train
#SBATCH --account p201250            # e.g. p200XXXX
#SBATCH --partition gpu
#SBATCH --qos default                     # 48hr limit; use "test" for quick debug runs
#SBATCH --time 2:00:00
#SBATCH --nodes 1
#SBATCH --hint=nomultithread              # use physical cores only (recommended)
#SBATCH --output logs/train_%j.out
#SBATCH --error logs/train_%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# Optional: module setup (adjust to site defaults)
if ! command -v module >/dev/null 2>&1; then
  # Some site scripts expect unset vars (e.g., MODULEPATH) and fail with `set -u`.
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

# Preferred runtime
if command -v uv >/dev/null 2>&1; then
  uv run python -u train.py --no_tensorboard
else
  python -u train.py --no_tensorboard
fi
