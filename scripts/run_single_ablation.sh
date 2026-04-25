#!/usr/bin/env bash
set -euo pipefail

CFG_PATH="${1:?Usage: run_single_ablation.sh <ablation-yaml-path>}"
OUTPUT_ROOT="${2:-results/ablations}"

cd "$(dirname "$0")/.."

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python run_inference_ablations.py \
  --single_config "$CFG_PATH" \
  --base_config inference_configs/base.yaml \
  --output_dir "$OUTPUT_ROOT"
