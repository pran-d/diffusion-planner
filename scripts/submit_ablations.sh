#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="${1:-inference_configs}"
OUTPUT_ROOT="${2:-results/ablations}"

cd "$(dirname "$0")/.."

mkdir -p "$OUTPUT_ROOT"

for cfg in "$CONFIG_DIR"/*.yaml; do
  base_name="$(basename "$cfg")"
  if [[ "$base_name" == "base.yaml" ]]; then
    continue
  fi

  job_name="abl_$(basename "$cfg" .yaml)"
  sbatch \
    --job-name="$job_name" \
    --gres=gpu:1 \
    --mem=16G \
    --cpus-per-task=4 \
    --output="$OUTPUT_ROOT/slurm-%x-%j.out" \
    --wrap="bash scripts/run_single_ablation.sh '$cfg' '$OUTPUT_ROOT'"
done

echo "Submitted ablation jobs for configs in $CONFIG_DIR"
echo "Run compare after completion: python compare_ablations.py --output_root $OUTPUT_ROOT"
