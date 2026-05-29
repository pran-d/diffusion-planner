#!/usr/bin/bash
#SBATCH --job-name diffusion-ablation
#SBATCH --account eu-26-32
#SBATCH --partition qgpu
#SBATCH --time 24:00:00
#SBATCH --gpus 1
#SBATCH --array 0-1
#SBATCH --output logs/ablation_%A_%a.out
#SBATCH --error logs/ablation_%A_%a.out

ablations_folder="${ABLATIONS_FOLDER:-config/ablations}"

start=$SLURM_ARRAY_TASK_ID
end=$((SLURM_ARRAY_TASK_ID + 1))

uv run python -u run_ablations.py \
    --ablations_folder "$ablations_folder" \
    --start_index "$start" \
    --end_index "$end" \
    "$@"
