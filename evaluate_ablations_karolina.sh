#!/usr/bin/bash
#SBATCH --job-name diffusion-eval-ablations
#SBATCH --account eu-26-32
#SBATCH --partition qgpu
#SBATCH --time 02:00:00
#SBATCH --gpus 1
#SBATCH --output logs/eval_ablations_%j.out
#SBATCH --error logs/eval_ablations_%j.out

export MUJOCO_GL=egl    # headless rendering, no display needed

ablations_folder="${ABLATIONS_FOLDER:-config/ablations}"
output_root="${OUTPUT_ROOT:-results/ablation_eval}"
epoch="${EPOCH:-latest}"

cmd=(uv run python -u evaluate_ablations.py
    --ablations_folder "$ablations_folder"
    --output_root "$output_root"
    --epoch "$epoch"
    --ema
    --video)

# Forward any extra argumentss
cmd+=("$@")
"${cmd[@]}"
