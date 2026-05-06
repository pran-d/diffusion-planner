#!/usr/bin/bash
#SBATCH --job-name diffusion-infer
#SBATCH --account eu-26-32
#SBATCH --partition qgpu
#SBATCH --time 00:15:00
#SBATCH --gpus 1
#SBATCH --output logs/infer_%j.out
#SBATCH --error logs/infer_%j.out

export MUJOCO_GL=egl

epoch="${EPOCH:-200}"
if [[ $# -gt 0 && "$1" != -* ]]; then
    epoch="$1"
    shift
fi

cmd=(uv run python inference_mg.py --epoch "$epoch" --ema --video)
cmd+=("$@")
"${cmd[@]}"
