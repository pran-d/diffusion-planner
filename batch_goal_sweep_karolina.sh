#!/usr/bin/bash
#SBATCH --job-name diffusion-train
#SBATCH --account eu-26-32
#SBATCH --partition qgpu
#SBATCH --time 00:30:00
#SBATCH --gpus 1
#SBATCH --output logs/sweep_%j.out
#SBATCH --error logs/sweep_%j.out

export MUJOCO_GL=egl    # headless rendering, no display needed

# Epoch can be provided as:
#   1) first positional script arg (if it is not an option),
#   2) EPOCH env var,
#   3) fallback default 200.
epoch="${EPOCH:-200}"
if [[ $# -gt 0 && "$1" != -* ]]; then
	epoch="$1"
	shift
fi

# Enable video export by setting ENABLE_VIDEO=1
cmd=(uv run python batch_goal_sweep.py --video --epoch "$epoch")

# Forward any extra arguments passed to this script
cmd+=("$@")
"${cmd[@]}"