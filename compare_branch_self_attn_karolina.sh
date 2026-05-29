#!/usr/bin/bash
#SBATCH --job-name diffusion-self-attn-cmp
#SBATCH --account eu-26-32
#SBATCH --partition qgpu
#SBATCH --time 00:30:00
#SBATCH --gpus 1
#SBATCH --output logs/self_attn_cmp_%j.out
#SBATCH --error logs/self_attn_cmp_%j.out

export MUJOCO_GL=egl    # headless rendering, no display needed

output_dir="${OUTPUT_DIR:-results/attention/branch_self_attn_cmp}"
epoch="${EPOCH:-latest}"
num_traj="${NUM_TRAJ:-8}"

cmd=(uv run python -u compare_branch_self_attn.py
    --output_dir "$output_dir"
    --epoch "$epoch"
    --num_trajectories "$num_traj"
    --ema)

# Forward any extra arguments
cmd+=("$@")
"${cmd[@]}"
