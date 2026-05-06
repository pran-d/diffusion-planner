#!/usr/bin/bash
#SBATCH --job-name diffusion-train
#SBATCH --account eu-26-32
#SBATCH --partition qgpu
#SBATCH --time 24:00:00
#SBATCH --gpus 1
#SBATCH --output logs/train_%j.out
#SBATCH --error logs/train_%j.out

conf_path="${CFGPATH:-config/config.yaml}"
cfgpath="--config $conf_path"

cmd=(uv run python -u train.py --config "$cfgpath" --no_tensorboard)

cmd+=("$@")
"${cmd[@]}"