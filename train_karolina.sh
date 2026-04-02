#!/usr/bin/bash
#SBATCH --job-name diffusion-train
#SBATCH --account eu-26-32
#SBATCH --partition qgpu
#SBATCH --time 03:00:00
#SBATCH --gpus 1
#SBATCH --output logs/train_%j.out
#SBATCH --error logs/train_%j.out
uv run python -u train.py --no_tensorboard