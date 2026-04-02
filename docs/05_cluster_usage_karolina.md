# IT4I Karolina quick usage guide (for this project)

This short guide focuses on day-to-day usage for training and inference ablations on Karolina.

## 1) Connect to the cluster

Use SSH with your key (IT4I supports RSA/ED25519 keys):

```bash
ssh -i ~/.ssh/id_ed25519 <username>@karolina.it4i.cz
```

You can also target a specific login node:

```bash
ssh -i ~/.ssh/id_ed25519 <username>@login1.karolina.it4i.cz
```

## 2) Go to project and submit jobs

```bash
cd ~/diffusion-planner
sbatch train_karolina.sh
```

For inference ablations (batch):

```bash
sbatch --job-name=diffusion-abl --partition=qgpu --gpus=1 --time=02:00:00 \
  --output=logs/abl_%j.out --error=logs/abl_%j.out \
  --wrap='cd ~/diffusion-planner && ./scripts/run_inference_ablations.sh --include-base'
```

## 3) Check what is running

### Your jobs (running + pending)

```bash
squeue -u $USER
```

### Running jobs only

```bash
squeue -u $USER -t RUNNING
```

### Pending jobs only

```bash
squeue -u $USER -t PENDING
```

### Detailed info about one job

```bash
scontrol show job <JOBID>
```

### Finished jobs (history)

```bash
sacct -u $USER --starttime today --format=JobID,JobName,Partition,State,Elapsed,ExitCode
```

## 4) Watch logs in real time

```bash
tail -f logs/train_<JOBID>.out
```

or for ablations:

```bash
tail -f logs/abl_<JOBID>.out
```

## 5) Cancel jobs

```bash
scancel <JOBID>
```

Cancel all your pending jobs:

```bash
scancel -u $USER -t PENDING
```

## 6) Interactive GPU session (quick debug)

```bash
salloc --partition=qgpu --gpus=1 --time=01:00:00 --account=eu-26-32
```

Then run:

```bash
cd ~/diffusion-planner
uv run python train.py
```

## 7) Data transfer basics

Upload to cluster:

```bash
scp -i ~/.ssh/id_ed25519 -r ./diffusion-planner <username>@karolina.it4i.cz:~
```

Download results:

```bash
scp -i ~/.ssh/id_ed25519 -r <username>@karolina.it4i.cz:~/diffusion-planner/results ./results_from_cluster
```

## 8) Commands used most often in this repo

Train:

```bash
sbatch train_karolina.sh
```

Run ablations directly (inside allocation or interactive shell):

```bash
./scripts/run_inference_ablations.sh --include-base
```

Run one ablation config:

```bash
./scripts/run_inference_ablations.sh --single inference_configs/ddim_3steps.yaml
```

---

## Notes from IT4I docs

- Cluster login is through SSH login nodes (e.g., `karolina.it4i.cz`).
- Key-based authentication is required.
- Common transfer tools: `scp`, `sftp`, `rsync`, `sshfs`.

Reference: IT4I Accessing the Clusters documentation.
