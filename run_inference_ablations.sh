#!/usr/bin/bash
#SBATCH --job-name diffusion-inference-ablations
#SBATCH --account eu-26-32
#SBATCH --partition qgpu
#SBATCH --time 01:00:00
#SBATCH --gpus 1
#SBATCH --output logs/inference_ablations_%j.out
#SBATCH --error logs/inference_ablations_%j.out

# Usage:
#   scripts/run_inference_ablations.sh
#   scripts/run_inference_ablations.sh --single inference_configs/ddim_3steps.yaml
#   scripts/run_inference_ablations.sh --output results/ablations_gpu0 --cuda 0 --skip-compare

CONFIG_DIR="inference_configs"
BASE_CONFIG="inference_configs/base.yaml"
OUTPUT_DIR="results/ablations"
SINGLE_CONFIG=""
TEST_INDICES=""
SKIP_COMPARE=0
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
INCLUDE_BASE=0
SAVE_VIDEOS=0
VIDEOS_PER_ABLATION=1
VIDEO_FPS=100
VIDEO_WIDTH=1280
VIDEO_HEIGHT=720

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-dir)
      CONFIG_DIR="$2"
      shift 2
      ;;
    --base)
      BASE_CONFIG="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --single)
      SINGLE_CONFIG="$2"
      shift 2
      ;;
    --test-indices)
      TEST_INDICES="$2"
      shift 2
      ;;
    --cuda)
      CUDA_DEVICE="$2"
      shift 2
      ;;
    --skip-compare)
      SKIP_COMPARE=1
      shift
      ;;
    --include-base)
      INCLUDE_BASE=1
      shift
      ;;
    --save-videos)
      SAVE_VIDEOS=1
      shift
      ;;
    --videos-per-ablation)
      VIDEOS_PER_ABLATION="$2"
      shift 2
      ;;
    --video-fps)
      VIDEO_FPS="$2"
      shift 2
      ;;
    --video-width)
      VIDEO_WIDTH="$2"
      shift 2
      ;;
    --video-height)
      VIDEO_HEIGHT="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Run inference ablations.

Options:
  --config-dir <dir>      Directory with ablation YAMLs (default: inference_configs)
  --base <path>           Base YAML config (default: inference_configs/base.yaml)
  --output <dir>          Output root (default: results/ablations)
  --single <path>         Run only one ablation YAML
  --test-indices <path>   Reuse fixed test indices .npy file
  --cuda <id(s)>          CUDA_VISIBLE_DEVICES value (default: 0)
  --skip-compare          Skip compare_ablations.py at the end
  --include-base          Include base.yaml in multi-config runs
  --save-videos           Save MuJoCo MP4 videos per ablation
  --videos-per-ablation N Number of sample videos per ablation (default: 1)
  --video-fps N           Video FPS (default: 100)
  --video-width N         Video width (default: 1280)
  --video-height N        Video height (default: 720)
  -h, --help              Show this help
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Use --help for usage."
      exit 1
      ;;
  esac
done

# In SLURM, job scripts may execute from /var/spool/slurmd; prefer submit dir.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "$SCRIPT_DIR/pyproject.toml" ]]; then
    cd "$SCRIPT_DIR"
  else
    cd "$SCRIPT_DIR/.."
  fi
fi

if [[ ! -f "run_inference_ablations.py" ]]; then
  echo "Error: run_inference_ablations.py not found in $(pwd)"
  exit 1
fi

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

CMD=(uv run python -u run_inference_ablations.py --config_dir "$CONFIG_DIR" --base_config "$BASE_CONFIG" --output_dir "$OUTPUT_DIR")

if [[ -n "$SINGLE_CONFIG" ]]; then
  CMD+=(--single_config "$SINGLE_CONFIG")
fi

if [[ -n "$TEST_INDICES" ]]; then
  CMD+=(--test_indices "$TEST_INDICES")
fi

if [[ "$SKIP_COMPARE" -eq 1 ]]; then
  CMD+=(--skip_compare)
fi

if [[ "$INCLUDE_BASE" -eq 1 ]]; then
  CMD+=(--include_base)
fi

if [[ "$SAVE_VIDEOS" -eq 1 ]]; then
  CMD+=(--save_videos)
  CMD+=(--videos_per_ablation "$VIDEOS_PER_ABLATION")
  CMD+=(--video_fps "$VIDEO_FPS")
  CMD+=(--video_width "$VIDEO_WIDTH")
  CMD+=(--video_height "$VIDEO_HEIGHT")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
