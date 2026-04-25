#!/usr/bin/env bash
set -euo pipefail

# Submit two 48h jobs for ablations and chain job2 after successful job1.
# Usage:
#   bash submit_ablations_chained.sh
#   bash submit_ablations_chained.sh ablations/training_cluster1 config/config.yaml
#   bash submit_ablations_chained.sh ablations/training_cluster1 config/config.yaml 48:00:00

cd "$(dirname "$0")"

ABLATIONS_FOLDER="${1:-ablations/training_cluster1}"
MAIN_CONFIG="${2:-config/config.yaml}"
TIME_LIMIT="${3:-48:00:00}"
RETRY_SECONDS="${RETRY_SECONDS:-60}"
MAX_RETRIES="${MAX_RETRIES:-120}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit_ablations_chained.sh is a submit helper."
  echo "Run it from a login shell, not with sbatch:"
  echo "  bash submit_ablations_chained.sh"
  exit 2
fi

if [[ ! -d "$ABLATIONS_FOLDER" ]]; then
  echo "ERROR: ablations folder not found: $ABLATIONS_FOLDER" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  PY_RUNNER=(uv run python)
elif command -v python >/dev/null 2>&1; then
  PY_RUNNER=(python)
elif command -v python3 >/dev/null 2>&1; then
  PY_RUNNER=(python3)
else
  echo "ERROR: no Python interpreter found (tried uv/python/python3)." >&2
  exit 1
fi

TOTAL=$("${PY_RUNNER[@]}" - <<'PY' "$ABLATIONS_FOLDER" "$MAIN_CONFIG"
import os, glob, re, sys
ablations_folder = sys.argv[1]
main_config = os.path.abspath(sys.argv[2])

patterns = [
    os.path.join(ablations_folder, "**", "*.yaml"),
    os.path.join(ablations_folder, "**", "*.yml"),
]
files = []
for p in patterns:
    files.extend(glob.glob(p, recursive=True))
files = sorted(files)

# Deduplicate whitespace-variants, match run_ablations.py behavior
seen = set()
deduped = []
for f in files:
    rel = os.path.relpath(f, ablations_folder)
    key = re.sub(r"\s+", "", rel)
    if key in seen:
        continue
    seen.add(key)
    deduped.append(f)

files = [f for f in deduped if os.path.abspath(f) != main_config]
files = [f for f in files if not f.endswith(".patched.yaml")]
files = [f for f in files if not os.path.basename(f).startswith("abl_train_cfg_")]
print(len(files))
PY
)

if [[ "$TOTAL" -eq 0 ]]; then
  echo "No ablation files found in $ABLATIONS_FOLDER"
  exit 0
fi

SPLIT=$(( (TOTAL + 1) / 2 ))

echo "Total ablations: $TOTAL"
echo "Split point: $SPLIT"

submit_with_retry() {
  local label="$1"
  shift
  local attempt=0
  local out
  local rc

  while true; do
    set +e
    out=$(sbatch --parsable "$@" 2>&1)
    rc=$?
    set -e

    if [[ "$rc" -eq 0 ]]; then
      echo "$out"
      return 0
    fi

    if echo "$out" | grep -Eqi "AssocMaxSubmitJobLimit|job violates accounting/QOS policy"; then
      attempt=$((attempt + 1))
      if [[ "$attempt" -ge "$MAX_RETRIES" ]]; then
        echo "ERROR: could not submit $label after $attempt attempts."
        echo "$out"
        return 2
      fi
      echo "Submit limit/QOS hit while submitting $label. Retry $attempt/$MAX_RETRIES in ${RETRY_SECONDS}s..."
      sleep "$RETRY_SECONDS"
      continue
    fi

    echo "ERROR: failed to submit $label"
    echo "$out"
    return "$rc"
  done
}

JOB1_ID=$(submit_with_retry "job1" \
  --time="$TIME_LIMIT" \
  run_ablations_cluster.sh \
  "$ABLATIONS_FOLDER" "$MAIN_CONFIG" 0 "$SPLIT")

echo "Submitted job1: $JOB1_ID (slice [0:$SPLIT))"

if [[ "$SPLIT" -lt "$TOTAL" ]]; then
  JOB2_ID=$(submit_with_retry "job2" \
    --time="$TIME_LIMIT" \
    --dependency=afterany:"$JOB1_ID" \
    run_ablations_cluster.sh \
    "$ABLATIONS_FOLDER" "$MAIN_CONFIG" "$SPLIT" "$TOTAL")
  echo "Submitted job2: $JOB2_ID (slice [$SPLIT:$TOTAL), dependency=afterany:$JOB1_ID)"
else
  echo "Only one slice needed; no second job submitted."
fi
