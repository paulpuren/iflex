#!/usr/bin/env bash
# Evaluate the selected full FLEX checkpoint on every NSKT Reynolds number for
# the existing held-out trajectory seed 3407.
#
# Submit with:
#   sbatch scripts/nskt/slurm_eval_full_flex_seed3407.sh
#SBATCH --job-name=flex-all-re-s3407
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/flex-all-re-s3407-%j.out
#SBATCH --error=logs/flex-all-re-s3407-%j.err
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

export MODEL=flex
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/nskt/flex3_best.pt}"
export TEST_SEED=3407
export NSKT_EVAL_MAX_SAMPLES="${NSKT_EVAL_MAX_SAMPLES:-256}"
export CONFIG_PATH=config/nskt/flex_posttrain.yaml
export TOTAL_INTERP_STEPS=20
export BATCH_SIZE="${BATCH_SIZE:-12}"
export EXTRA_ARGS="--epochs 300 --is_T_fixed false"

test -f "${CHECKPOINT_PATH}" || {
  echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 2
}

# Run four Reynolds-number evaluations concurrently, one process per GPU.
# Process-local CUDA_VISIBLE_DEVICES makes the selected GPU appear as cuda:0,
# which matches eval.py's device handling.
RE_IDS=(0 1 2 3 4 5 6 7 8 9)
for batch_start in 0 4 8; do
  pids=()
  for gpu_index in 0 1 2 3; do
    re_index=$((batch_start + gpu_index))
    if ((re_index >= ${#RE_IDS[@]})); then
      break
    fi
    re_id="${RE_IDS[re_index]}"
    echo "Launching re_id=${re_id}, seed=${TEST_SEED} on GPU ${gpu_index}"
    CUDA_VISIBLE_DEVICES="${gpu_index}" RE_ID="${re_id}" \
      bash scripts/nskt/eval_flex_nskt.sh &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
done
