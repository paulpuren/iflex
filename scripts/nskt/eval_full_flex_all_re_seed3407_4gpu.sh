#!/usr/bin/env bash
# Run on a compute node where four GPUs are already allocated.
# This script does not submit or allocate a Slurm job.
set -euo pipefail
module load python
conda activate "${CONDA_ENV_PATH:-/pscratch/sd/k/kenzhong/conda_env/genai}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

export CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/checkpoint_Model_ft_FLEX3_small_mlp2_Data_nskt_Optim_adam_cosine_lr0.0001_epoch300_patchsize256_stride32_T20_TfixedFalse_best.pt}"
export CONFIG_PATH="${CONFIG_PATH:-config/nskt/flex_posttrain.yaml}"
export TEST_SEED="${TEST_SEED:-3407}"
export TOTAL_INTERP_STEPS="${TOTAL_INTERP_STEPS:-20}"
export BATCH_SIZE="${BATCH_SIZE:-12}"
export NSKT_EVAL_MAX_SAMPLES="${NSKT_EVAL_MAX_SAMPLES:-256}"
export EXTRA_ARGS="${EXTRA_ARGS:---epochs 300 --is_T_fixed false}"

test -f "${CHECKPOINT_PATH}" || {
  echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 2
}

if ! command -v nvidia-smi >/dev/null || (( $(nvidia-smi -L | wc -l) < 4 )); then
  echo "This script requires a compute node with four visible GPUs." >&2
  exit 2
fi

mkdir -p logs

# data_nskt_updated.py mapping:
# 0:600, 1:1000, 2:2000, 3:4000, 4:8000,
# 5:12000, 6:16000, 7:24000, 8:32000, 9:36000.
RE_IDS=(0 1 2 3 4 5 6 7 8 9)

for batch_start in 0 4 8; do
  pids=()
  for gpu_id in 0 1 2 3; do
    index=$((batch_start + gpu_id))
    if ((index >= ${#RE_IDS[@]})); then
      break
    fi

    re_id="${RE_IDS[index]}"
    log_path="logs/eval_full_flex_reid${re_id}_seed${TEST_SEED}.log"
    echo "Launching re_id=${re_id}, seed=${TEST_SEED} on GPU ${gpu_id}; log=${log_path}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" RE_ID="${re_id}" \
      bash scripts/nskt/eval_flex_nskt.sh >"${log_path}" 2>&1 &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if ((failed)); then
    echo "At least one evaluation failed; inspect logs/eval_full_flex_reid*_seed${TEST_SEED}.log" >&2
    exit 1
  fi
done

echo "Completed all Reynolds numbers for seed ${TEST_SEED}."
echo "Metrics: assets/eval_re*_seed${TEST_SEED}_*.txt"
