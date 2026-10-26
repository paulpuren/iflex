#!/bin/bash
set -euo pipefail

# Example:
#   bash scripts/ablation/eval_flex_no_total_interp_steps_nskt.sh
#   RE_ID=0 TOTAL_INTERP_STEPS=20 bash scripts/ablation/eval_flex_no_total_interp_steps_nskt.sh

# Uses the active Python environment unless an explicit Conda environment is supplied.
if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi
cd "$(dirname "$0")/../.."

CONFIG_PATH="${CONFIG_PATH:-config/ablation/flex_no_total_interp_steps_nskt_posttrain.yaml}"
RE_ID="${RE_ID:--1}"
TOTAL_INTERP_STEPS="${TOTAL_INTERP_STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-12}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/nskt/flex_not_best.pt}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CMD=(
  python eval.py
  --config "${CONFIG_PATH}"
  --re_id "${RE_ID}"
  --total_interp_steps "${TOTAL_INTERP_STEPS}"
  --batch_size "${BATCH_SIZE}"
)

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  CMD+=(--checkpoint_path "${CHECKPOINT_PATH}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARR=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARGS_ARR[@]}")
fi

"${CMD[@]}"
