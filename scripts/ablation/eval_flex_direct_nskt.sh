#!/bin/bash
set -euo pipefail

# CHECKPOINT_PATH must point to the trained FLEXDirect checkpoint.
# Uses the active Python environment unless an explicit Conda environment is supplied.
if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi
cd "$(dirname "$0")/../.."

CONFIG_PATH="${CONFIG_PATH:-config/ablation/flex_direct_nskt_posttrain.yaml}"
RE_ID="${RE_ID:--1}"
TEST_SEED="${TEST_SEED:-3407}"
TOTAL_INTERP_STEPS="${TOTAL_INTERP_STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-12}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/nskt/flexdirect_best.pt}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "Set CHECKPOINT_PATH to a trained FLEXDirect checkpoint." >&2
  exit 2
fi

python eval.py \
  --config "${CONFIG_PATH}" \
  --re_id "${RE_ID}" \
  --test_seed "${TEST_SEED}" \
  --total_interp_steps "${TOTAL_INTERP_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --checkpoint_path "${CHECKPOINT_PATH}"
