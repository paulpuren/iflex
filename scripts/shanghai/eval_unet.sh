#!/usr/bin/env bash
set -euo pipefail

# Example:
#   bash scripts/shanghai/eval_unet_shanghai.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/shanghai/eval_unet_shanghai.sh
#   TOTAL_INTERP_STEPS_EVAL=20 BATCH_SIZE=8 bash scripts/shanghai/eval_unet_shanghai.sh

# Uses the active Python environment unless an explicit Conda environment is supplied.
if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi

# Run from repository root (script lives in scripts/shanghai/)
cd "$(dirname "$0")/../.."

# Support both filenames; repo currently has eval_shanghai_update.py.
if [[ -f "eval_shanghai_updated.py" ]]; then
  EVAL_SCRIPT="eval_shanghai_updated.py"
else
  EVAL_SCRIPT="eval_shanghai_update.py"
fi

# ---- checkpoint/run-name-matching args (must match training config) ----
MODEL="${MODEL:-UNet}"
DATA_NAME="${DATA_NAME:-shanghai}"
OPTIMIZER="${OPTIMIZER:-adam}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
EPOCHS="${EPOCHS:-400}"
STRIDE="${STRIDE:-32}"
TOTAL_INTERP_STEPS_TRAIN="${TOTAL_INTERP_STEPS_TRAIN:-8}"
IS_T_FIXED="${IS_T_FIXED:-False}"

# ---- evaluation args ----
BATCH_SIZE="${BATCH_SIZE:-12}"
PATCH_SIZE="${PATCH_SIZE:-128}"
TIME_STEPS="${TIME_STEPS:-10}"
PREDICTION_TYPE="${PREDICTION_TYPE:-v}"
TOTAL_INTERP_STEPS_EVAL="${TOTAL_INTERP_STEPS_EVAL:-8}"
SCRATCH_DIR="${SCRATCH_DIR:-data/shanghai/shanghai.h5}"

python "${EVAL_SCRIPT}" \
  --model "${MODEL}" \
  --data_name "${DATA_NAME}" \
  --optimizer "${OPTIMIZER}" \
  --learning_rate "${LEARNING_RATE}" \
  --epochs "${EPOCHS}" \
  --stride "${STRIDE}" \
  --total_interp_steps_train "${TOTAL_INTERP_STEPS_TRAIN}" \
  --is_T_fixed "${IS_T_FIXED}" \
  --batch_size "${BATCH_SIZE}" \
  --patch_size "${PATCH_SIZE}" \
  --time_steps "${TIME_STEPS}" \
  --prediction_type "${PREDICTION_TYPE}" \
  --total_interp_steps "${TOTAL_INTERP_STEPS_EVAL}" \
  --scratch_dir "${SCRATCH_DIR}"
