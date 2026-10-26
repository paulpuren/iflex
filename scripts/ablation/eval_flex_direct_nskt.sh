#!/bin/bash
set -euo pipefail

# CHECKPOINT_PATH must point to the trained FLEXDirect checkpoint.
module load python
conda activate /pscratch/sd/k/kenzhong/conda_env/genai
cd "$(dirname "$0")/../.."

CONFIG_PATH="${CONFIG_PATH:-config/ablation/flex_direct_nskt_posttrain.yaml}"
RE_ID="${RE_ID:--1}"
TEST_SEED="${TEST_SEED:-3407}"
TOTAL_INTERP_STEPS="${TOTAL_INTERP_STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-12}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/checkpoint_Model_ft_FLEXDirect_small_mlp2_Data_nskt_Optim_adam_cosine_lr0.0001_epoch150_patchsize256_stride32_T20_TfixedFalse_best.pt}"

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
