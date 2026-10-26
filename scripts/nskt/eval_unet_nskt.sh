#!/usr/bin/env bash
set -euo pipefail

# Uses the active Python environment unless an explicit Conda environment is supplied.
if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi
cd "$(dirname "$0")/../.."

PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-checkpoints/nskt/unet_best.pt}"

# select 12000, 24000, and 36000: 5, 7, 9 for eval OOD generalization

# run nskt
python eval.py \
    --model UNet \
    --data_name 'nskt' \
    --re_id 9 \
    --optimizer 'adam' \
    --epochs 200 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --total_interp_steps 16 \
    --total_interp_steps_train 20 \
    --is_T_fixed False \
    --patch_size 256 \
    --stride 32 \
    --checkpoint_path "${PRETRAINED_CHECKPOINT}" \
    --scratch_dir "${SCRATCH_DIR:-data/nskt/}"