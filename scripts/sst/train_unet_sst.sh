#!/usr/bin/env bash
set -euo pipefail

# Uses the active Python environment unless an explicit Conda environment is supplied.
if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi
cd "$(dirname "$0")/../.."

PRETRAINED_PATH="${PRETRAINED_PATH:-checkpoints/sst/unet_best.pt}"

# run sea temperature data
python train_unet.py \
    --run_name UNet \
    --data_name 'sea_temp' \
    --model UNet \
    --sampling_freq 2 \
    --optimizer 'adam' \
    --epochs 30 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --checkpoint_path "${PRETRAINED_PATH}" \
    --total_interp_steps_train 10 \
    --patch_size 128 \
    --stride 64 \
    --is_T_fixed False \
    --scratch_dir "${SCRATCH_DIR:-data/sst/}"

#     --checkpoint_path '' \