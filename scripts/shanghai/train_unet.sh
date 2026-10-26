#!/usr/bin/env bash
set -euo pipefail

# Uses the active Python environment unless an explicit Conda environment is supplied.
if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi
cd "$(dirname "$0")/../.."

# run shanghai radar data
python train_unet.py \
    --model UNet \
    --run_name UNet \
    --data_name 'shanghai' \
    --sampling_freq 10 \
    --optimizer 'adam' \
    --epochs 400 \
    --batch_size 12 \
    --learning_rate 1e-4 \
    --checkpoint_path '' \
    --total_interp_steps_train 8 \
    --is_T_fixed False \
    --patch_size 128 \
    --stride 32 \
    --scratch_dir "${SCRATCH_DIR:-data/shanghai/shanghai.h5}"
