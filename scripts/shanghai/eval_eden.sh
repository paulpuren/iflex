#!/usr/bin/env bash
set -euo pipefail

# Uses the active Python environment unless an explicit Conda environment is supplied.
if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}/EDEN"

CONFIG_PATH="${CONFIG_PATH:-configs/eval_eden_shanghai.yaml}"

VAE_CKPT="${VAE_CKPT:-${REPO_ROOT}/checkpoints/shanghai/eden_vae.pt}"

DIT_CKPT="${DIT_CKPT:-${REPO_ROOT}/checkpoints/shanghai/eden_dit.pt}"

SCRATCH_DIR="${SCRATCH_DIR:-${REPO_ROOT}/data/shanghai/shanghai.h5}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"

CMD=(python -m accelerate.commands.launch --num_processes "${NUM_PROCESSES}" eval_scientific.py
  --config "${CONFIG_PATH}"
  --vae_checkpoint_path "${VAE_CKPT}"
  --dit_checkpoint_path "${DIT_CKPT}")
if [[ -n "${SCRATCH_DIR}" ]]; then
  CMD+=(--data_path "${SCRATCH_DIR}")
fi

"${CMD[@]}"
