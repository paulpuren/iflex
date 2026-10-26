#!/usr/bin/env bash
set -euo pipefail

# Uses the active Python environment unless an explicit Conda environment is supplied.
if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}/EDEN"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
CONFIG_PATH="${CONFIG_PATH:-configs/train_dit_sst.yaml}"
PRETRAINED_VAE_PATH="${PRETRAINED_VAE_PATH:-${REPO_ROOT}/checkpoints/sst/eden_vae.pt}"
RESUME_FROM_CKPT="${RESUME_FROM_CKPT:-}"
SCRATCH_DIR="${SCRATCH_DIR:-${REPO_ROOT}/data/sst/}"

CMD=(python -m accelerate.commands.launch --num_processes "${NUM_PROCESSES}" train_dit.py --config "${CONFIG_PATH}" --pretrained_vae_path "${PRETRAINED_VAE_PATH}")
if [[ -n "${RESUME_FROM_CKPT}" ]]; then
  CMD+=(--resume_from_ckpt "${RESUME_FROM_CKPT}")
fi
if [[ -n "${SCRATCH_DIR}" ]]; then
  CMD+=(--data_path "${SCRATCH_DIR}")
fi

"${CMD[@]}"
