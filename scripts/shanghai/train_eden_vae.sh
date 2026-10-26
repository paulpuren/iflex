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
CONFIG_PATH="${CONFIG_PATH:-configs/train_vae_shanghai.yaml}"
RESUME_FROM_CKPT="${RESUME_FROM_CKPT:-}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/shanghai/shanghai.h5}"

CMD=(python -m accelerate.commands.launch --num_processes "${NUM_PROCESSES}" train_vae.py --config "${CONFIG_PATH}")
if [[ -n "${RESUME_FROM_CKPT}" ]]; then
  CMD+=(--resume_from_ckpt "${RESUME_FROM_CKPT}")
fi

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Shanghai dataset not found: ${DATA_PATH}" >&2
  exit 1
fi

CMD+=(--data_path "${DATA_PATH}")

"${CMD[@]}"
