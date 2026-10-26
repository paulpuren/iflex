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
CONFIG_PATH="${CONFIG_PATH:-configs/train_vae_nskt_science.yaml}"
RESUME_FROM_CKPT="${RESUME_FROM_CKPT:-}"
SCRATCH_DIR="${SCRATCH_DIR:-${REPO_ROOT}/data/nskt/}"
CHECK_FILE="${SCRATCH_DIR%/}/2000_2048_2048_seed_2150.h5"

echo "NSKT scratch_dir: ${SCRATCH_DIR}"
if [[ ! -f "${CHECK_FILE}" ]]; then
  echo "NSKT dataset not found: ${CHECK_FILE}" >&2
  exit 1
fi

CMD=(python -m accelerate.commands.launch --num_processes "${NUM_PROCESSES}" train_vae.py --config "${CONFIG_PATH}")
if [[ -n "${RESUME_FROM_CKPT}" ]]; then
  CMD+=(--resume_from_ckpt "${RESUME_FROM_CKPT}")
fi
CMD+=(--scratch_dir "${SCRATCH_DIR}")

"${CMD[@]}"
