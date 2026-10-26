#!/usr/bin/env bash
set -euo pipefail

# Uses the active Python environment unless an explicit Conda environment is supplied.
if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}/EDEN"

CONFIG_PATH="${CONFIG_PATH:-configs/eval_eden_nskt.yaml}"
VAE_CKPT="${VAE_CKPT:-${REPO_ROOT}/checkpoints/nskt/eden_vae.pt}"
DIT_CKPT="${DIT_CKPT:-${REPO_ROOT}/checkpoints/nskt/eden_dit.pt}"

RE_ID="${RE_ID:-9}"
TOTAL_INTERP_STEPS="${TOTAL_INTERP_STEPS:-24}"
TEST_SEED="${TEST_SEED:-3407}"
SCRATCH_DIR="${SCRATCH_DIR:-${REPO_ROOT}/data/nskt/}"
CHECK_FILE="${SCRATCH_DIR%/}/600_2048_2048_seed_3407.h5"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

echo "NSKT scratch_dir: ${SCRATCH_DIR}"
echo "NSKT total interpolation steps: ${TOTAL_INTERP_STEPS}"
if [[ ! -f "${CHECK_FILE}" ]]; then
  echo "NSKT dataset not found: ${CHECK_FILE}" >&2
  exit 1
fi

CMD=(python -m accelerate.commands.launch
  --num_processes "${NUM_PROCESSES}"
  --num_machines 1
  --mixed_precision no
  --dynamo_backend no
  eval_scientific.py
  --config "${CONFIG_PATH}"
  --vae_checkpoint_path "${VAE_CKPT}"
  --dit_checkpoint_path "${DIT_CKPT}"
  --re_id "${RE_ID}"
  --test_seed "${TEST_SEED}"
  --total_interp_steps "${TOTAL_INTERP_STEPS}")
CMD+=(--scratch_dir "${SCRATCH_DIR}")

"${CMD[@]}"
