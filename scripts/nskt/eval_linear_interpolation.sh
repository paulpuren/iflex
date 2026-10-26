#!/bin/bash
set -euo pipefail

# Plain endpoint linear-interpolation baseline. No checkpoint or GPU is needed.
# Example: RE_ID=5 TOTAL_INTERP_STEPS=20 bash scripts/nskt/eval_linear_interpolation.sh

# Uses the active Python environment unless an explicit Conda environment is supplied.
if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi
cd "$(dirname "$0")/../.."

RE_ID="${RE_ID:-5}"
TEST_SEED="${TEST_SEED:-3407}"
TOTAL_INTERP_STEPS="${TOTAL_INTERP_STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-12}"
PATCH_SIZE="${PATCH_SIZE:-256}"
CROP_SIZE="${CROP_SIZE:-256}"
STRIDE="${STRIDE:-32}"
SCRATCH_DIR="${SCRATCH_DIR:-data/nskt/}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# shellcheck disable=SC2086
python eval_linear_interpolation.py \
  --data_name nskt \
  --re_id "${RE_ID}" \
  --test_seed "${TEST_SEED}" \
  --total_interp_steps "${TOTAL_INTERP_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --patch_size "${PATCH_SIZE}" \
  --crop_size "${CROP_SIZE}" \
  --stride "${STRIDE}" \
  --scratch_dir "${SCRATCH_DIR}" \
  ${EXTRA_ARGS}
