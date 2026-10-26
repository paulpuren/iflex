#!/bin/bash
set -euo pipefail

# Plain endpoint linear-interpolation baseline. No checkpoint or GPU is needed.

# Uses the active Python environment unless an explicit Conda environment is supplied.
if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi
cd "$(dirname "$0")/../.."

TOTAL_INTERP_STEPS="${TOTAL_INTERP_STEPS:-8}"
BATCH_SIZE="${BATCH_SIZE:-12}"
PATCH_SIZE="${PATCH_SIZE:-128}"
SCRATCH_DIR="${SCRATCH_DIR:-data/shanghai/shanghai.h5}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# shellcheck disable=SC2086
python eval_linear_interpolation.py \
  --data_name shanghai \
  --total_interp_steps "${TOTAL_INTERP_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --patch_size "${PATCH_SIZE}" \
  --scratch_dir "${SCRATCH_DIR}" \
  ${EXTRA_ARGS}
