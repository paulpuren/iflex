#!/usr/bin/env bash
set -euo pipefail

# Plain endpoint linear-interpolation baseline. No checkpoint or GPU is needed.

module load python
conda activate "${CONDA_ENV_PATH:-/pscratch/sd/p/puren93/conda_env/genai}"
cd "$(dirname "$0")/../.."

TOTAL_INTERP_STEPS="${TOTAL_INTERP_STEPS:-10}"
BATCH_SIZE="${BATCH_SIZE:-12}"
PATCH_SIZE="${PATCH_SIZE:-128}"
SCRATCH_DIR="${SCRATCH_DIR:-/global/cfs/projectdirs/m4633/puren/interp_dm/sea_temp/}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# shellcheck disable=SC2086
python eval_linear_interpolation.py \
  --data_name sea_temp \
  --total_interp_steps "${TOTAL_INTERP_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --patch_size "${PATCH_SIZE}" \
  --scratch_dir "${SCRATCH_DIR}" \
  ${EXTRA_ARGS}
