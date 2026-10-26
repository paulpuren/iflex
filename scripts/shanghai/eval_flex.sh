#!/bin/bash
set -euo pipefail

# Example:
#   bash scripts/shanghai/eval_flex.sh
#   TOTAL_INTERP_STEPS=8 BATCH_SIZE=12 bash scripts/shanghai/eval_flex.sh

# Uses the active Python environment unless an explicit Conda environment is supplied.
if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi
cd "$(dirname "$0")/../.."

CONFIG_PATH="${CONFIG_PATH:-config/shanghai/flex.yaml}"
TOTAL_INTERP_STEPS="${TOTAL_INTERP_STEPS:-8}"
BATCH_SIZE="${BATCH_SIZE:-12}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

python eval_shanghai_update.py \
  --config "${CONFIG_PATH}" \
  --total_interp_steps "${TOTAL_INTERP_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  ${EXTRA_ARGS}
