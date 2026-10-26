#!/bin/bash
set -euo pipefail

# Example:
#   bash scripts/forecasting/eval_flex_forecast_nskt.sh
#   RE_ID=9 FORECAST_HORIZON=20 bash scripts/forecasting/eval_flex_forecast_nskt.sh

# Uses the active Python environment unless an explicit Conda environment is supplied.
if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi
cd "$(dirname "$0")/../.."

CONFIG_PATH="${CONFIG_PATH:-config/forecasting/eval_flex_forecast_nskt.yaml}"
RE_ID="${RE_ID:-5}"
FORECAST_HORIZON="${FORECAST_HORIZON:-20}"
BATCH_SIZE="${BATCH_SIZE:-12}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/forecasting_nskt/flexforecast_renorm40000_best.pt}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CMD=(
  python forecasting/eval.py
  --config "${CONFIG_PATH}"
  --re_id "${RE_ID}"
  --forecast_horizon "${FORECAST_HORIZON}"
  --batch_size "${BATCH_SIZE}"
)

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  CMD+=(--checkpoint_path "${CHECKPOINT_PATH}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARR=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARGS_ARR[@]}")
fi

"${CMD[@]}"
