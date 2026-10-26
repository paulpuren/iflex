#!/bin/bash
set -euo pipefail

#
# Example:
#   bash scripts/forecasting/train_flex_forecast_nskt_posttrain.sh
#   EPOCHS=500 bash scripts/forecasting/train_flex_forecast_nskt_posttrain.sh

# Uses the active Python environment unless an explicit Conda environment is supplied.
if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi
cd "$(dirname "$0")/../.."

CONFIG_PATH="${CONFIG_PATH:-config/forecasting/flex_forecast_nskt_posttrain.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/forecasting_nskt/flexforecast_renorm40000_last.pt}"
EPOCHS="${EPOCHS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CMD=(
  python forecasting/train.py
  --config "${CONFIG_PATH}"
)

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  CMD+=(--checkpoint_path "${CHECKPOINT_PATH}")
fi

if [[ -n "${EPOCHS}" ]]; then
  CMD+=(--epochs "${EPOCHS}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARR=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARGS_ARR[@]}")
fi

"${CMD[@]}"
