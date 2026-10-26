#!/usr/bin/env bash
set -euo pipefail

set -euo pipefail

# Uses the active Python environment unless an explicit Conda environment is supplied.
if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi
cd "$(dirname "$0")/../.."

CONFIG_PATH="${CONFIG_PATH:-config/nskt/unet.yaml}"

python train_unet.py --config "${CONFIG_PATH}"
