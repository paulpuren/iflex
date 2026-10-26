#!/bin/bash
set -euo pipefail

# Resume a 300-epoch FLEXDirect checkpoint and train through epoch 450.
# Uses the active Python environment unless an explicit Conda environment is supplied.
if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi
cd "$(dirname "$0")/../.."

CONFIG_PATH="${CONFIG_PATH:-config/ablation/flex_direct_nskt_posttrain.yaml}"
python train.py --config "${CONFIG_PATH}"
