#!/bin/bash
set -euo pipefail

module load python
conda activate /pscratch/sd/k/kenzhong/conda_env/genai
cd "$(dirname "$0")/../.."

CONFIG_PATH="${CONFIG_PATH:-config/ablation/flex_direct_nskt.yaml}"
python train.py --config "${CONFIG_PATH}"

