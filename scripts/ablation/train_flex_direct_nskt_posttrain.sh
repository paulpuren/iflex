#!/bin/bash
set -euo pipefail

# Resume a 300-epoch FLEXDirect checkpoint and train through epoch 450.
module load python
conda activate /pscratch/sd/k/kenzhong/conda_env/genai
cd "$(dirname "$0")/../.."

CONFIG_PATH="${CONFIG_PATH:-config/ablation/flex_direct_nskt_posttrain.yaml}"
python train.py --config "${CONFIG_PATH}"
