#!/bin/bash
set -euo pipefail

# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m4633

module load python
conda activate "${CONDA_ENV_PATH:-/pscratch/sd/k/kenzhong/conda_env/genai}"
cd "$(dirname "$0")/../.."

CONFIG_PATH="${CONFIG_PATH:-config/nskt/super_slomo.yaml}"

python train_super_slomo.py --config "${CONFIG_PATH}"
