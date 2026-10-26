#!/usr/bin/env bash
set -euo pipefail

module load python
conda activate "${CONDA_ENV_PATH:-/pscratch/sd/k/kenzhong/conda_env/genai}"

CONFIG_PATH="config/nskt/super_slomo_re.yaml" \
  bash "$(dirname "$0")/train_super_slomo.sh"
