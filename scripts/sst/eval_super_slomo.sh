#!/bin/bash
set -euo pipefail

# Uses the active Python environment unless an explicit Conda environment is supplied.
if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi
cd "$(dirname "$0")/../.."

if [[ "${CUDA_VISIBLE_DEVICES:-}" == *,* ]]; then
  FIRST_VISIBLE_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
  echo "Using a single GPU for eval: CUDA_VISIBLE_DEVICES=${FIRST_VISIBLE_GPU}" >&2
  export CUDA_VISIBLE_DEVICES="${FIRST_VISIBLE_GPU}"
fi

CONFIG_PATH="${CONFIG_PATH:-config/sst/eval_super_slomo.yaml}"

python eval_super_slomo.py --config "${CONFIG_PATH}"
