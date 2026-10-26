#!/bin/bash
set -euo pipefail

# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m4633

module load python
conda activate "${CONDA_ENV_PATH:-/pscratch/sd/k/kenzhong/conda_env/genai}"
cd "$(dirname "$0")/../.."

CONFIG_PATH="${CONFIG_PATH:-config/nskt/eval_super_slomo.yaml}"
RE_ID="${RE_ID:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
TEST_SEED="${TEST_SEED:-}"

CMD=(python eval_super_slomo.py --config "${CONFIG_PATH}")
[[ -n "${RE_ID}" ]] && CMD+=(--re_id "${RE_ID}")
[[ -n "${CHECKPOINT_PATH}" ]] && CMD+=(--checkpoint_path "${CHECKPOINT_PATH}")
[[ -n "${TEST_SEED}" ]] && CMD+=(--test_seed "${TEST_SEED}")
"${CMD[@]}"
