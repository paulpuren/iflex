#!/bin/bash
set -euo pipefail

# Uses the active Python environment unless an explicit Conda environment is supplied.
if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi
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
