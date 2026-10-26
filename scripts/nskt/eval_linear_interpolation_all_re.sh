#!/usr/bin/env bash
set -euo pipefail

# Linear interpolation baseline for the three reported NSKT OOD Reynolds numbers.
# RE_EVAL_LIST indices: 12000 -> 5, 24000 -> 7, 36000 -> 9.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RE_IDS=(5 7 9)

export TOTAL_INTERP_STEPS="${TOTAL_INTERP_STEPS:-20}"
export TEST_SEED="${TEST_SEED:-3407}"
export NSKT_EVAL_MAX_SAMPLES="${NSKT_EVAL_MAX_SAMPLES:-256}"

for re_id in "${RE_IDS[@]}"; do
  echo "Running NSKT linear interpolation: T=${TOTAL_INTERP_STEPS}, re_id=${re_id}, seed=${TEST_SEED}"
  RE_ID="${re_id}" bash "${SCRIPT_DIR}/eval_linear_interpolation.sh"
done
