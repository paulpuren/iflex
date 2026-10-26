#!/usr/bin/env bash
# Evaluate one model independently for all Re/trajectory pairs.
# Usage: MODEL=flex|eden_re|slomo_re CHECKPOINT_PATH=... bash this_file
set -euo pipefail

: "${MODEL:?Set MODEL to flex, eden_re, or slomo_re}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the matching trained checkpoint}"
RE_IDS=(0 1 2 3 4 5 6 7 8 9)
TEST_SEEDS=(${TEST_SEEDS:-4610 7727 9871})
# Number of candidate patches evaluated per trajectory. eval.py samples these
# deterministically from 2000 candidates; set 0 only when a full pass is wanted.
NSKT_EVAL_MAX_SAMPLES="${NSKT_EVAL_MAX_SAMPLES:-256}"
export NSKT_EVAL_MAX_SAMPLES

echo "Evaluating ${NSKT_EVAL_MAX_SAMPLES} sampled patches per Re/trajectory"

for seed in "${TEST_SEEDS[@]}"; do
  for re_id in "${RE_IDS[@]}"; do
    case "${MODEL}" in
      flex)
        RE_ID="${re_id}" TEST_SEED="${seed}" CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
          bash "$(dirname "$0")/eval_flex_nskt.sh"
        ;;
      eden_re)
        RE_ID="${re_id}" TEST_SEED="${seed}" DIT_CKPT="${CHECKPOINT_PATH}" \
          CONFIG_PATH="configs/eval_eden_nskt_re.yaml" bash "$(dirname "$0")/eval_eden.sh"
        ;;
      slomo_re)
        RE_ID="${re_id}" TEST_SEED="${seed}" CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
          bash "$(dirname "$0")/eval_super_slomo_re.sh"
        ;;
      *) echo "Unsupported MODEL=${MODEL}" >&2; exit 2 ;;
    esac
  done
done
