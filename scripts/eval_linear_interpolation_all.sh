#!/usr/bin/env bash
set -euo pipefail

# Run every plain linear-interpolation baseline requested for the paper:
#   NSKT: T=20, Re=12000/24000/36000
#   Shanghai radar: T=8
#   SST: T=10
# No checkpoint or GPU is required.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[1/3] NSKT: T=20, Re=12000/24000/36000"
TOTAL_INTERP_STEPS=20 bash "${SCRIPT_DIR}/nskt/eval_linear_interpolation_all_re.sh"

echo "[2/3] Shanghai radar: T=8"
TOTAL_INTERP_STEPS=8 bash "${SCRIPT_DIR}/shanghai/eval_linear_interpolation.sh"

echo "[3/3] SST: T=10"
TOTAL_INTERP_STEPS=10 bash "${SCRIPT_DIR}/sst/eval_linear_interpolation.sh"

echo "All linear-interpolation evaluations completed. Metrics are under assets/."
