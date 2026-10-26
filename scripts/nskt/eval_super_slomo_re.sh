#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="config/nskt/eval_super_slomo_re.yaml" \
  bash "$(dirname "$0")/eval_super_slomo_nskt.sh"
