#!/usr/bin/env bash
set -euo pipefail

# Runs GPU-only per-frame timing for UNet, Super SloMo, and FLEX on
# NSKT, Shanghai, and SST. EDEN uses a different environment; run
# scripts/benchmark_inference_time_eden.sh for EDEN timings.
# Uses the active Python environment unless an explicit Conda environment is supplied.
if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi

cd "$(dirname "$0")/.."

NUM_BATCHES="${NUM_BATCHES:-20}"
WARMUP_BATCHES="${WARMUP_BATCHES:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_CSV="${OUTPUT_CSV:-assets/inference_time_per_frame.csv}"

run_bench() {
  python scripts/benchmark_inference_time.py \
    --num_batches "${NUM_BATCHES}" \
    --warmup_batches "${WARMUP_BATCHES}" \
    --num_workers "${NUM_WORKERS}" \
    --output_csv "${OUTPUT_CSV}" \
    "$@"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

# NSKT, Re=36000 by default.
NSKT_RE_ID="${NSKT_RE_ID:-9}"
NSKT_STEPS="${NSKT_STEPS:-20}"
NSKT_SCRATCH="${NSKT_SCRATCH:-data/nskt/}"
NSKT_UNET_CHECKPOINT="${NSKT_UNET_CHECKPOINT:-checkpoints/nskt/unet_best.pt}"
NSKT_SLOMO_CHECKPOINT="${NSKT_SLOMO_CHECKPOINT:-checkpoints/nskt/super_slomo_best.pt}"
NSKT_FLEX_CHECKPOINT="${NSKT_FLEX_CHECKPOINT:-checkpoints/nskt/flex3_best.pt}"
require_file "${NSKT_UNET_CHECKPOINT}"
require_file "${NSKT_SLOMO_CHECKPOINT}"
require_file "${NSKT_FLEX_CHECKPOINT}"

run_bench --model UNet --config "${NSKT_UNET_CONFIG:-config/nskt/unet.yaml}" \
  --data_name nskt --re_id "${NSKT_RE_ID}" --total_interp_steps "${NSKT_STEPS}" \
  --scratch_dir "${NSKT_SCRATCH}" --checkpoint_path "${NSKT_UNET_CHECKPOINT}" \
  --epochs 200 --patch_size 256

run_bench --model SuperSloMo --config "${NSKT_SLOMO_CONFIG:-config/nskt/eval_super_slomo.yaml}" \
  --data_name nskt --re_id "${NSKT_RE_ID}" --total_interp_steps "${NSKT_STEPS}" \
  --scratch_dir "${NSKT_SCRATCH}" --checkpoint_path "${NSKT_SLOMO_CHECKPOINT}"

run_bench --model FLEX --config "${NSKT_FLEX_CONFIG:-config/nskt/flex_posttrain.yaml}" \
  --data_name nskt --re_id "${NSKT_RE_ID}" --total_interp_steps "${NSKT_STEPS}" \
  --scratch_dir "${NSKT_SCRATCH}" --checkpoint_path "${NSKT_FLEX_CHECKPOINT}" \
  --epochs 300 --is_T_fixed False

# Shanghai.
SHANGHAI_STEPS="${SHANGHAI_STEPS:-8}"
SHANGHAI_DATA="${SHANGHAI_DATA:-data/shanghai/shanghai.h5}"
SHANGHAI_UNET_CHECKPOINT="${SHANGHAI_UNET_CHECKPOINT:-checkpoints/shanghai/unet_best.pt}"
SHANGHAI_SLOMO_CHECKPOINT="${SHANGHAI_SLOMO_CHECKPOINT:-checkpoints/shanghai/super_slomo_best.pt}"
SHANGHAI_FLEX_CHECKPOINT="${SHANGHAI_FLEX_CHECKPOINT:-checkpoints/shanghai/flex3_best.pt}"
require_file "${SHANGHAI_UNET_CHECKPOINT}"
require_file "${SHANGHAI_SLOMO_CHECKPOINT}"
require_file "${SHANGHAI_FLEX_CHECKPOINT}"

run_bench --model UNet --config "${SHANGHAI_UNET_CONFIG:-config/shanghai/unet.yaml}" \
  --data_name shanghai --total_interp_steps "${SHANGHAI_STEPS}" \
  --scratch_dir "${SHANGHAI_DATA}" --checkpoint_path "${SHANGHAI_UNET_CHECKPOINT}" \
  --optimizer adam --learning_rate 1e-4 --epochs 400 \
  --stride 32 --total_interp_steps_train 8 --is_T_fixed False \
  --batch_size 12 --patch_size 128 --time_steps 10

run_bench --model SuperSloMo --config "${SHANGHAI_SLOMO_CONFIG:-config/shanghai/eval_super_slomo.yaml}" \
  --data_name shanghai --total_interp_steps "${SHANGHAI_STEPS}" \
  --scratch_dir "${SHANGHAI_DATA}" --checkpoint_path "${SHANGHAI_SLOMO_CHECKPOINT}"

run_bench --model FLEX --config "${SHANGHAI_FLEX_CONFIG:-config/shanghai/flex.yaml}" \
  --data_name shanghai --total_interp_steps "${SHANGHAI_STEPS}" \
  --scratch_dir "${SHANGHAI_DATA}" --checkpoint_path "${SHANGHAI_FLEX_CHECKPOINT}"

# SST / sea temperature.
SST_STEPS="${SST_STEPS:-10}"
SST_DATA="${SST_DATA:-data/sst/}"
SST_UNET_CHECKPOINT="${SST_UNET_CHECKPOINT:-checkpoints/sst/unet_best.pt}"
SST_SLOMO_CHECKPOINT="${SST_SLOMO_CHECKPOINT:-checkpoints/sst/super_slomo_best.pt}"
SST_FLEX_CHECKPOINT="${SST_FLEX_CHECKPOINT:-checkpoints/sst/flex3_best.pt}"
require_file "${SST_UNET_CHECKPOINT}"
require_file "${SST_SLOMO_CHECKPOINT}"
require_file "${SST_FLEX_CHECKPOINT}"

run_bench --model UNet --config "${SST_UNET_CONFIG:-config/sst/unet.yaml}" \
  --data_name sea_temp --total_interp_steps "${SST_STEPS}" \
  --scratch_dir "${SST_DATA}" --checkpoint_path "${SST_UNET_CHECKPOINT}"

run_bench --model SuperSloMo --config "${SST_SLOMO_CONFIG:-config/sst/eval_super_slomo.yaml}" \
  --data_name sea_temp --total_interp_steps "${SST_STEPS}" \
  --scratch_dir "${SST_DATA}" --checkpoint_path "${SST_SLOMO_CHECKPOINT}"

run_bench --model FLEX --config "${SST_FLEX_CONFIG:-config/sst/flex_posttrain.yaml}" \
  --data_name sea_temp --total_interp_steps "${SST_STEPS}" \
  --scratch_dir "${SST_DATA}" --checkpoint_path "${SST_FLEX_CHECKPOINT}"

echo "Timing results appended to ${OUTPUT_CSV}"
