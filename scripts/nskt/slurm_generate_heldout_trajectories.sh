#!/usr/bin/env bash
# Submit as a 30-task array: sbatch scripts/nskt/slurm_generate_heldout_trajectories.sh
# Required exports: NSKT_SOLVER_DIR and NSKT_CONFIG_TEMPLATE.
# Outputs contain only 140 vorticity snapshots by default (~2.2 GiB each),
# instead of all u/v/w snapshots (~71 GiB each). Override PACK_MAX_FRAMES if
# the evaluation indexing protocol changes.
#SBATCH --job-name=nskt-heldout
#SBATCH --account=m4633
#SBATCH --constraint=cpu
#SBATCH --qos=regular
# Run one at a time by default: the solver may still create its full raw result
# in temporary storage before the compact HDF5 is packed and the raw data removed.
#SBATCH --array=0-29%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/nskt-heldout-%A_%a.out
#SBATCH --error=logs/nskt-heldout-%A_%a.err
set -euo pipefail

RE_VALUES=(600 1000 2000 4000 8000 12000 16000 24000 32000 36000)
# Three new held-out initial-condition seeds; none overlap train=2150 or old test=3407.
SEEDS=(4610 7727 9871)
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array}"
RE_INDEX=$((TASK_ID / ${#SEEDS[@]}))
SEED_INDEX=$((TASK_ID % ${#SEEDS[@]}))
export REYNOLDS="${RE_VALUES[RE_INDEX]}"
export TRAJECTORY_SEED="${SEEDS[SEED_INDEX]}"
export PACK_FIELDS="${PACK_FIELDS:-w}"
export PACK_MAX_FRAMES="${PACK_MAX_FRAMES:-140}"

bash "$(dirname "$0")/generate_heldout_trajectory.sh"
