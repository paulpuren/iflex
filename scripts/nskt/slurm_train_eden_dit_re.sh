#!/usr/bin/env bash
#SBATCH --job-name=eden-nskt-re
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/eden-nskt-re-%j.out
#SBATCH --error=logs/eden-nskt-re-%j.err
set -euo pipefail

NUM_PROCESSES="${NUM_PROCESSES:-4}" bash "$(dirname "$0")/train_eden_dit_re.sh"
