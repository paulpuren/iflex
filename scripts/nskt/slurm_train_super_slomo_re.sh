#!/usr/bin/env bash
#SBATCH --job-name=slomo-nskt-re
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/slomo-nskt-re-%j.out
#SBATCH --error=logs/slomo-nskt-re-%j.err
set -euo pipefail

bash "$(dirname "$0")/train_super_slomo_re.sh"
