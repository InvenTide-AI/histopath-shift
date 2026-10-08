#!/bin/bash
# Timing benchmark: sbatch slurm/v2/bench.sh C|S
#SBATCH --job-name=v2bench
#SBATCH --partition=gpu        # adapt to your cluster
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err
set -euo pipefail
# module load PyTorch/2.7.0-CONDA   # cluster-specific; or activate an environment from requirements.txt
cd "${SLURM_SUBMIT_DIR:-.}"   # run from the repository root
export PYTHONUNBUFFERED=1
nvidia-smi -L
python src/v2/bench.py --tier "${1:?C or S}" "${@:2}"
