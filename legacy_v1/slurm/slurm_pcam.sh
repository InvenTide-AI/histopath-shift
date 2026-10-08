#!/bin/bash
#SBATCH --job-name=histopath-pcam
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=./data/logs/pcam_%j.out
#SBATCH --error=./data/logs/pcam_%j.err

set -euo pipefail
module load PyTorch/2.7.0-CONDA
cd .
export PYTHONUNBUFFERED=1
nvidia-smi -L

python src/eval_pcam.py \
    --pcam_root ./data/pcam \
    --runs_root ./data/runs \
    --out results/camelyon17_extended/pcam_external.csv \
    --arms "baseline sam ssl ssl_sam" \
    --seeds "0 1 2"
