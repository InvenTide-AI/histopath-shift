#!/bin/bash
#SBATCH --job-name=histopath-probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=./data/logs/probe_%A_%a.out
#SBATCH --error=./data/logs/probe_%A_%a.err
#SBATCH --array=0-4

set -euo pipefail
mkdir -p ./data/logs
module load PyTorch/2.7.0-CONDA
cd .
export PYTHONUNBUFFERED=1

FOLD=$SLURM_ARRAY_TASK_ID
DATA=./data/cache/fold${FOLD}
OUT=./data/runs/fold${FOLD}
mkdir -p "$OUT"
echo "== probe fold $FOLD =="; nvidia-smi -L

if [ ! -f "$OUT/probe_resnet50_seed0.json" ]; then
    python src/run_arm_probe.py --data "$DATA" --out "$OUT" --seed 0 --epochs 15 --bs 256 --device cuda
fi
