#!/bin/bash
#SBATCH --job-name=histopath-sota
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=./data/logs/sota_%A_%a.out
#SBATCH --error=./data/logs/sota_%A_%a.err
#SBATCH --array=0-4

# Site-complete factorial for three SOTA-adjacent methods:
# Fish (Shi 2021), LISA (Yao 2022), ERM + H&E jitter (Taori).
# 5 folds x 3 arms x 1 seed = 15 GPU runs.
set -euo pipefail
mkdir -p ./data/logs
module load PyTorch/2.7.0-CONDA
cd .
export PYTHONUNBUFFERED=1

FOLD=$SLURM_ARRAY_TASK_ID
DATA=./data/cache/fold${FOLD}
OUT=./data/runs/fold${FOLD}
mkdir -p "$OUT"
echo "== SOTA fold $FOLD =="; nvidia-smi -L

for arm in fish lisa erm_henorm; do
    if [ ! -f "$OUT/${arm}_seed0.json" ]; then
        echo "== fold=$FOLD $arm =="
        python src/run_arm_ext.py --data "$DATA" --out "$OUT" --arm "$arm" \
            --seed 0 --epochs 5 --bs 128 --lr 0.02 --device cuda --threads 8
    fi
done
