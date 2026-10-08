#!/bin/bash
#SBATCH --job-name=histopath-ms
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=./data/logs/ms_%A_%a.out
#SBATCH --error=./data/logs/ms_%A_%a.err
#SBATCH --array=0-9

# 10 array tasks: 5 folds x 2 additional seeds (1, 2).
# Each task: pretrain SSL for that (fold, seed), then run baseline/sam/ssl/ssl_sam.
# Also runs macenko for seed 0 on that fold (since Macenko is a one-seed baseline).
set -euo pipefail
mkdir -p ./data/logs
module load PyTorch/2.7.0-CONDA
cd .
export PYTHONUNBUFFERED=1

# Map task index -> (fold, seed).  seed 1 = tasks 0..4, seed 2 = tasks 5..9.
FOLD=$(( SLURM_ARRAY_TASK_ID % 5 ))
SEED=$(( SLURM_ARRAY_TASK_ID / 5 + 1 ))

DATA=./data/cache/fold${FOLD}
OUT=./data/runs/fold${FOLD}
mkdir -p "$OUT"
echo "== task $SLURM_ARRAY_TASK_ID: fold=$FOLD seed=$SEED =="; nvidia-smi -L

# SSL pretrain for this (fold, seed)
if [ ! -f "$DATA/ssl_encoder_seed${SEED}.pt" ]; then
    python src/pretrain_ssl_gpu_ext.py --data "$DATA" --seed $SEED --epochs 6 --bs 256 --device cuda
fi

# Four core arms at this seed
for arm in baseline sam ssl ssl_sam; do
    if [ ! -f "$OUT/${arm}_seed${SEED}.json" ]; then
        echo "== fold=$FOLD $arm seed=$SEED =="
        python src/run_arm_ext.py --data "$DATA" --out "$OUT" --arm "$arm" \
            --seed $SEED --epochs 5 --bs 128 --lr 0.02 --device cuda --threads 8
    fi
done

# Macenko: run only on the first task per fold (i.e. seed 1) using seed=0 for reproducibility with Table III.
if [ $SEED -eq 1 ] && [ ! -f "$OUT/macenko_seed0.json" ]; then
    echo "== fold=$FOLD macenko seed=0 =="
    python src/run_arm_ext.py --data "$DATA" --out "$OUT" --arm macenko \
        --seed 0 --epochs 5 --bs 128 --lr 0.02 --device cuda --threads 8
fi

echo "== fold $FOLD seed $SEED summary =="
ls "$OUT"/*_seed${SEED}.json 2>/dev/null | wc -l
