#!/bin/bash
#SBATCH --job-name=histopath-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/%x_%A_%a.out
#SBATCH --error=slurm/%x_%A_%a.err
#SBATCH --array=0-14   # 5 folds x 3 seeds = 15 array tasks; each task runs the 8 arms

# Extended site-complete factorial. One SLURM task per (fold, seed); each task
# runs the eight objectives (baseline, sam, ssl, ssl_sam, groupdro, coral, irm,
# mixstyle) so an array of 15 tasks completes 120 runs total. Resumable: any
# task can be re-queued; per-run JSON files are the checkpoint.
#
# Usage:
#   sbatch slurm_run_sweep.sh                          # 5 folds x 3 seeds
#   sbatch --array=0-4 slurm_run_sweep.sh SEEDS="0"    # 5-fold single-seed check
#   sbatch --array=15-29 slurm_run_sweep.sh            # add seeds 3,4

set -euo pipefail
mkdir -p slurm cache runs

module load PyTorch/2.7.0-CONDA

# Map array index -> (fold, seed).  Default: 5 folds x 3 seeds.
N_SEEDS=${N_SEEDS:-3}
FOLD=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

echo "== task $SLURM_ARRAY_TASK_ID: fold=$FOLD seed=$SEED =="
nvidia-smi -L || true

python src/drive_folds_ext.py \
    --folds "$FOLD" --seeds "$SEED" \
    --arms "baseline sam ssl ssl_sam groupdro coral irm mixstyle" \
    --cache_root cache --out_root runs \
    --epochs 8 --bs 128 --threads 8 --device cuda
