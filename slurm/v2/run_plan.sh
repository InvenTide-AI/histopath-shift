#!/bin/bash
# One array task = one plan shard = one (tier, cohort, fold). Submit with
#   slurm/v2/submit.sh <shards.txt> [extra sbatch args, e.g. --time 08:00:00]
# or directly:
#   sbatch --array=0-$(( $(wc -l < SHARDS) - 1 )) slurm/v2/run_plan.sh SHARDS [extra run_v2.py args]
#SBATCH --job-name=v2plan
#SBATCH --partition=gpu        # adapt to your cluster
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=logs/plan_%A_%a.out
#SBATCH --error=logs/plan_%A_%a.err
set -euo pipefail
SHARDS=${1:?usage: run_plan.sh SHARDS_FILE [run_v2 args...]}
shift || true
# module load PyTorch/2.7.0-CONDA   # cluster-specific; or activate an environment from requirements.txt
cd "${SLURM_SUBMIT_DIR:-.}"   # run from the repository root
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
PLAN=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" "$SHARDS")
echo "== task ${SLURM_ARRAY_TASK_ID:-0} plan $PLAN on $(hostname) =="; nvidia-smi -L
python src/v2/run_v2.py --plan "$PLAN" "$@"
