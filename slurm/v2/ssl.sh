#!/bin/bash
# SimCLR pre-training for one (tier, cohort, fold) per array task. (run_v2.py also pre-trains
# a missing SSL checkpoint on the fly; this script lets SSL run ahead of / apart from the plan.)
#   sbatch --array=0-4 slurm/v2/ssl.sh C c17 "0,1,2,3,4"          # task id = fold
#   sbatch --array=0-4 slurm/v2/ssl.sh S c17 "0,1,2" resnet50 imagenet
#SBATCH --job-name=v2ssl
#SBATCH --partition=gpu        # adapt to your cluster
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ssl_%A_%a.out
#SBATCH --error=logs/ssl_%A_%a.err
set -euo pipefail
TIER=${1:?tier}; COHORT=${2:?cohort}; SEEDS=${3:-0}; BB=${4:-}; INIT=${5:-}
shift $(( $# < 5 ? $# : 5 )) || true
# module load PyTorch/2.7.0-CONDA   # cluster-specific; or activate an environment from requirements.txt
cd "${SLURM_SUBMIT_DIR:-.}"   # run from the repository root
export PYTHONUNBUFFERED=1
ARGS=(--tier "$TIER" --cohort "$COHORT" --fold "${SLURM_ARRAY_TASK_ID:-0}" --seeds "$SEEDS")
[ -n "$BB" ] && ARGS+=(--backbone "$BB")
[ -n "$INIT" ] && ARGS+=(--init "$INIT")
echo "== ssl $TIER $COHORT fold ${SLURM_ARRAY_TASK_ID:-0} seeds $SEEDS on $(hostname) =="; nvidia-smi -L
python src/v2/ssl_v2.py "${ARGS[@]}" "$@"
