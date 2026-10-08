#!/bin/bash
# GPU smoke test: every arm for 60 steps on the smoke caches. Usage: sbatch slurm/v2/smoke.sh C|S
#SBATCH --job-name=v2smoke
#SBATCH --partition=gpu        # adapt to your cluster
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
set -euo pipefail
TIER=${1:?C or S}; shift || true
# module load PyTorch/2.7.0-CONDA   # cluster-specific; or activate an environment from requirements.txt
cd "${SLURM_SUBMIT_DIR:-.}"   # run from the repository root
export PYTHONUNBUFFERED=1
V2="${HISTOPATH_DATA:-data}/v2"
P=$([ "$TIER" = C ] && echo 64 || echo 96)
echo "== smoke tier $TIER on $(hostname) =="; nvidia-smi -L
python src/v2/make_smoke_plan.py
python src/v2/run_v2.py --plan $V2/smoke/plans/${TIER}_c17_fold0.json --cache_dir $V2/smoke/cache$P/c17/fold0 \
  --out_root $V2/runs_smoke --ssl_root $V2/smoke/ssl --feats_root $V2/smoke/feats "$@"
