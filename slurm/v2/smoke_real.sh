#!/bin/bash
# Real-data GPU smoke (short budgets) on the real v2 c17 fold-0 caches, both tiers, run twice
# (2nd pass must skip every job = resume + fingerprint check). Usage: sbatch slurm/v2/smoke_real.sh
#SBATCH --job-name=v2smokeR
#SBATCH --partition=gpu        # adapt to your cluster
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke_real_%j.out
#SBATCH --error=logs/smoke_real_%j.err
set -euo pipefail
# module load PyTorch/2.7.0-CONDA   # cluster-specific; or activate an environment from requirements.txt
cd "${SLURM_SUBMIT_DIR:-.}"   # run from the repository root
export PYTHONUNBUFFERED=1
V2="${HISTOPATH_DATA:-data}/v2"
echo "== real smoke on $(hostname) =="; nvidia-smi -L
python src/v2/make_smoke_plan_real.py
for T in C S; do
  for pass in 1 2; do
    echo "== tier $T pass $pass =="
    python src/v2/run_v2.py --plan $V2/smoke/plans/real_${T}_c17_fold0.json \
      --out_root $V2/runs_smoke/real --ssl_root $V2/smoke/ssl_real --feats_root $V2/smoke/feats_real
  done
done
