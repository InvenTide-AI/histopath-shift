#!/bin/bash
#SBATCH --job-name=tta
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=./data/logs/tta_%A_%a.out
#SBATCH --error=./data/logs/tta_%A_%a.err

# Test-time adaptation, the fourth intervention level in the taxonomy.
# Submit per cohort, e.g.
#   sbatch --array=0-4 --export=ALL,COHORT=cache,RUNS=runs slurm_tta.sh
set -euo pipefail
module load PyTorch/2.7.0-CONDA
cd .
export PYTHONUNBUFFERED=1
D=./data
FOLD=$SLURM_ARRAY_TASK_ID
DATA=$D/${COHORT}/fold${FOLD}
OUT=$D/${RUNS}/fold${FOLD}
echo "== TTA cohort=$COHORT fold=$FOLD on $(hostname) =="

for SEED in 0 1 2; do
  for ARM in bnadapt tent; do
    if [ -f "$OUT/baseline_seed${SEED}.pt" ] && [ ! -f "$OUT/${ARM}_seed${SEED}.json" ]; then
      python src/run_arm_tta.py --data "$DATA" --out "$OUT" --arm "$ARM" \
        --source baseline --seed "$SEED" --device cuda
    fi
  done
  # does test-time adaptation compose with the best training-time arm?
  for SRC in ssl erm_henorm; do
    if [ -f "$OUT/${SRC}_seed${SEED}.pt" ] && [ ! -f "$OUT/bnadapt-on-${SRC}_seed${SEED}.json" ]; then
      python src/run_arm_tta.py --data "$DATA" --out "$OUT" --arm bnadapt \
        --source "$SRC" --seed "$SEED" --arm_name "bnadapt-on-${SRC}" --device cuda
    fi
  done
done
echo "== done =="
