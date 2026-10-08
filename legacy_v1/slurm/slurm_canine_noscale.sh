#!/bin/bash
#SBATCH --job-name=canine-noscale
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=./data/logs/noscale_%A_%a.out
#SBATCH --error=./data/logs/noscale_%A_%a.err
#SBATCH --array=0-4

# Canine cohort with scale normalization switched OFF. Identical slides,
# scanners, folds, task and patch counts as canine_cache; the only difference
# is that the scanners' 0.22-0.26 um/px spread is left in the data. The
# contrast between the two caches isolates the resolution component of a
# scanner change.
#
# Prediction worth recording in advance: sup_augment (used by every supervised
# arm) contains flips and 90-degree rotations but no scale augmentation,
# whereas the SimCLR view contains random-resized-crop. If robustness tracks
# whether the nuisance is represented in a method's augmentation, SSL should
# gain here relative to the scale-normalized cohort and the photometric arms
# should not.
set -euo pipefail
module load PyTorch/2.7.0-CONDA
cd .
export PYTHONUNBUFFERED=1
D=./data
FOLD=$SLURM_ARRAY_TASK_ID
DATA=$D/canine_noscale_cache/fold${FOLD}
OUT=$D/canine_noscale_runs/fold${FOLD}
mkdir -p "$OUT"
echo "== canine no-scale fold $FOLD on $(hostname) =="

for SEED in 0 1 2; do
  if [ ! -f "$DATA/ssl_encoder_seed${SEED}.pt" ]; then
    python src/pretrain_ssl_gpu_ext.py --data "$DATA" --seed "$SEED" --epochs 6 --bs 256 --device cuda
  fi
done

for arm in baseline ssl ssl_sam; do
  for SEED in 0 1 2; do
    if [ ! -f "$OUT/${arm}_seed${SEED}.json" ]; then
      python src/run_arm_ext.py --data "$DATA" --out "$OUT" --arm "$arm" \
        --seed "$SEED" --epochs 5 --bs 128 --lr 0.02 --device cuda --threads 8
    fi
  done
done

for arm in macenko erm_henorm tia_style; do
  if [ ! -f "$OUT/${arm}_seed0.json" ]; then
    python src/run_arm_ext.py --data "$DATA" --out "$OUT" --arm "$arm" \
      --seed 0 --epochs 5 --bs 128 --lr 0.02 --device cuda --threads 8
  fi
done

for SEED in 0 1 2; do
  for ARM in bnadapt tent; do
    if [ -f "$OUT/baseline_seed${SEED}.pt" ] && [ ! -f "$OUT/${ARM}_seed${SEED}.json" ]; then
      python src/run_arm_tta.py --data "$DATA" --out "$OUT" --arm "$ARM" \
        --source baseline --seed "$SEED" --device cuda
    fi
  done
done

if [ ! -f "$OUT/probe_resnet50_seed0.json" ]; then
  python src/run_probe_ladder.py --data "$DATA" --out "$OUT" --arch resnet50 \
    --seed 0 --epochs 15 --bs 256 --device cuda
fi
echo "== done =="
