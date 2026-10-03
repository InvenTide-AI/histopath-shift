#!/bin/bash
#SBATCH --job-name=probe-ladder
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=./data/logs/ladder_%A_%a.out
#SBATCH --error=./data/logs/ladder_%A_%a.err

# Frozen-backbone pretraining-scale ladder, one SLURM task per fold.
# Submit with COHORT set, e.g.
#   sbatch --array=0-4 --export=ALL,COHORT=cache,RUNS=runs slurm_probe_ladder.sh
#   sbatch --array=0-2 --export=ALL,COHORT=midog_cache,RUNS=midog_runs slurm_probe_ladder.sh
#   sbatch --array=0-4 --export=ALL,COHORT=canine_cache,RUNS=canine_runs slurm_probe_ladder.sh
#
# vit_b_16 against vit_b_16_swag is the controlled pair: same architecture,
# ~2800x the pretraining corpus. Everything after the frozen features is held
# fixed across the ladder.
set -euo pipefail
mkdir -p ./data/logs
module load PyTorch/2.7.0-CONDA
cd .
export PYTHONUNBUFFERED=1

D=./data
FOLD=$SLURM_ARRAY_TASK_ID
DATA=$D/${COHORT}/fold${FOLD}
OUT=$D/${RUNS}/fold${FOLD}
mkdir -p "$OUT"
echo "== ladder, cohort=$COHORT fold=$FOLD on $(hostname) =="; nvidia-smi -L

for ARCH in resnet18 resnet50 convnext_tiny vit_b_16 vit_b_16_swag; do
    if [ ! -f "$OUT/probe_${ARCH}_seed0.json" ]; then
        echo "== $COHORT fold=$FOLD $ARCH =="
        python src/run_probe_ladder.py --data "$DATA" --out "$OUT" \
            --arch "$ARCH" --seed 0 --epochs 15 --bs 256 --device cuda
    fi
done

echo "== ladder summary, $COHORT fold $FOLD =="
python -c "
import json, glob, os
for f in sorted(glob.glob('$OUT/probe_*.json')):
    r = json.load(open(f))
    print(f\"{r['arm']:<24} pretrain={r.get('pretrain_images',0):.2e} AUROC={r['ood_test']['auroc']:.4f} ECE={r['ood_test']['ece']:.4f}\")
"
