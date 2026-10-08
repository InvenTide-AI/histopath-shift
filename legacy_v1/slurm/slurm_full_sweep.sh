#!/bin/bash
#SBATCH --job-name=histopath-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=./data/logs/full_%A_%a.out
#SBATCH --error=./data/logs/full_%A_%a.err
#SBATCH --array=0-4

# One SLURM task per fold; each task runs the 8 arms sequentially on one H100.
set -euo pipefail
mkdir -p ./data/logs
module load PyTorch/2.7.0-CONDA
cd .
export PYTHONUNBUFFERED=1

FOLD=$SLURM_ARRAY_TASK_ID
DATA=./data/cache/fold${FOLD}
OUT=./data/runs/fold${FOLD}
mkdir -p "$OUT"
echo "== fold $FOLD on $(hostname) =="; nvidia-smi -L

if [ ! -f "$DATA/ssl_encoder_seed0.pt" ]; then
    python src/pretrain_ssl_gpu_ext.py --data "$DATA" --seed 0 --epochs 6 --bs 256 --device cuda
fi

for arm in baseline sam ssl ssl_sam groupdro coral irm mixstyle; do
    if [ ! -f "$OUT/${arm}_seed0.json" ]; then
        echo "== fold=$FOLD $arm =="
        python src/run_arm_ext.py --data "$DATA" --out "$OUT" --arm "$arm" \
            --seed 0 --epochs 5 --bs 128 --lr 0.02 --device cuda --threads 8
    fi
done

echo "== fold $FOLD summary =="
python -c "
import json, os, glob
for f in sorted(glob.glob('$OUT/*.json')):
    r = json.load(open(f))
    a = r['ood_test']['auroc']; e = r['ood_test']['ece']; wall = r['wall_s']
    print(f'{os.path.basename(f).replace(\".json\",\"\"):<20} AUROC={a:.4f}  ECE={e:.4f}  wall={wall/60:.1f}m')
"
