#!/bin/bash
#SBATCH --job-name=canine-complete
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=./data/logs/canine_%A_%a.out
#SBATCH --error=./data/logs/canine_%A_%a.err
#SBATCH --array=0-4

# Scanner-complete grid on the multi-scanner canine SCC cohort, one task per
# held-out scanner. This cohort has K=5 domains and 24.5k-28.6k training
# patches per fold, so the Camelyon17 schedule transfers unchanged: same
# optimizer, same five epochs, same batch size, same held-out-domain selection
# rule. That makes it the one cohort here that is protocol-identical to
# Camelyon17 rather than protocol-adapted.
set -euo pipefail
mkdir -p ./data/logs
module load PyTorch/2.7.0-CONDA
cd .
export PYTHONUNBUFFERED=1

FOLD=$SLURM_ARRAY_TASK_ID
DATA=./data/canine_cache/fold${FOLD}
OUT=./data/canine_runs/fold${FOLD}
mkdir -p "$OUT"
echo "== canine fold $FOLD on $(hostname) =="; nvidia-smi -L
python -c "import json; m=json.load(open('$DATA/manifest.json')); print('  test:', m['test_scanner'], '| ood_val:', m['ood_val_scanner'], '| train:', m['train_scanners'], '| n_train', m['n_train'], 'n_test', m['n_ood_test'])"

for SEED in 0 1 2; do
    if [ ! -f "$DATA/ssl_encoder_seed${SEED}.pt" ]; then
        python src/pretrain_ssl_gpu_ext.py --data "$DATA" --seed "$SEED" \
            --epochs 6 --bs 256 --device cuda
    fi
done

for arm in baseline sam ssl ssl_sam; do
    for SEED in 0 1 2; do
        if [ ! -f "$OUT/${arm}_seed${SEED}.json" ]; then
            echo "== fold=$FOLD $arm seed=$SEED =="
            python src/run_arm_ext.py --data "$DATA" --out "$OUT" --arm "$arm" \
                --seed "$SEED" --epochs 5 --bs 128 --lr 0.02 \
                --device cuda --threads 8
        fi
    done
done

for arm in groupdro coral irm mixstyle macenko fish lisa erm_henorm rotinv tia_style; do
    if [ ! -f "$OUT/${arm}_seed0.json" ]; then
        echo "== fold=$FOLD $arm =="
        python src/run_arm_ext.py --data "$DATA" --out "$OUT" --arm "$arm" \
            --seed 0 --epochs 5 --bs 128 --lr 0.02 --device cuda --threads 8
    fi
done

if [ ! -f "$OUT/probe_resnet50_seed0.json" ]; then
    echo "== fold=$FOLD probe_resnet50 =="
    python src/run_arm_probe.py --data "$DATA" --out "$OUT" --seed 0 \
        --epochs 15 --bs 256 --device cuda
fi

echo "== fold $FOLD summary =="
python -c "
import json, os, glob
rows = []
for f in sorted(glob.glob('$OUT/*.json')):
    r = json.load(open(f))
    rows.append((os.path.basename(f)[:-5], r['ood_test']['auroc'], r['ood_test']['ece']))
for n, a, e in sorted(rows, key=lambda t: -t[1]):
    print(f'{n:<22} AUROC={a:.4f}  ECE={e:.4f}')
"
