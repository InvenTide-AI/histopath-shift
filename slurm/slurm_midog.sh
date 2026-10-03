#!/bin/bash
#SBATCH --job-name=midog-complete
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=./data/logs/midog_%A_%a.out
#SBATCH --error=./data/logs/midog_%A_%a.err
#SBATCH --array=0-2

# Scanner-complete grid on MIDOG 2021: one SLURM task per held-out scanner.
#
# Schedule note. MIDOG carries ~2.8k training patches per fold against
# Camelyon17's 30k, so matching *epochs* would give roughly a tenth of the
# optimizer steps and would not be the same training pipeline in any
# meaningful sense. We match total steps instead: 30000/128*5 = 1172 steps on
# Camelyon17, and 2800/128*50 = 1094 here. Everything else (optimizer, lr,
# schedule shape, batch size, selection rule) is unchanged. The IRM annealing
# epoch is scaled by the same factor to keep it at 40% of training.
set -euo pipefail
mkdir -p ./data/logs
module load PyTorch/2.7.0-CONDA
cd .
export PYTHONUNBUFFERED=1

EPOCHS=50
IRM_ANNEAL=20
FOLD=$SLURM_ARRAY_TASK_ID
DATA=./data/midog_cache/fold${FOLD}
OUT=./data/midog_runs/fold${FOLD}
mkdir -p "$OUT"
echo "== MIDOG fold $FOLD on $(hostname) =="; nvidia-smi -L
python -c "import json; m=json.load(open('$DATA/manifest.json')); print('  test scanner:', m['test_scanner'], '| train:', m['train_scanners'], '| n_train', m['n_train'], 'n_test', m['n_ood_test'])"

# SSL encoders, one per seed, pre-trained only on this fold's own unlabelled
# pool (crops from the training scanners; never the test scanner).
for SEED in 0 1 2; do
    if [ ! -f "$DATA/ssl_encoder_seed${SEED}.pt" ]; then
        python src/pretrain_ssl_gpu_ext.py --data "$DATA" --seed "$SEED" \
            --epochs 6 --bs 256 --device cuda
    fi
done

# Core arms: three seeds each.
for arm in baseline sam ssl ssl_sam; do
    for SEED in 0 1 2; do
        if [ ! -f "$OUT/${arm}_seed${SEED}.json" ]; then
            echo "== fold=$FOLD $arm seed=$SEED =="
            python src/run_arm_ext.py --data "$DATA" --out "$OUT" --arm "$arm" \
                --seed "$SEED" --epochs "$EPOCHS" --bs 128 --lr 0.02 \
                --device cuda --threads 8
        fi
    done
done

# Extended arms, one seed each. rotinv and tia_style are the MIDOG-specific
# published top performers (Lafarge & Koelzer; Jahanifar et al.).
for arm in groupdro coral irm mixstyle macenko fish lisa erm_henorm rotinv tia_style; do
    if [ ! -f "$OUT/${arm}_seed0.json" ]; then
        echo "== fold=$FOLD $arm =="
        python src/run_arm_ext.py --data "$DATA" --out "$OUT" --arm "$arm" \
            --seed 0 --epochs "$EPOCHS" --bs 128 --lr 0.02 \
            --irm_anneal_epoch "$IRM_ANNEAL" --device cuda --threads 8
    fi
done

# Frozen off-the-shelf reference.
if [ ! -f "$OUT/probe_resnet50_seed0.json" ]; then
    echo "== fold=$FOLD probe_resnet50 =="
    python src/run_arm_probe.py --data "$DATA" --out "$OUT" --seed 0 \
        --epochs 40 --bs 256 --device cuda
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
