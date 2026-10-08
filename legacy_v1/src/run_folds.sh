#!/usr/bin/env bash
# Four new leave-one-hospital-out folds, four arms each, on the published protocol.
# Fold 2 is NOT here: it is the published run (train {0,3,4}, ood_val 1, test 2).
#
# Measured cost on the CPU box the published runs used: ~71 h per fold at one
# seed per arm (baseline 16.7 h, sam 18.4 h, ssl 16.7 h, ssl_sam 18.4 h, plus
# 0.9 h SSL pre-training), so ~12 days for four folds -- ~24 days at two seeds.
# A GPU is strongly advised.  SEEDS="0 1" doubles cost and halves the standard
# error on every contrast.
set -euo pipefail
CODE=${CODE:-.}                 # dir holding run_arm.py / pretrain_ssl.py
SEEDS=${SEEDS:-0}
THREADS=${THREADS:-2}

# One-time: fetch the 21 parquet shards and decode, once, every patch any fold needs
# (389,716 rows -> union_store/patches_union.npy, 4.79 GB; shards deleted as it goes,
# resumable).  Then each fold's cache is a gather over that store.
python build_union_store.py --out union_store

for t in 0 1 3 4; do
  python assemble_fold_cache.py --fold "$t" --store union_store --out cache
  for s in $SEEDS; do
    # one encoder per (fold, seed): run_arm.py loads ssl_encoder_seed${s}.pt from
    # --data, and an encoder pre-trained on another fold's hospitals would leak
    # this fold's test hospital.
    python "$CODE/pretrain_ssl.py" --data "cache/fold$t" --seed "$s" --threads "$THREADS"
    for arm in baseline sam ssl ssl_sam; do
      python "$CODE/run_arm.py" --data "cache/fold$t" --out "runs/fold$t" \
             --arm "$arm" --seed "$s" --ntrain 30000 --threads "$THREADS"
    done
  done
done
echo "done; per-run metrics in runs/fold*/ -- aggregate with the released aggregate.py"
