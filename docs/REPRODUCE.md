# Full reproduction

The shipped records already reproduce every table and figure (`make records analysis tables figures`). This
guide re-creates the records themselves. Commands assume the repository root as working directory,
`export HISTOPATH_DATA=/path/to/data`, and a SLURM cluster with one GPU per array task (adapt
`slurm/v2/run_plan.sh`: partition, modules, time). Each plan shard is one (tier, cohort, fold); `run_v2.py` skips
jobs whose record already exists for the same cache fingerprint and hyperparameters, so every step can be resumed.

## 1. Data, caches and weights

```bash
# download the cohorts (docs/DATA.md), then build the fold caches at 64 and 96 px
python src/v2/data/prepare_c17.py      --wilds_dir $HISTOPATH_DATA/wilds/camelyon17_v1.0
python src/v2/data/prepare_canine.py   --src $HISTOPATH_DATA/canine_scc
python src/v2/data/prepare_midog21.py  --src $HISTOPATH_DATA/midog                 # un-normalized (sensitivity)
python src/v2/data/prepare_midog21.py  --src $HISTOPATH_DATA/midog --scale_norm    # primary: midog21sn
python src/v2/data/prepare_midogpp.py  --src $HISTOPATH_DATA/midogpp
python src/v2/data/verify_v2.py        # disjointness, duplicates, class balance, 64/96 px consistency
python scripts/fetch_backbones.py
```

Each cache carries a fingerprint; every run record stores it, so records and caches can be matched.

## 2. Hyperparameter tuning (seed 0, selection on ID validation only)

```bash
python src/v2/make_plan.py --phase tune --tier C                        # method grids, compact tier
slurm/v2/submit.sh slurm/v2/plans/tune/shards_C.txt
python src/v2/make_plan.py --phase tune --tier S --stage lr             # ResNet-50 SGD and ViT AdamW LR grids
slurm/v2/submit.sh slurm/v2/plans/tune_lr/shards_S.txt
python src/v2/select_hparams.py
python src/v2/make_plan.py --phase tune --tier S --stage arms           # method grids at each fold's LR
slurm/v2/submit.sh slurm/v2/plans/tune/shards_S.txt
python src/v2/make_plan.py --phase tune --tier S --stage lr_bt          # Barlow Twins fine-tunes: own LR grid
slurm/v2/submit.sh slurm/v2/plans/tune_lr_bt/shards_S.txt
python src/v2/select_hparams.py          # -> results/v2/selected_hparams.json (per fold), tuning_results.csv
```

Hyperparameters are selected per fold, never pooled across folds (pooling would leak test-domain labels).
`make_plan.py` rewrites `shards_*.txt`; copy a shard list before regenerating it.

## 3. Final runs

```bash
python src/v2/make_plan.py --phase final --tier C,S      # every arm, 5 seeds (C) / 3 seeds (S), all folds
slurm/v2/submit.sh slurm/v2/plans/final/shards_CS.txt --time 08:00:00
python src/v2/make_plan.py --phase final --tier C --cohorts midog21     # un-normalized MIDOG 2021 sensitivity
python src/v2/make_plan.py --phase ladder --tier S                      # frozen-probe representation ladder
slurm/v2/submit.sh slurm/v2/plans/ladder/shards_S.txt --time 04:00:00
```

SimCLR pre-training runs inside the first job that needs it (or ahead of time with `slurm/v2/ssl.sh`).
Test-time adaptation and the PatchCamelyon score are evaluated inside the base arm's run.

The original sweep was submitted in several parts because two arms were fixed after the first submission:
the Macenko family (per-slide estimation, `plans/final_mac`) and the Barlow Twins fine-tunes (own learning
rate, `plans/final_bt`). All plans that were actually run are in `slurm/v2/plans/`. The superseded records are
shipped separately (`records/v2_quarantine_*`) because two appendix tables use them.

## 4. Analysis, tables, figures

```bash
SNAP=$HISTOPATH_DATA/v2/snap_final2_20261005
python src/v2/make_figures_v2.py --runs $HISTOPATH_DATA/v2 --freeze $SNAP --analysis results/v2/analysis --freeze-only
make analysis tables figures HISTOPATH_DATA=$HISTOPATH_DATA
```

The site acceptance test is validated with `make sat` (both tiers, ~45 min) and tabulated by `make tables`.

`--freeze` copies the records into one snapshot directory with a manifest (`SNAPSHOT.json`); the analysis and
figure scripts refuse to mix records from different snapshots.

## Compute

About 34 GPU-hours on NVIDIA H100 NVL for the 4,778 tuning and final runs (development and diagnostic runs not
included). Six final runs diverged, all IRM; they are kept in the analysis as recorded.
