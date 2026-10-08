# Domain-Complete Evaluation of Distribution Shift in Histopathology

Code, experimental records, and reproduction protocol for the manuscript
submitted to *Transactions on Machine Learning Research (TMLR)*.

Every number in the paper's tables and figures can be re-derived from the files
in this repository.

## Key finding

The **type** of domain shift—not just its presence—determines which robustness
method works for histopathology AI.

We tested eleven interventions across three cohorts that isolate different
sources of shift:

| Cohort | What changes between domains | Best method (worst-site AUROC) |
|---|---|---|
| **Camelyon17-WILDS** (5 hospitals) | Scanner + stain + patients | Frozen ImageNet probe (0.929) |
| **MIDOG 2021** (3 scanners, 1 lab) | Scanner only | Stain normalization (0.865) |
| **Canine SCC** (5 scanners) | Scanner only | Frozen ImageNet probe (0.792) |

**Under hospital shift** (scanner, stain, and patient population all change),
self-supervised pre-training on unlabelled tissue lifts the worst hospital's
AUROC by +0.175 and compresses the across-hospital spread 3.2×. Under
**scanner-only shift**, stain normalization dominates and pre-training adds
nothing. Most domain-generalization objectives (GroupDRO, IRM, DeepCORAL) hurt
the worst domain on every cohort.

We propose the **intraclass correlation coefficient (ICC)** from a variance
decomposition of off-site performance as a quantitative measure for evaluating
robustness claims. The ICC separates true site sensitivity from random-seed
noise and yields a closed-form power formula: detecting a 0.10 AUROC mean
effect requires at least 8 hospitals at 3 seeds.

## Layout

```
src/                        Pipeline: data prep, SSL pre-training, training arms,
                            aggregation, variance decomposition, figures
paper/tmlr/                 Manuscript source (LaTeX) and compiled PDF
paper/tmlr/figs/            All figures (PDF)
protocol/                   Fold protocols, hospital splits, dataset manifests
results/
  camelyon17_fivehospital/  Five-hospital sweep (Tables 1–3, Figs 1–2)
  camelyon17_extended/      Extended grid: 13 arms × 5 hospitals
  camelyon17_cohort/        Variance decomposition for Camelyon17
  midog/                    Scanner-complete grid: 15 arms × 3 scanners
  canine/                   Scanner-complete grid: 15 arms × 5 scanners
  cohorts/                  Cross-cohort summary (Table 4)
  camelyon17_center2/       Single held-out hospital (legacy experiment)
  nct_crc_he/               NCT-CRC-HE replication (Appendix D)
  rules/                    Response-shape analysis
  site_characterization/    Hospital separability, stain distance, case mix
runs/camelyon17_center2/    Per-run JSON records for the center-2 experiment
slurm/                      Example SLURM job scripts (adapt paths for your cluster)
```

## Quick start: reproduce the paper from shipped results

No dataset download or GPU required—these read only the CSVs and JSONs in
`results/`.

```bash
pip install -r requirements.txt

# Verify shipped tables match the paper
python src/check_release.py

# Regenerate figures
python src/make_figures_tmlr.py          # Main-text figures
python src/make_figures_cohorts.py       # Cross-cohort figures
python src/emit_tables.py               # LaTeX tables
```

`make check` and `make figures` provide shortcuts.

## Reproducing from raw data

Datasets are public and are **not** redistributed here:

- **Camelyon17-WILDS** — CC0, via the [WILDS benchmark](https://wilds.stanford.edu/).
  `protocol/data_manifest.json` records the exact patch assignment.
- **MIDOG 2021** — CC-BY, from [Zenodo](https://zenodo.org/records/4573978).
- **Multi-scanner canine SCC** — CC-BY, from [Zenodo](https://zenodo.org/records/7548828).
- **NCT-CRC-HE / CRC-VAL-HE-7K** — CC-BY, from [Zenodo](https://zenodo.org/records/1214456).
- **PatchCamelyon** — CC0, from [Zenodo](https://zenodo.org/records/2546921).

All are de-identified public releases; no new patient data were collected and
no ethics approval was required.

### Camelyon17 full pipeline

```bash
# 1. Download and prepare data
python src/materialize_wilds_folds.py --wilds_dir ./data/wilds/camelyon17_v1.0

# 2. Self-supervised pre-training (one per fold)
python src/pretrain_ssl_gpu_ext.py --fold 0 --data_root ./data/cache/fold0

# 3. Run all arms across all folds
python src/drive_folds_ext.py --data_root ./data/cache --out_root ./data/runs

# 4. Aggregate results
python src/aggregate_ext.py --runs_root ./data/runs

# 5. Evaluate on PatchCamelyon
python src/eval_pcam.py --pcam_root ./data/pcam --runs_root ./data/runs
```

### MIDOG and canine cohorts

```bash
python src/fetch_midog.py               # Download MIDOG data
python src/prepare_midog.py             # Prepare patches
python src/prepare_canine.py            # Prepare canine cohort
# Then run the same drive_folds_ext.py / aggregate pipeline
```

See `slurm/` for example SLURM job scripts.

### Compute

At compact-model scale (0.58M-parameter CNN, 64×64 input):
- **Camelyon17 grid**: 180 GPU jobs, 0.53 H100-hours
- **Full study (3 cohorts)**: 388 GPU jobs, 1.23 H100-hours

## Methods tested

| Arm | Type | Reference |
|---|---|---|
| ERM (baseline) | Supervised | — |
| SAM | Optimizer | Foret et al., 2021 |
| SSL (SimCLR) | Pre-training | Chen et al., 2020 |
| SSL + SAM | Pre-training + optimizer | — |
| GroupDRO | Domain generalization | Sagawa et al., 2020 |
| DeepCORAL | Domain generalization | Sun & Saenko, 2016 |
| IRMv1 | Domain generalization | Arjovsky et al., 2019 |
| MixStyle | Domain generalization | Zhou et al., 2021 |
| Macenko + ERM | Stain normalization | Macenko et al., 2009 |
| Fish | Gradient matching | Shi et al., 2022 |
| LISA | Selective augmentation | Yao et al., 2022 |
| ERM + H&E jitter | Augmentation | Tellez et al., 2019 |
| Frozen probe (ImageNet) | Transfer learning | He et al., 2016 |
| TIA-style | Challenge winner | Jahanifar et al., 2024 |
| RotInv | Dihedral invariance | Lafarge & Koelzer, 2021 |

## Citation

If you use this code or these experimental records, please cite:

```bibtex
@article{chatterjee2026domcomplete,
  title   = {Domain-Complete Evaluation of Distribution Shift in Histopathology},
  author  = {Chatterjee, Ayan and Mukherjee, Amitava},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code is MIT-licensed (see `LICENSE`).
