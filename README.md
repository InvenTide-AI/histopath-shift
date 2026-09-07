# Self-Supervised Pre-Training Drives Cross-Hospital Generalization in Histopathology

Code, result records and reproduction protocol for the manuscript submitted to
*IEEE Transactions on Medical Imaging*. Every number in its four tables and its
figure can be re-derived from the files in this repository.

## The finding

A four-arm factorial — supervised baseline (ERM), self-supervised pre-training
(SSL), sharpness-aware minimization (SAM), and both — was run on
Camelyon17-WILDS with **each of the five hospitals held out in turn**.

| | ERM | SAM | SSL | SSL+SAM |
|---|---|---|---|---|
| Mean off-site AUROC | 0.851 | 0.848 | 0.943 | 0.947 |
| SD across hospitals | 0.127 | 0.098 | **0.023** | **0.014** |
| Range across hospitals | 0.287 | 0.250 | 0.057 | 0.030 |

* **Which hospital you hold out matters more than which method you use.**
  Baseline off-site AUROC ranges from 0.667 to 0.954 across the five held-out
  hospitals — a 0.287 swing produced by the choice of test site alone, larger
  than any training intervention produced.
* **SSL pre-training compresses that spread and raises the floor.** It cuts the
  hospital-to-hospital SD 5.6-fold (0.127 → 0.023) and lifts the worst hospital
  (C4) by +0.274 AUROC. The gain is concentrated where ERM fails: Pearson
  *r* = −0.98 (n = 5) between baseline AUROC at a hospital and the gain there.
* **The SSL × SAM synergy we previously reported at one hospital does not
  replicate.** The interaction was +0.161 AUROC at center 2 alone; across all
  five hospitals it is +0.007 ± 0.047 (mean ± SD), with a 95% CI that covers
  zero and a sign that flips between hospitals. Single-site evidence for the
  interaction was a property of that site, not of the methods.

The last point is why this repository ships the earlier single-hospital
experiment alongside the five-hospital sweep rather than replacing it: the
single-hospital result is reproducible and is reported in the paper as Table
III, but it is superseded as evidence by the sweep in Table II.

## Layout

```
src/                    pipeline: data prep, SSL pre-training, training arms,
                        aggregation, bootstrap CIs, sharpness, site shift, figures
src/check_release.py    re-derives every published table from results/ (112 checks)
src/make_figure_fivefold.py   regenerates Fig. 1 from results/
protocol/               fold protocol, hospital splits, dataset manifests and
                        shard indices — the exact sample assignment used
results/
  camelyon17_fivehospital/   Table II, Fig. 1 — the five-hospital sweep
  camelyon17_center2/        Table III — the single held-out hospital, 2 seeds
  nct_crc_he/                Table IV — colorectal replication, 2 shift axes
  site_characterization/     hospital separability, stain distance, case mix
  chest_xray_incomplete/     partial third task, not reported in the paper
runs/camelyon17_center2/     per-run JSON records for the center-2 experiment
```

## Reproducing the paper from the shipped results

No dataset download or GPU required — these read only the CSVs in `results/`.

```bash
pip install -r requirements.txt
python src/check_release.py            # 112 checks: shipped tables vs. paper
python src/make_figure_fivefold.py     # -> paper/fig1_site_complete_factorial.pdf
```

`make check` and `make figures` do the same. Each published item maps to one file:

| Paper item | File | Produced by |
|---|---|---|
| Table II, Fig. 1 | `results/camelyon17_fivehospital/results_folds.csv` | `drive_folds.py` → `analyze_sweep.py` |
| Table II summary rows | `results/camelyon17_fivehospital/spread_by_arm.csv` | `analyze_sweep.py` |
| Per-hospital contrasts | `results/camelyon17_fivehospital/contrasts_folds.csv` | `analyze_sweep.py` |
| Table III | `results/camelyon17_center2/results_raw.csv` | `run_arm.py` → `aggregate.py` |
| Table III CIs | `results/camelyon17_center2/bootstrap_ci.csv` | `bootstrap_ci.py` |
| Sharpness analysis | `results/camelyon17_center2/flatness_metrics.csv` | `flatness.py` |
| Table IV | `results/nct_crc_he/crc_results_summary.csv` | `run_all_crc.py` → `aggregate_crc.py` |
| Table IV decomposition | `results/nct_crc_he/crc_interaction.csv` | `mechanisms.py` |
| Site characterization | `results/site_characterization/site_shift.csv` | `site_shift.py` |

Note on Table IV: the decomposition printed in the paper is computed from the
seed-averaged arm means in `crc_results_summary.csv`. `crc_interaction.csv`
holds the paired bootstrap estimate of the same terms, which differs slightly
(interaction +0.078 vs. +0.087) because it resamples patches rather than
averaging seeds. `check_release.py` verifies the published values against the
estimator the paper used.

## Reproducing from raw data

Datasets are public and are not redistributed here:

* **Camelyon17-WILDS** — CC0, via the WILDS benchmark. `protocol/data_manifest.json`
  and `protocol/hospital_splits_manifest.json` record the exact patch and patient
  assignment; `src/verify_cache.py` checks a rebuilt cache is sample-identical to
  the one the runs used.
* **NCT-CRC-HE** — CC-BY, from Zenodo. `protocol/crc_data_manifest.json` and the
  `protocol/crc_rg_index_*.json` shard indices record the assignment.

Both are de-identified public releases; no new patient data were collected and
no IRB approval was required.

```bash
python src/prepare_data.py              # build the Camelyon17 patch cache
python src/verify_cache.py              # confirm it matches the shipped manifest
python src/make_matched_folds.py        # five-hospital fold protocol
python src/materialize_fold_cache.py    # per-fold caches
bash   src/run_folds.sh                 # 20 runs: 4 arms x 5 held-out hospitals
python src/analyze_sweep.py             # -> results/camelyon17_fivehospital/*.csv
```

**Compute.** Everything is CPU-only; there is no GPU code path in the reported
experiments. On the 10-core macOS host used for the paper: the five-hospital
sweep is 20 runs / 23.3 h wall (38 min median per run), the center-2 experiment
is 6 longer runs / 91.3 h, and the NCT-CRC-HE replication is 8 runs / 27.9 h —
about 143 CPU-hours in total, read from the `wall_s` fields of the shipped run
records.

## What is not in this repository

Stated explicitly so the release is not read as more complete than it is.

* **Per-run JSON records for the five-hospital sweep were not retained.** Only
  the aggregated tables (`results_folds.csv`, `contrasts_folds.csv`,
  `spread_by_arm.csv`) survive for that experiment, so Table II and Fig. 1 are
  reproducible from the shipped data but the underlying 20 run records are not
  included. Per-run JSON *is* shipped for the center-2 experiment
  (`runs/camelyon17_center2/`).
* **One seed per arm per hospital in the sweep.** The five-hospital design trades
  seed replication for site coverage; between-hospital variation is therefore
  estimated across hospitals, not across seeds, and per-hospital interaction
  estimates carry no within-hospital error bar.
* **No trained model weights.** Checkpoints were not retained; the pipeline
  retrains from scratch.
* **The chest-radiograph task is incomplete** and is not reported in the paper.
  Its partial results are kept in `results/chest_xray_incomplete/` with a note,
  and its scripts in `src/*_cxr.py`, for provenance only. Do not cite them.
* **The manuscript itself is not included.** The submitted PDF, its design/
  revision-history notes, and citation metadata are tracked separately from
  this code release.

## Citation

Code is MIT-licensed (`LICENSE`); the result tables are not covered by that
licence.
