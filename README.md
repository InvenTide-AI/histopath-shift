# Deploying AI pathology models at new hospitals

Code and run records for

> A. Chatterjee and A. Mukherjee. *Deploying AI pathology models at new hospitals.*
> Submitted to *Nature Communications*, 2026.

Every table and figure can be regenerated on a CPU from the records in this repository. Re-training every model
from scratch needs the public datasets and about 34 GPU-hours (NVIDIA H100).

## Overview

Can a pathology model trained at a few hospitals be used at a new one, and how can that hospital decide? We hold
out every site of four public cohorts in turn and train up to 21 methods with several seeds at two model scales
(4,778 runs). Hyperparameters and checkpoints are never selected on test data.

| Cohort | Task | Site | Shift | Sites |
|---|---|---|---|---|
| Camelyon17-WILDS | tumor vs normal (lymph node) | hospital | institutional | 5 |
| Canine cutaneous SCC | tumor vs non-tumor | scanner (same slides) | acquisition only | 5 |
| MIDOG 2021 | mitotic figure vs look-alike | scanner (one lab) | acquisition only | 3 |
| MIDOG++ | mitotic figure vs look-alike | tumor type | institutional | 7 |

We propose a **site acceptance test**: an empirical-Bayes procedure that combines a model's development results
with a small labeled sample from the new site and accepts the site if P(AUROC ≥ target) ≥ 0.95. At standard scale
it kept false acceptance at or below 2.1% (development results alone: 8.6% at a target of 0.80) and, for targets
of 0.80 and 0.85, needed about half as many labels as local labels alone.

## Site acceptance test

`src/v2/site_acceptance.py` has no dependency on the rest of the pipeline (NumPy, SciPy).

```python
import site_acceptance as SA                       # src/v2 on PYTHONPATH
from scipy.special import logit
Z = logit(dev_auroc)                               # (n_dev_sites, n_runs): leave-one-site-out AUROCs of the model
prior = SA.fit_prior(Z)                            # or SA.robustify(SA.fit_prior(Z)) for the robust variant
m, curve = SA.labels_needed(prior, target=0.85)    # labels to collect at the new site
z, v, _, _ = SA.local_estimate(y_local[None], p_local[None])   # labels and model scores of the local sample
post = SA.posterior(prior, z, v)
print(SA.prob_above(post, logit(0.85)), SA.accept(post, 0.85))
```

## Reproduce every table and figure (CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export HISTOPATH_DATA=$PWD/data   # records are unpacked here
make records     # verify checksums, unpack records/ into $HISTOPATH_DATA/v2
make test        # unit tests of the training code and the statistics
make analysis    # statistics under all four selection rules, both tiers (~15 min on 8 cores)
make sat         # validation of the site acceptance test (~45 min)
make tables      # outputs/tables/*.tex and results/v2/analysis/paper_numbers.json
make figures     # outputs/figs/*.{pdf,png}
```

Re-training from the public datasets (fold caches, tuning, final runs) is described in
[docs/REPRODUCE.md](docs/REPRODUCE.md). Data and weight sources and licences are in [docs/DATA.md](docs/DATA.md),
and the formats of the fold caches and run records in [docs/CONTRACTS.md](docs/CONTRACTS.md).

### Where each item of the article comes from

| Article | Output |
|---|---|
| Fig. 1 | schematic, not generated from data |
| Fig. 2, Table 2 | `figs/fig_sat_S`, `tables/tab_sat_S.tex` |
| Fig. 3 | `figs/fig_perdomain_S` |
| Fig. 4 | `figs/fig_variance_S` (a, b), `figs/fig_power_S` (c) |
| Fig. 5 | `figs/fig_ranks_S` |
| Fig. 6 | `figs/fig_roadmap` |
| Table 1 | `tables/tab_main_S.tex` |
| Supplementary Figs. S1–S19 | `fig_design`, `fig_sat_C`, `fig_kendall_S`, `fig_kendall_C`, `fig_perdomain_C`, `fig_variance_C`, `fig_power_C`, `fig_ranks_C`, `fig_seedspread`, `fig_scale`, `fig_selection_S`, `fig_selection_C`, `fig_calibration_S`, `fig_calibration_C`, `fig_ladder`, `fig_sensitivity_C`, `fig_tuning_S`, `fig_tuning_C`, `fig_domainbed` |
| Supplementary Tables S4–S11 | `tab_sat_C`, `tab_sat_robust_{S,C}`, `tab_sat_cohorts`, `tab_evidence`, `tab_main_C`, `tab_selection`, `tab_pcam`, `tab_response` |
| Supplementary Table S18 | `results/v2/external/domainbed_icc_training_domain.tex` |

## Repository layout

```
src/v2/              training (run_v2.py, arms.py, backbones.py, stain.py, ssl_v2.py), planning (make_plan.py,
                     select_hparams.py), statistics (stats_v2.py, analyze_v2.py), site acceptance test
                     (site_acceptance.py), tables (paper_tables.py), figures (make_figures_v2.py)
src/v2/data/         fold-cache construction and verification for every cohort
src/v2/tests/        unit tests of the training code;  src/v2/stats_tests/  tests of the statistics
records/             records of all 4,778 runs and the test predictions of every final run -> records/README.md
results/v2/          selected hyperparameters (input to the final runs) and the DomainBed example
slurm/v2/            SLURM wrappers used for training
scripts/             record unpacking, dataset and weight download helpers
docs/                reproduction guide, data sources, file formats
legacy_v1/           records of the earlier pipeline behind Supplementary Note S15 -> legacy_v1/README.md
```

## Notes

- Transductive methods (BN-adapt, TENT, per-slide Macenko, TIA-style) use unlabeled images of the test site at
  prediction time and are marked with † in every table and figure.
- Run records keep their provenance fields (code hash, fold-cache fingerprint) and match the frozen snapshot
  `1381bdd759f9`; cluster paths in them are replaced by `$HISTOPATH_DATA`.
- Datasets and pretrained weights are not redistributed. The Lunit weights are for non-commercial research use
  only, and the SWAG weights are CC BY-NC 4.0.

## Licence

Code: MIT ([LICENSE](LICENSE)). Run records and derived results: CC BY 4.0. Datasets and weights keep their own
licences ([docs/DATA.md](docs/DATA.md)).

## Citation

See [CITATION.cff](CITATION.cff).
