# Train once, deploy across hospitals?

Code, run records and manuscript source for

> A. Chatterjee and A. Mukherjee. *Train once, deploy across hospitals? A domain-complete evaluation and an
> evidence-graded roadmap for cross-site generalization in computational pathology.* Submitted to
> *Medical Image Analysis*, 2026.

Every table and figure of the article can be regenerated on a CPU from the records in this repository.
Re-training every model from scratch needs the public datasets and about 34 GPU-hours (NVIDIA H100).

## What the study asks

Can a pathology model be trained once and then used at hospitals and scanners it has never seen, without
retraining? And how can a practitioner decide whether it is good enough at a particular new site? We hold out every domain of four public cohorts in turn, run every
method with several seeds under one fixed budget, and never use test data for tuning or model selection.

| Cohort | Task | Domain | Shift | Domains |
|---|---|---|---|---|
| Camelyon17-WILDS | tumor vs normal (lymph node) | hospital | institutional | 5 |
| Canine cutaneous SCC | tumor vs non-tumor | scanner (same slides) | acquisition only | 5 |
| MIDOG 2021 | mitotic figure vs look-alike | scanner (one lab) | acquisition only | 3 |
| MIDOG++ | mitotic figure vs look-alike | tumor type / lab / species | institutional | 7 |

Up to 21 methods are compared at two scales (a 0.58 M-parameter CNN at 64 px, five seeds; ImageNet ResNet-50
at 96 px, three seeds), together with Lunit pathology foundation models (Barlow Twins ResNet-50, DINO ViT-S/16)
frozen and fine-tuned: ERM, SAM, SimCLR, GroupDRO, IRM, CORAL, MixStyle, Fish, LISA, Macenko (per slide and per
tile), H&E color jitter, TIA-style normalization, BN-adapt, TENT, and compute-matched and augmentation-only
controls. In total 4,778 runs (1,378 tuning, 3,400 final, including a nine-backbone frozen-probe ladder).

## Main findings

- Without retraining, the best standard-scale method of each cohort kept a worst-site AUROC of 0.88–0.99,
  against 0.53–0.90 for ERM. No method was best everywhere.
- For most methods the held-out site explains most of the variance of a single run. With 3–7 sites, no
  standard-scale method beats ERM after correction for multiple comparisons; confirming a 0.05 AUROC gain on the
  tumor cohorts would need 7–53 sites.
- Fine-tuned pathology foundation models with color jitter were the most reliable recipe on average. Label-free
  batch-norm adaptation helped mainly under severe scanner shift. Under the primary selection rule no
  domain-generalization objective reached the top five on any cohort.
- At standard scale, rankings agree more between cohorts that share a task than between cohorts that share a
  shift type (descriptive; one cohort per cell).
- Development results alone would have accepted 8.6% of the sites that fell below an AUROC of 0.80. The proposed
  **site acceptance test** kept false acceptance at or below 2.1% at standard scale and needed about half as many
  local labels as local labels alone.

## The site acceptance test (`src/v2/site_acceptance.py`)

An empirical-Bayes test for a new hospital or scanner. Its prior is the posterior predictive distribution of one
run's logit AUROC at a new site, from the method's leave-one-site-out development results (one-way random-effects
model, half-Cauchy priors). It is updated with the AUROC of a small random labeled sample from the new site
(Hanley-McNeil variance), and the site is accepted if P(AUROC >= target) >= 0.95. A pre-posterior planner gives the
number of labels to collect, and an epsilon-contaminated variant guards against sites unlike all development sites.

```python
import numpy as np, site_acceptance as SA          # src/v2 on PYTHONPATH
from scipy.special import logit
Z = logit(dev_auroc)                               # (n_dev_sites, n_runs) leave-one-site-out AUROCs of the method
prior = SA.robustify(SA.fit_prior(Z))              # or SA.fit_prior(Z)
m, curve = SA.labels_needed(prior, target=0.85)    # labels to collect
z, v, _, _ = SA.local_estimate(y_local[None], p_local[None])   # labels and model scores of the local sample
post = SA.posterior(prior, z, v)
print(SA.prob_above(post, logit(0.85)), SA.accept(post, 0.85))
```

## The roadmap (Section 6.2 of the article)

1. **Define the target**: fix the task, the sites and a worst-site acceptance level before training.
2. **Collect sites, not only slides**: split by site; size the number of sites from a pilot.
3. **Train once, strongly**: fine-tune a pathology foundation model with color jitter; tune its own learning rate.
4. **Adapt without labels** (optional): re-estimate batch-norm statistics or normalize stain per slide.
5. **Select and validate** on training-site data only; hold out whole sites; report every site, seeds and ECE.
6. **Accept each new site** with the site acceptance test on a small labeled sample, then monitor.

## Quick start: regenerate the paper from the shipped records (CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export HISTOPATH_DATA=$PWD/data          # where records (and, optionally, datasets) live
make records      # unpack records/*.tar.xz into data/v2 (checksums verified)
make test         # unit tests of the training code and the statistics
make analysis     # all statistics, four selection rules, both tiers (~15 min on 8 cores)
make tables       # paper/media/tables/*.tex, results/v2/analysis/paper_numbers.json, site-test tables
make sat          # optional: re-run the site acceptance validation (~45 min)
make figures      # paper/media/figs/*
make paper        # paper/media/main.pdf and supplementary.pdf (needs TeX)
```

`make analysis tables` reproduces every shipped CSV in `results/v2/analysis` and every table in `paper/media/tables` byte for byte (only the `snapshot` path field of the `meta_*.json` files changes).

Full re-training (datasets, caches, tuning, final sweeps) is described in [docs/REPRODUCE.md](docs/REPRODUCE.md);
dataset and weight sources and licences are in [docs/DATA.md](docs/DATA.md).

## Repository layout

```
src/v2/                 pipeline: training (run_v2.py, arms.py, backbones.py, stain.py, ssl_v2.py),
                        planning (make_plan.py, select_hparams.py), statistics (stats_v2.py, analyze_v2.py),
                        figures (make_figures_v2.py), tables (paper_tables.py)
src/v2/data/            fold-cache construction and verification for every cohort
src/v2/tests/           unit tests (training);  src/v2/stats_tests/  tests and coverage simulations (statistics)
slurm/v2/               SLURM scripts and the exact job plans that were run (plans/*/)
records/                run records (JSON, xz-compressed) + SHA256SUMS + MANIFEST.json  -> records/README.md
results/v2/             analysis outputs, selected hyperparameters, tuning table, DomainBed external example
paper/media/            manuscript and supplement source (Elsevier CAS), figures, generated tables, PDFs
scripts/                record unpacking, weight and dataset download helpers
docs/                   reproduction guide, data sources, interface contracts, design plan, Macenko notes
legacy_v1/              earlier (v1) pipeline and records, used only by the two case studies -> legacy_v1/README.md
```

## Notes

- Transductive methods (BN-adapt, TENT, per-slide Macenko, TIA-style) use unlabeled test-domain images and are
  marked with † in every table and figure.
- Run records keep their original provenance fields (code hashes, cache fingerprints), so that they can be
  matched to the frozen snapshot (id `1381bdd759f9`); cluster paths in them are replaced by `$HISTOPATH_DATA`.
- Datasets and pretrained weights are not redistributed. The Lunit weights are for non-commercial research use
  only, and the SWAG weights are CC BY-NC 4.0.

## Licence

Code: MIT ([LICENSE](LICENSE)). Run records, result tables and figures: CC BY 4.0. Datasets and weights keep
their own licences ([docs/DATA.md](docs/DATA.md)).

## Citation

See [CITATION.cff](CITATION.cff).
