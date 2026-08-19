# SSL × SAM interaction under distribution shift in histopathology

Code, result records and manuscript source for the paper *(title in
`paper/main_journal.tex`)*, formatted for npj Digital Medicine.

**The headline result.** Self-supervised pre-training (SSL) and sharpness-aware
minimization (SAM) are *super-additive under distribution shift*. Across two
histopathology datasets and five evaluation axes, the SSL × SAM interaction term
is positive on every axis under shift (+0.004 to +0.161 AUROC), at every
available seed, while each mechanism applied alone is at or below the supervised
baseline — and the interaction vanishes in-distribution (−0.0001, −0.045). The
synergy is therefore specific to shift rather than a general optimization effect.

**The condition for net benefit.** Because the 2×2 decomposition is exact, this
converts "does SSL+SAM help?" into a measurable condition: with
`c = -(Δ_SSL + Δ_SAM)` the cost of each mechanism alone, the combination beats
the baseline exactly when the interaction `I > c`. That condition is met at an
unseen hospital (`I=+0.161` vs `c=+0.142`, net **+0.019** AUROC, ECE 5.1× lower)
and not met under a stain-appearance shift (`I=+0.087` vs `c=+0.154`, net
−0.067). The two datasets thus agree on mechanism and differ only in whether the
interaction is large enough to pay for itself.

The main text reports the interaction and the condition; full per-arm results for
the axes without a net gain are in the Supplementary Information. Reproducing
either requires no GPU — all runs are CPU-only (see **Compute** below).

## What is here

```
paper/          manuscript source, figures, build script
src/            data preparation, training, aggregation, sharpness, figures
src/verify_cache.py   checks a rebuilt cache is sample-identical to the runs
results/        per-run metric records and aggregated tables (three experiments)
runs_camelyon17/  per-run JSON records for the primary experiment
DESIGN_crc.md   design note for the replication (why its protocol differs)
DESIGN_crc_pool40k.md  design note for the nested-pool follow-up
DESIGN_cxr.md   frozen pre-registration for the incomplete third experiment
CITATION.cff    citation metadata (author list is a stub -- fill before release)
```

Result records are tracked in git; the ~5 GB decoded caches and model weights are
not (see `.gitignore`, which keeps the small `*_manifest.json` provenance files
inside `src/cache/` while excluding everything else there).

## Multi-hospital extension (in progress)

The published result above holds out ONE hospital. The extension generalises it
to all five Camelyon17 sites and to eight mechanisms, so that "which combination
helps under site shift?" is answered across shift magnitudes rather than at a
single one.

```
src/make_hospital_splits.py  leave-one-hospital-out splits, patient-disjoint
src/sample_site_images.py    per-site image sample for the model-free analysis
src/site_shift.py            stain distance + domain separability per fold
src/mechanisms.py            8 mechanisms (4 representation x 2 flatness)
src/test_mechanisms.py       unit tests: each mechanism does what it claims
src/train_gpu.py             ResNet-50 pipeline, any mechanism combination
src/run_sweep.py             resumable fold x mechanism sweep
src/analyze_sweep.py         2x2 interaction decomposition across folds
```

**Splits.** 5 folds, one per held-out hospital: 40k train / 8k in-distribution
val / 20k held-out test, class-balanced 50/50. No patient appears at more than
one site, so leave-one-hospital-out is clean; patient-disjointness, site purity
and index disjointness are verified independently of the builder
(`results/split_composition.csv`, `patient_overlap.csv`).

**Shift is graded, and the folds are not interchangeable.** Restricting to
normal tissue and equalising n per site (n=736, the smallest site's pool) —
necessary because the shards each site lives in are not class-balanced, so mixed
tissue would confound site with tumour content — stain distance to the training
sites ranges 0.34–0.56 across folds while domain separability sits near ceiling
(0.93–1.00). The two rank-correlate only ρ=0.30, so stain distance is the usable
graded shift variable. Since patient is nested in site, a control compares
patient-vs-patient separability within a site against across sites: 0.933 vs
1.000 median (p≈1e-54). Site shift is therefore real above patient variation,
but within-site patient variation alone already reaches 0.93 — worth stating
plainly when interpreting any site-level number.

**Data.** Patches are natively 96×96, so the decoded cache is stored at native
resolution (7.0 GB for the 251,496 patches the five folds reference) and
upsampled in the dataloader; caching at 224px would have stored pure
interpolation at 5.4× the disk. Both metadata and images stream over HTTP range
requests from the `wltjr1007/Camelyon17-WILDS` mirror, which carries the
center/patient/node/slide columns the WILDS release needs for site splits.
Sequential row-group reads run ~180 patches/s versus ~7/s for scattered
single-row reads, which is why `decode_patches.py` walks row groups in order.

**Verification.** `python src/verify_cache.py` (needs `pyarrow` + `torch`; run it in
the data environment, not the LaTeX one) covers the failure modes that would
corrupt a sweep silently rather than crash it: per fold, that the test set is
exactly one held-out centre, that centre is absent from training, and no patient
appears on both sides; then that the decoded array covers every row the folds
index, its length matches its manifest, and no patch is all-zero (an interrupted
decode leaves rows that train as black images without erroring). Pixel fidelity
against the source shards is a separate one-off check — 42 patches across all 21
shards were re-fetched and compared byte-for-byte, all identical — because it
needs 21 network reads.

**Status: the sweep has not run.** No GPU is configured, so *no model-based number
exists for the multi-hospital design*; the two findings above are model-free
measurements and stand on their own. The full design is ~1,280 CPU-hours here
versus roughly 1–2 days on one A100. The manuscript's title and abstract still
frame the earlier two-mechanism study, deliberately: retargeting them would commit
to a mechanism-ranking claim the sweep has not produced. `RUN_STATE.md` has the
launch checklist.

## The three experiments

**1. Camelyon17-WILDS (primary).** Four arms — ERM baseline, +SSL, +SAM,
SSL+SAM — trained on hospitals 0/3/4, tested on held-out hospital 2. Model
selection on the WILDS `ood_val` split (hospital 1). Results in
`results/camelyon17/`, per-run records in `runs_camelyon17/`. This is the
experiment behind Table 1 and Figures 2–4.

**2. NCT-CRC-HE (independent replication).** The same four arms on a different
tissue type and a different shift structure: stain/acquisition shift
(`NONORM`, primary axis) and a patient-disjoint external cohort
(CRC-VAL-HE-7K). Results in `results/nct_crc_he/`, behind Table 3 and Figure 5.

This replication **reproduces the interaction but not a net benefit**, which is
the distinction the paper is built on. The interaction term is positive and
bootstrap-separated from zero on *both* shift axes (+0.078 [0.070, 0.085] on
stain appearance; +0.005 [0.002, 0.008] external) and null in-distribution — the
same pattern as at the held-out hospital. But here it only *cancels* two negative
main effects rather than exceeding them, so no intervened arm reaches the
supervised baseline. This is the dataset that carries the seed evidence: it has
**both seeds for all four arms**, and the interaction is positive at each seed
independently on both shift axes.

Two ways of pooling seeds appear in the outputs and they are not
interchangeable — `aggregate_crc.py` bootstraps *seed-averaged predicted
probabilities* (a two-model ensemble, which is what permits a paired interval on
one test set), while the figures average *seed-wise metrics*. They agree in sign
and magnitude; the paper labels which is used where.

Its protocol differs from experiment 1 in two ways, both forced by the dataset
and both documented in `DESIGN_crc.md`: model selection is on in-distribution
validation, because NCT-CRC-HE has no third site to hold out; and two shift
test sets are scored instead of one.

**3. Chest radiographs (incomplete).** A cross-modality test, pre-registered in
`DESIGN_cxr.md` before any run. Nine of twelve planned runs completed before the
compute environment was lost; three exited with an error, and the arm the
pre-registered contrast turns on — SSL(multi-site)+SAM — never ran.

`results/chest_xray_incomplete/` holds the surviving point estimates.
**They admit no inference.** The pre-registered analysis is a paired bootstrap
over per-example predictions, which those files no longer contain, and seed
coverage across arms is uneven. They are released so the experiment's state at
the time of loss is on the record, not as evidence either way. The pipeline is
complete and re-runnable if anyone wants to finish it.

## Reproducing

Cached arrays are **not** distributed (~5 GB decoded), but they are exactly
reproducible from the public sources. Each `*_manifest.json` pins the source
shards, the subsampling seed (20240), per-split caps and class balances, and the
preparation scripts are deterministic given those.

```bash
pip install -r requirements.txt

python src/prepare_data.py        # Camelyon17-WILDS
python src/prepare_crc_data.py    # NCT-CRC-HE
python src/prepare_cxr_data.py    # chest radiographs (experiment 3)
```

**Verifying a rebuild.** Determinism here is load-bearing: new runs are only
comparable to the published ones if the rebuilt cache contains the *same samples*.
The Camelyon17 cache was rebuilt from scratch during this work and verified
sample-identical by re-deriving the selection indices independently from the raw
label columns under the recorded seed and confirming they reproduce the cached
label vectors exactly (split sizes, class balances and shard set also match the
manifest). If you rebuild, run the same check rather than assuming it — a
`prepare_*.py` that silently reads a different shard subset produces
plausible-looking splits that are not the published ones.

```bash
python src/verify_cache.py        # manifest ↔ cache agreement, sample-level
```

Then per experiment, e.g. for the replication:

```bash
python src/pretrain_ssl_crc.py --seed 0        # SSL arms only
python src/run_arm_crc.py --arm ssl_sam --seed 0
python src/aggregate_crc.py                    # tables + bootstrap CIs
python src/flatness_crc.py                     # sharpness measurements
python src/make_figures_crc.py
```

`run_all_crc.py` and `run_all_cxr.py` drive full sweeps. The `_crc` and `_cxr`
scripts are deliberate near-duplicates of the Camelyon17 originals rather than a
parameterised pipeline: the differences are dataset-forced (grayscale loader and
modality-appropriate augmentations for radiographs; two shift axes and
ID-validation selection for CRC), and keeping them separate means the completed
experiments stay reproducible exactly as run.

### Per-example predictions

Not distributed, and not recoverable — a compute-environment failure destroyed
them. This has one consequence that matters: the bootstrap confidence intervals
and paired contrasts in `results/*/bootstrap_*.csv` were computed from those
predictions before the loss and are released as computed values, but they
cannot be recomputed from the released files alone. Re-running the training
scripts regenerates predictions and therefore new intervals; these will differ
in the last digits from the published ones (different RNG draws), which is
expected and not a discrepancy.

## Compute

Everything here runs on CPU; no GPU is required or used. All timings are
wall-clock from the run records (`runs_camelyon17/*.json`, `wall_s`) on an
8-thread CPU:

| stage | cost |
|---|---|
| `prepare_data.py` (Camelyon17, decode + splits) | ~25 min, ~5 GB cache |
| SimCLR pre-training, 6 epochs | ~55 min/seed (~550 s/epoch) |
| one training arm, 5 epochs | 4.4–18.4 h (see below) |
| the six Camelyon17 arm-runs released here | 91 h total |
| `aggregate_*.py`, `flatness.py`, `make_figures*.py` | < 5 min |

Set `--threads` to match your machine; the released timings are 8-thread. Per-arm
cost varies more than the epoch count suggests. SAM arms average 18.4 h against
13.6 h for non-SAM (1.35×, since each SAM step needs two forward/backward
passes). But the dominant term is machine contention, not arm: `ssl_seed0` and
`ssl_seed1` ran the identical configuration for the identical 5 epochs and took
16.7 h and 4.4 h respectively, a 3.8× spread from competing load alone. Treat the
released `wall_s` values as one machine's history rather than a benchmark, and
budget from your own first epoch.

## Citation

If you use this code or the released result records, please cite the paper. A
`CITATION.cff` is included and GitHub renders it as a "Cite this repository"
button; the manuscript is the citable reference for the findings.

## Building the manuscript

Preferred route — `tectonic` fetches whatever TeX files it needs on demand, so
there is no tree to install or lose:

```bash
cd paper
TECTONIC_CACHE_DIR=../.tectonic_cache tectonic -X compile main_journal.tex --keep-intermediates
python check_bib.py          # audits citations; does NOT write the .bbl
```

Alternative — `./build.sh` netinstalls a real TeX Live tree into the workspace
and compiles with `pdflatex`. It is idempotent, but the tree lands outside the
repository and is destroyed by any workspace sweep, after which it must re-download.
See the comment at its head for why a conda `texlive-core` install is not a working
substitute: it ships TeX *binaries* with no macro tree, and its `mktexfmt`/`tlmgr`
are broken by a missing Perl module, so it cannot even rebuild a format file.

**Check the reference list before circulating a draft.** A TeX installation
lacking `unsrtnat.bst` does not fail the build — `bibtex` writes a zero-byte
`.bbl` and LaTeX produces a complete-looking PDF with *no bibliography at all*
and no error. A correct build ends with references [1]–[35]. `check_bib.py`
catches the adjacent silent failures: a cited key absent from the `.bib`,
duplicate keys for one work (this file accumulated several across revisions), and
entries with neither author nor title. It exits non-zero on an unresolved key, so
it can gate a release. It deliberately does not generate the bibliography — a
hand-built one would drift from the compiled version.

### One substituted file

`src/figure_style.py` is a standalone reimplementation. The published figures
were produced with an equivalent helper that lived in the authoring environment
and is not distributable; this version exposes the same two functions with the
same signatures and the same settings. It sets matplotlib rcParams and draws
panel letters — it touches no computed value.

## Provenance note

Part of this repository was reconstructed after a workspace sweep destroyed the
original working tree. `paper/CHANGES.md` records what was lost, what was
recovered and how, and which numbers are transcript-recovered rather than
recomputed. Every recovered script was verified to parse; every recovered
number carries its provenance in the data file that holds it.

## Licence

Code: MIT (`LICENSE`). Camelyon17-WILDS is CC0; NCT-CRC-HE-100K and
CRC-VAL-HE-7K are CC-BY-4.0 — see the manuscript's data availability statement
for citations.
