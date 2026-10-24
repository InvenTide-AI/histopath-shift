# Run records

One JSON record per training run, xz-compressed. `scripts/unpack_records.sh` (or `make records`) verifies
`SHA256SUMS` and unpacks them under `$HISTOPATH_DATA/v2`.

| Archive | Records | Used for |
|---|---|---|
| `v2_final_runs.tar.xz` | 3,400 | every result of the article and the Supplementary Information (`runs/{C,S}/<cohort>/fold<t>/<arm>__<hp>__s<seed>.json`) |
| `v2_tuning_runs.tar.xz` | 1,378 | hyperparameter selection (`results/selected_hparams.json`), Supplementary Note S12 |
| `v2_quarantine_bt_lr.tar.xz` | 120 | Barlow Twins fine-tunes with the inherited ImageNet learning rate (Supplementary Note S12) |
| `v2_quarantine_macenko_prefix.tar.xz` | 230 | per-tile Macenko / TIA-style records before the fix (Supplementary Note S13) |
| `v2_diagnostics_macenko.tar.xz` | 131 | Macenko diagnostics: dependence on the tile set (Supplementary Note S13) |
| `v2_test_predictions_C.tar`, `v2_test_predictions_S.tar` | 1,840 + 1,560 | test labels and ID-selected test probabilities (float32) of every final run, for the site acceptance test (`make sat`) |
| `v1_fivehospital_case_study.tar.xz` | 60 | v1 records behind Supplementary Note S15 (see below) |
| `v1_case_studies_extra.tar.xz` | 6 | v1 records behind Supplementary Note S15 (see below) |

`SNAPSHOT.json` is the manifest of the frozen snapshot (id `1381bdd759f9`) from which all analysis outputs were
computed; `MANIFEST.json` lists sizes and hashes.

A record contains the full hyperparameters, the SHA-256 of the training code, the fingerprint of the fold cache,
`status` (`ok` or `diverged`), eight metrics per split at each of the ten checkpoints, the per-domain
ID-validation AUROC, the checkpoint selected by each of the four rules, test-time-adaptation results and, for
Camelyon17, the PatchCamelyon score. Of the prediction files only the test labels and the probabilities at the
ID-selected checkpoint are included (same file names and keys); model weights (`*.pt`) are not included.
Provenance fields are kept as recorded, except that the cluster data root in path fields such as `cache_dir`
is replaced by `$HISTOPATH_DATA`. The record sizes therefore differ slightly from those listed in `SNAPSHOT.json`,
which is kept as frozen so that the snapshot id can still be recomputed from it.

## Earlier (v1) pipeline

Only the records behind the case studies of Supplementary Note S15 are kept; no other result of the article uses
them. `make records` unpacks both archives into `$HISTOPATH_DATA`.

| Path after unpacking | Used for |
|---|---|
| `runs/fold<t>/{baseline,sam,ssl,ssl_sam}_seed<s>.json` | the v1 five-hospital grid (ERM, SAM, SimCLR, SimCLR+SAM × 3 seeds); checked by `src/stats_tests/test_v1_regression.py` |
| `v1/runs/camelyon17_center2/*.json` | the withdrawn single-hospital SimCLR × SAM synergy claim (one test hospital, one or two seeds per arm) |
| `v1/results/camelyon17_center2/results_raw.csv` | the same runs as one table |
| `v1/results/nct_crc_he/crc_results_raw.csv` | the NCT-CRC-HE single-institution result (S15.2) |

The v1 model definition is kept as `src/tests/v1_train_lib.py`; `src/tests/test_v2.py` checks that the compact
CNN is unchanged. An audit of the v1 code found defects that the current pipeline (`src/`) fixes (Supplementary
Note S15). They do not touch the four arms of the case studies, except that v1 SAM updated batch-norm statistics
on the perturbed pass.
