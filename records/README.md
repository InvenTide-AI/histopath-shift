# Run records

One JSON record per training run, xz-compressed. `scripts/unpack_records.sh` (or `make records`) verifies
`SHA256SUMS` and unpacks them under `$HISTOPATH_DATA/v2`.

| Archive | Records | Used for |
|---|---|---|
| `v2_final_runs.tar.xz` | 3,400 | every main and appendix result (`runs/{C,S}/<cohort>/fold<t>/<arm>__<hp>__s<seed>.json`) |
| `v2_tuning_runs.tar.xz` | 1,378 | hyperparameter selection (`results/v2/selected_hparams.json`), tuning appendix |
| `v2_quarantine_bt_lr.tar.xz` | 120 | Barlow Twins fine-tunes with the inherited ImageNet learning rate (tuning appendix) |
| `v2_quarantine_macenko_prefix.tar.xz` | 230 | per-tile Macenko / TIA-style records before the fix (Macenko appendix) |
| `v2_diagnostics_macenko.tar.xz` | 131 | Macenko diagnostics: dependence on the tile set (Macenko appendix) |
| `v2_test_predictions_C.tar`, `v2_test_predictions_S.tar` | 1,840 + 1,560 | test labels and ID-selected test probabilities (float32) of every final run, for the site acceptance test |
| `../legacy_v1/records/v1_fivehospital_case_study.tar.xz` | 60 | v1 records behind the two case studies |

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
