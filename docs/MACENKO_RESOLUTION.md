# Macenko arms: resolution (2026-10-05)

This memo replaces the "decide whether stain.py needs a further change" caveat at the top of
`revision/rerun_macenko.txt`. Code: `src/v2/stain.py`, `src/v2/run_v2.py` (`split_groups`, `c17_slides`,
`macenko_splits`), `src/v2/arms.py` (macenko-family hparams), `src/v2/tests/test_v2.py` (35/35 pass).
Diagnostic records: `$V2/runs_diag/macgroup/` (SLURM jobs 9287973 and 9287974). Nothing under `runs/` or
`runs_tune/` was written. No production rerun has been submitted.

## 1. The bug: negative stain concentrations
`stain.concentrations()` used an unconstrained least-squares fit, `pinv(HE) @ OD`, as Macenko's MATLAB `HE\Y`
and torchstain do. In a 64-px tile the per-tile H and E vectors are nearly collinear (cos 0.95-0.98). Pixels
rich in eosin then get large negative H, and after the maxC rescale and the reconstruction with `HE_ref` they
come out saturated pink. On Camelyon17 center 4 (c17 fold 4 test) this artefact is correlated with the label:
it hits 90% of tumour tiles but 54% of normal tiles (2% vs 20% in training). The tier-C macenko off-site AUROC
(test@ID-selected checkpoint) was 0.455/0.389/0.403/0.457/0.399 (mean 0.421), against ERM at 0.59-0.80.

## 2. The clamp (`mac_nonneg=1`)
Concentrations are now clamped to C >= 0, since stain amounts are physically non-negative (StainTools uses a
positive lasso). With the same weights, clamping only at test time gave 0.95. With the models retrained on the
clamped transform (runs_diag/macnonneg, jobs 9287843/9287862), AUROC was 0.539/0.538/0.424/0.649/0.682
(mean 0.566). That is better, but still below ERM. Setting `mac_nonneg=0` reproduces the pre-fix records
(s0 0.464 vs 0.455).

## 3. Why per-tile Macenko is still weak
The rest of the gap comes from estimating the stain matrix on each 64-px tile. Center-4 tiles are pale: 27-34%
of their pixels are tissue, against 58-83% elsewhere. A per-tile eigen-plane and angle-percentile estimate from
about 1,100-1,400 pixels of one tissue type is unreliable. Taking the retrained weights and reconstructing
center 4 with the reference `HE_ref` instead raised the score from 0.553 to 0.934. In this sweep the per-tile
arm (`macenko_tile`) scores 0.473/0.521/0.398/0.468/0.415 (mean 0.455) on c17 fold 4. The inputs are
bit-identical to the clamped per-tile macenko, so the gap to the 0.566 above is GPU run-to-run
nondeterminism. For example, s1 selects step 960 in both runs and scores 0.538 vs 0.521. The ID-val selection
on this fold is very unstable.

## 4. The fix: group-level ("per-slide") Macenko, as in standard practice
Standard practice estimates the stain matrix once per slide or WSI. Final configuration of `macenko` and
`tia_style` (new keys exist only in the macenko family; `macenko_tile` keeps `mac_level=tile`,
`mac_ref=train`):

| hparam | value | meaning |
|---|---|---|
| `mac_level` | `group` | one (HE, maxC) per group, estimated from the pooled tissue pixels of up to 256 tiles (content-hash subsample, at most 200k px, ordering does not depend on row order) |
| `mac_group_unit` | `slide_domain` | group = slide crossed with the split's domain id (see below) |
| `mac_group_maxc` | `pooled` | maxC = 99th percentile of C over all pixels of the sampled tiles (the torchstain / StainTools / TIAToolbox statistic) |
| `mac_ref` | `train_group` | the target is fitted with the same group estimator on every training group: HE_ref = median of the group stain vectors, maxC_ref = median of the group maxC |
| `mac_nonneg` | 1 | C >= 0 (section 2) |

The first group-level version (runs_diag/maclevel: c17 fold 4 mean 0.768) was reviewed. Four verified
findings were applied:
1. **Canine groups mixed scanners.** A canine group id is the slide number, and the same slide imaged on
   three scanners shares it. In tr/iv, all groups contained three scanners, so the "per-slide" estimate pooled
   three scanner images and left the scanner differences in place. On canine fold 0 tr, the per-scanner range
   of mean RGB was: raw 25.5/17.3/20.6; slide-pooled 17.6/22.6/16.7 (G worse than raw); per-tile
   10.7/10.5/9.1. **Fix:** group by (group id, domain id) wherever the split has domain ids. Now: 7.2/5.9/5.4.
   ov/ot are single-scanner and unchanged. c17 and MIDOG groups already lie within one domain.
2. **c17 groups were patients, not slides.** 6 of the 43 patients have 2-3 slides. **Fix:** the c17 group is
   now the WILDS `metadata.csv` `slide` at the `rows.npz` row ids. Only the row id, patient and slide columns
   are read, and the patient must match the cached `g*.npy` or the code falls back to it. Fold 4: tr has
   24 slides (21 patients), ot has 10 slides (9 patients).
3. **Source and target used different estimators (high).** `HE_ref`/`maxC_ref` were the median of per-tile
   estimates, while the source was one pooled estimate. On c17 fold 4 training slides, the pooled H scale was
   0.69-0.93 against 1.6-1.9 per tile, so every slide got an H rescale of about 2x. This caused the darkening.
   **Fix:** `mac_ref=train_group`. Source and target now share one estimator, as in the reference
   implementations whose `fit()` and `transform()` call the same routine. The fold-4 median training E vector
   is HE_ref E = (0.316, 0.866, 0.387). Some single slides (centers 0/1) give a pooled E with red OD 0.43-0.50.
   Since source and target now use the same estimator, this no longer biases the mapping.
4. **The maxC statistic was non-standard (medium).** It is now the pooled 99th percentile (`mac_group_maxc`).
   The earlier statistic is still available as `tile_median`.

Reference fit: every training group was used on c17, MIDOG and canine folds 0 and 4. On canine folds 1-3, 1-2
slide x scanner groups fall back (too little tissue), so 64-65 of 66 groups are used. Across all diagnostic
records, 5 of 4,705 split-groups fall back (all canine tr/iv) and none in any test split. PCam is grouped by
its `wsi` column (129 WSIs, 0 fallbacks).

**Sensitivity to label-selected tile makeup (reviewer finding, high).** The c17 cache is class-balanced, so
the fraction of tumour tiles per test slide ranges from 0.00 to 0.93. Slide 48 holds 10,444 of the 25,000
test tiles and is 93% tumour. The test is to fix the trained weights and re-estimate each test slide's (HE,
maxC) from a subset of its own tiles. Labels are used only to form the diagnostic subsets; the pipeline uses
none.

| tiles used for each test slide's estimate | first group version (s0, reviewer) | final config, macenko s0-s4 (mean) | final config, tia_style s0-s4 (mean) |
|---|---|---|---|
| all (= pipeline) | 0.784 | 0.923/0.917/0.968/0.969/0.949 (0.945) | 0.981 |
| negative tiles only | 0.972 | 0.946/0.914/0.967/0.941/0.969 (0.948) | 0.972 |
| positive tiles only | 0.720 | 0.902/0.905/0.965/0.968/0.929 (0.934) | 0.978 |
| none (HE_ref/maxC_ref only) | 0.782 | 0.559/0.418/0.583/0.421/0.817 (0.560) | 0.897 |

With the consistent estimator, the spread between makeups drops from 0.25 to 0.014 AUROC. The method is still
transductive: each test slide's estimate uses that slide's own unlabelled tiles, and the tile set is the
benchmark's label-selected one. A label-independent WSI tissue sample would be needed to match deployment,
and it is not available: the WILDS patches are themselves taken from annotated regions, and there are no WSIs.
The paper should mark macenko and tia_style as transductive (like the bnadapt/tent arms), state this
limitation, and report the sensitivity above. The "none" row shows that the trained models depend on the
per-slide normalisation, as intended.

## 5. Results (tier C, off-site AUROC at the ID-val-selected checkpoint)
"old" = production records in `runs/C`, written with the pre-fix stain.py (per-tile, unclamped). "diag" =
`runs_diag/macgroup` (final config; `macenko_tile` = clamped per-tile). ERM and he_jitter are the `runs/C`
5-seed means. Folds other than c17 fold 4 have seed 0 only for the macenko family.

c17 fold 4 (test = center 4), 5 seeds:

| arm | s0 | s1 | s2 | s3 | s4 | mean (sd) |
|---|---|---|---|---|---|---|
| ERM (runs/C) | 0.797 | 0.602 | 0.726 | 0.772 | 0.592 | 0.698 (0.096) |
| he_jitter (runs/C) | 0.945 | 0.766 | 0.944 | 0.967 | 0.960 | 0.917 (0.085) |
| macenko old (pre-fix) | 0.455 | 0.389 | 0.403 | 0.457 | 0.399 | 0.421 (0.033) |
| macenko clamp-only (macnonneg) | 0.539 | 0.538 | 0.424 | 0.649 | 0.682 | 0.566 |
| macenko first group version (maclevel) | 0.771 | 0.837 | 0.709 | 0.714 | 0.810 | 0.768 |
| **macenko diag (final)** | 0.923 | 0.917 | 0.968 | 0.969 | 0.950 | **0.946 (0.025)** |
| macenko_tile diag (clamped per-tile) | 0.473 | 0.521 | 0.398 | 0.468 | 0.415 | 0.455 (0.049) |
| tia_style old (pre-fix) | 0.936 | 0.830 | 0.885 | 0.830 | 0.851 | 0.866 (0.045) |
| tia_style first group version | 0.954 | 0.961 | 0.950 | 0.962 | 0.922 | 0.950 |
| **tia_style diag (final)** | 0.975 | 0.982 | 0.982 | 0.983 | 0.983 | **0.981 (0.003)** |

All folds, seed 0 for the macenko family:

| cohort | fold | ERM (5s) | he_jitter (5s) | macenko old | macenko diag | macenko_tile diag | tia_style old | tia_style diag |
|---|---|---|---|---|---|---|---|---|
| c17 | 0 | 0.937 | 0.968 | 0.969 | 0.948 | 0.968 | 0.968 | 0.961 |
| c17 | 1 | 0.820 | 0.911 | 0.904 | 0.907 | 0.875 | 0.906 | 0.896 |
| c17 | 2 | 0.803 | 0.904 | 0.967 | 0.966 | 0.965 | 0.955 | 0.967 |
| c17 | 3 | 0.962 | 0.963 | 0.967 | 0.938 | 0.955 | 0.962 | 0.964 |
| c17 | 4 | 0.698 | 0.917 | 0.455 | 0.923 | 0.473 | 0.936 | 0.975 |
| canine | 0 | 0.807 | 0.819 | 0.869 | 0.876 | 0.848 | 0.804 | 0.805 |
| canine | 1 | 0.753 | 0.674 | 0.700 | 0.820 | 0.704 | 0.792 | 0.770 |
| canine | 2 | 0.551 | 0.853 | 0.842 | 0.839 | 0.821 | 0.844 | 0.804 |
| canine | 3 | 0.789 | 0.833 | 0.841 | 0.837 | 0.809 | 0.837 | 0.764 |
| canine | 4 | 0.669 | 0.795 | 0.684 | 0.714 | 0.662 | 0.789 | 0.763 |
| midog21sn | 0 | 0.841 | 0.885 | 0.877 | 0.888 | 0.882 | 0.885 | 0.899 |
| midog21sn | 1 | 0.797 | 0.879 | 0.850 | 0.878 | 0.855 | 0.875 | 0.878 |
| midog21sn | 2 | 0.765 | 0.864 | 0.846 | 0.874 | 0.851 | 0.861 | 0.867 |
| midogpp | 0 | 0.855 | 0.854 | 0.863 | 0.892 | 0.864 | 0.857 | 0.858 |
| midogpp | 1 | 0.815 | 0.809 | 0.815 | 0.836 | 0.822 | 0.803 | 0.811 |
| midogpp | 2 | 0.830 | 0.831 | 0.847 | 0.854 | 0.842 | 0.841 | 0.855 |
| midogpp | 3 | 0.871 | 0.888 | 0.889 | 0.922 | 0.899 | 0.901 | 0.917 |
| midogpp | 4 | 0.823 | 0.862 | 0.844 | 0.852 | 0.839 | 0.847 | 0.867 |
| midogpp | 5 | 0.827 | 0.848 | 0.825 | 0.871 | 0.840 | 0.853 | 0.867 |
| midogpp | 6 | 0.766 | 0.811 | 0.839 | 0.824 | 0.841 | 0.836 | 0.834 |
| **c17 mean** | 5 folds | **0.844** | **0.933** | **0.852** | **0.937** | **0.847** | **0.946** | **0.953** |
| **canine mean** | 5 folds | **0.714** | **0.795** | **0.787** | **0.817** | **0.769** | **0.813** | **0.781** |
| **midog21sn mean** | 3 folds | **0.801** | **0.876** | **0.858** | **0.880** | **0.862** | **0.874** | **0.881** |
| **midogpp mean** | 7 folds | **0.827** | **0.843** | **0.846** | **0.864** | **0.850** | **0.848** | **0.858** |
| **all 20 folds** | | **0.799** | **0.858** | **0.835** | **0.873** | **0.831** | **0.868** | **0.866** |

PCam (c17 folds 0-4, seed 0, WSI-grouped): macenko diag 0.751/0.793/0.689/0.775/0.837; old
0.768/0.764/0.779/0.767/0.788; ERM 0.693/0.730/0.827/0.761/0.692; tia_style diag 0.831/0.824/0.752/0.692/0.865.

How to read these results:
- **The center-4 failure is resolved.** macenko goes from 0.421 to 0.946 (5 seeds), above he_jitter
  (0.917), and tia_style from 0.866 to 0.981. The per-tile variant stays at about 0.46 and is kept as
  `macenko_tile` for the appendix.
- **macenko improves outside c17 fold 4 as well (seed 0, so read with care).** Across the 20 folds, the diag
  run beats the old one on 14, and the mean is 0.873 vs 0.835. Canine gains the most (0.817 vs 0.787) from
  the slide x scanner grouping. It is slightly lower on c17 folds 0 and 3 (-0.021 and -0.029, one seed).
- **tia_style is lower on canine:** 0.781 vs 0.813, lower on 4 of 5 folds. This is one seed per fold against
  old seed 0. The 5-seed tier-C rerun will show whether it is real; do not interpret it before then.
- Single-seed differences below about 0.02-0.03 are within seed noise; tier-C seed sd is 0.01-0.1, depending
  on the fold.

## 6. Isolation from the running tier-S sweep (array 9287868)
- Files were written atomically (temp file in the same directory, then `os.replace`), in the order stain.py,
  run_v2.py, arms.py, tests. Every intermediate mix of old and new modules is compatible.
- **Hyperparameters:** the new keys (`mac_group_maxc`, `mac_group_unit`) and the new `mac_ref` value
  `train_group` exist only for USES_MACENKO. `test_nonmacenko_hp_unchanged` checks the default_hp and
  resolve_arm of every non-macenko arm against the snapshot.
- **Resume decisions:** I ran `_stale_reason` on all 4,041 jobs in `slurm/v2/plans/*`, with the old and the
  new code. No non-macenko job changed: 3,030 reuse and 781 missing, as the sweep progressed. The 230 jobs
  that differ are exactly the tier-C macenko/tia_style records, flagged "hyperparameters differ: mac_group_*".
  No S plan contains a macenko-family job.
- **Tests:** `python src/v2/tests/test_v2.py` passes 35/35. The new test is
  `test_macenko_group_reference_and_units`; the c17 center-4 test, the no-labels test and the hp tests were
  extended.

## 7. Final-phase jobs to run (NOT submitted)
Plans: `slurm/v2/plans/final_mac/{C,S}_<cohort>_fold<t>.json`, with shard lists `shards_C.txt` (23 shards) and
`shards_S.txt` (20 shards). They were made with `make_plan.final_jobs(arms=[macenko, macenko_tile,
tia_style])` from `results/v2/selected_hparams.json`: tier C uses default hp (tag `default`); tier S uses the
fold's selected ERM base LR (tag `selected`, as for every other tier-S arm); PCam is on for c17. The
record-by-record list (525 jobs: tier C 23 folds x 3 arms x 5 seeds = 345; tier S 20 folds x 3 arms x 3
seeds = 180) is in `slurm/v2/plans/final_mac/jobs.txt`.

Before submitting:
1. Quarantine the 230 stale tier-C records listed in section A of `revision/rerun_macenko.txt`
   (macenko/tia_style, `__default__s0-4`, 23 folds; 3 files each). Otherwise `run_v2` re-runs them in place
   under the same tags. The `mv` loop is in that file's header. There are no macenko_tile records and no
   tier-S macenko-family records.
2. Submit only after array 9287868 has finished, or accept that the two will share the GPU partition:
   `slurm/v2/submit.sh slurm/v2/plans/final_mac/shards_C.txt --time 01:00:00` and
   `slurm/v2/submit.sh slurm/v2/plans/final_mac/shards_S.txt --time 02:00:00`. Tier-C macenko jobs take about
   15 s each (about 4 min per c17 shard); tier S about 30-60 s per job.
3. When analysing, mark macenko and tia_style as transductive (section 4).

Jobs per shard (record tags; all under `$V2/runs/<tier>/<cohort>/fold<t>/`):
```
C c17       fold0  macenko__default__s{0,1,2,3,4}  macenko_tile__default__s{0,1,2,3,4}  tia_style__default__s{0,1,2,3,4}  hp={}  pcam=True
C c17       fold1  macenko__default__s{0,1,2,3,4}  macenko_tile__default__s{0,1,2,3,4}  tia_style__default__s{0,1,2,3,4}  hp={}  pcam=True
C c17       fold2  macenko__default__s{0,1,2,3,4}  macenko_tile__default__s{0,1,2,3,4}  tia_style__default__s{0,1,2,3,4}  hp={}  pcam=True
C c17       fold3  macenko__default__s{0,1,2,3,4}  macenko_tile__default__s{0,1,2,3,4}  tia_style__default__s{0,1,2,3,4}  hp={}  pcam=True
C c17       fold4  macenko__default__s{0,1,2,3,4}  macenko_tile__default__s{0,1,2,3,4}  tia_style__default__s{0,1,2,3,4}  hp={}  pcam=True
C canine    fold0  macenko__default__s{0,1,2,3,4}  macenko_tile__default__s{0,1,2,3,4}  tia_style__default__s{0,1,2,3,4}  hp={}  pcam=False
C canine    fold1  macenko__default__s{0,1,2,3,4}  macenko_tile__default__s{0,1,2,3,4}  tia_style__default__s{0,1,2,3,4}  hp={}  pcam=False
C canine    fold2  macenko__default__s{0,1,2,3,4}  macenko_tile__default__s{0,1,2,3,4}  tia_style__default__s{0,1,2,3,4}  hp={}  pcam=False
C canine    fold3  macenko__default__s{0,1,2,3,4}  macenko_tile__default__s{0,1,2,3,4}  tia_style__default__s{0,1,2,3,4}  hp={}  pcam=False
C canine    fold4  macenko__default__s{0,1,2,3,4}  macenko_tile__default__s{0,1,2,3,4}  tia_style__default__s{0,1,2,3,4}  hp={}  pcam=False
C midog21sn fold0  macenko__default__s{0,1,2,3,4}  macenko_tile__default__s{0,1,2,3,4}  tia_style__default__s{0,1,2,3,4}  hp={}  pcam=False
C midog21sn fold1  macenko__default__s{0,1,2,3,4}  macenko_tile__default__s{0,1,2,3,4}  tia_style__default__s{0,1,2,3,4}  hp={}  pcam=False
C midog21sn fold2  macenko__default__s{0,1,2,3,4}  macenko_tile__default__s{0,1,2,3,4}  tia_style__default__s{0,1,2,3,4}  hp={}  pcam=False
C midogpp   fold0  macenko__default__s{0,1,2,3,4}  macenko_tile__default__s{0,1,2,3,4}  tia_style__default__s{0,1,2,3,4}  hp={}  pcam=False
C midogpp   fold1  macenko__default__s{0,1,2,3,4}  macenko_tile__default__s{0,1,2,3,4}  tia_style__default__s{0,1,2,3,4}  hp={}  pcam=False
C midogpp   fold2  macenko__default__s{0,1,2,3,4}  macenko_tile__default__s{0,1,2,3,4}  tia_style__default__s{0,1,2,3,4}  hp={}  pcam=False
C midogpp   fold3  macenko__default__s{0,1,2,3,4}  macenko_tile__default__s{0,1,2,3,4}  tia_style__default__s{0,1,2,3,4}  hp={}  pcam=False
C midogpp   fold4  macenko__default__s{0,1,2,3,4}  macenko_tile__default__s{0,1,2,3,4}  tia_style__default__s{0,1,2,3,4}  hp={}  pcam=False
C midogpp   fold5  macenko__default__s{0,1,2,3,4}  macenko_tile__default__s{0,1,2,3,4}  tia_style__default__s{0,1,2,3,4}  hp={}  pcam=False
C midogpp   fold6  macenko__default__s{0,1,2,3,4}  macenko_tile__default__s{0,1,2,3,4}  tia_style__default__s{0,1,2,3,4}  hp={}  pcam=False
C midog21   fold0  macenko__default__s{0,1,2,3,4}  macenko_tile__default__s{0,1,2,3,4}  tia_style__default__s{0,1,2,3,4}  hp={}  pcam=False
C midog21   fold1  macenko__default__s{0,1,2,3,4}  macenko_tile__default__s{0,1,2,3,4}  tia_style__default__s{0,1,2,3,4}  hp={}  pcam=False
C midog21   fold2  macenko__default__s{0,1,2,3,4}  macenko_tile__default__s{0,1,2,3,4}  tia_style__default__s{0,1,2,3,4}  hp={}  pcam=False
S c17       fold0  macenko__selected__s{0,1,2}  macenko_tile__selected__s{0,1,2}  tia_style__selected__s{0,1,2}  hp={"lr": 0.03}  pcam=True
S c17       fold1  macenko__selected__s{0,1,2}  macenko_tile__selected__s{0,1,2}  tia_style__selected__s{0,1,2}  hp={"lr": 0.03}  pcam=True
S c17       fold2  macenko__selected__s{0,1,2}  macenko_tile__selected__s{0,1,2}  tia_style__selected__s{0,1,2}  hp={"lr": 0.03}  pcam=True
S c17       fold3  macenko__selected__s{0,1,2}  macenko_tile__selected__s{0,1,2}  tia_style__selected__s{0,1,2}  hp={"lr": 0.1}  pcam=True
S c17       fold4  macenko__selected__s{0,1,2}  macenko_tile__selected__s{0,1,2}  tia_style__selected__s{0,1,2}  hp={"lr": 0.03}  pcam=True
S canine    fold0  macenko__selected__s{0,1,2}  macenko_tile__selected__s{0,1,2}  tia_style__selected__s{0,1,2}  hp={"lr": 0.01}  pcam=False
S canine    fold1  macenko__selected__s{0,1,2}  macenko_tile__selected__s{0,1,2}  tia_style__selected__s{0,1,2}  hp={"lr": 0.1}  pcam=False
S canine    fold2  macenko__selected__s{0,1,2}  macenko_tile__selected__s{0,1,2}  tia_style__selected__s{0,1,2}  hp={"lr": 0.03}  pcam=False
S canine    fold3  macenko__selected__s{0,1,2}  macenko_tile__selected__s{0,1,2}  tia_style__selected__s{0,1,2}  hp={"lr": 0.1}  pcam=False
S canine    fold4  macenko__selected__s{0,1,2}  macenko_tile__selected__s{0,1,2}  tia_style__selected__s{0,1,2}  hp={"lr": 0.03}  pcam=False
S midog21sn fold0  macenko__selected__s{0,1,2}  macenko_tile__selected__s{0,1,2}  tia_style__selected__s{0,1,2}  hp={"lr": 0.03}  pcam=False
S midog21sn fold1  macenko__selected__s{0,1,2}  macenko_tile__selected__s{0,1,2}  tia_style__selected__s{0,1,2}  hp={"lr": 0.01}  pcam=False
S midog21sn fold2  macenko__selected__s{0,1,2}  macenko_tile__selected__s{0,1,2}  tia_style__selected__s{0,1,2}  hp={"lr": 0.01}  pcam=False
S midogpp   fold0  macenko__selected__s{0,1,2}  macenko_tile__selected__s{0,1,2}  tia_style__selected__s{0,1,2}  hp={"lr": 0.03}  pcam=False
S midogpp   fold1  macenko__selected__s{0,1,2}  macenko_tile__selected__s{0,1,2}  tia_style__selected__s{0,1,2}  hp={"lr": 0.03}  pcam=False
S midogpp   fold2  macenko__selected__s{0,1,2}  macenko_tile__selected__s{0,1,2}  tia_style__selected__s{0,1,2}  hp={"lr": 0.1}  pcam=False
S midogpp   fold3  macenko__selected__s{0,1,2}  macenko_tile__selected__s{0,1,2}  tia_style__selected__s{0,1,2}  hp={"lr": 0.03}  pcam=False
S midogpp   fold4  macenko__selected__s{0,1,2}  macenko_tile__selected__s{0,1,2}  tia_style__selected__s{0,1,2}  hp={"lr": 0.1}  pcam=False
S midogpp   fold5  macenko__selected__s{0,1,2}  macenko_tile__selected__s{0,1,2}  tia_style__selected__s{0,1,2}  hp={"lr": 0.03}  pcam=False
S midogpp   fold6  macenko__selected__s{0,1,2}  macenko_tile__selected__s{0,1,2}  tia_style__selected__s{0,1,2}  hp={"lr": 0.03}  pcam=False
```
