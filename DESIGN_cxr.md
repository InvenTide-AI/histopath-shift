# Cross-institution chest radiography experiment — design registered before running

Written and frozen **before** any model was trained. The prediction below is recorded so
that the outcome is interpretable whichever way it lands.

## Why this experiment

The two completed experiments disagree, and the disagreement is structured:

| experiment | shift | SSL+SAM vs supervised |
|---|---|---|
| Camelyon17-WILDS | unseen hospital (centre 2) | **+0.019** AUROC |
| NCT-CRC-HE | stain appearance (photometric) | **−0.072** AUROC |
| NCT-CRC-HE | external patient cohort | **−0.009** AUROC |

The interaction term between the two mechanisms was positive in *both* experiments
(+0.161 and +0.078). What differed was whether it exceeded the cost of each mechanism
applied alone. This generates a mechanism-level hypothesis rather than a benchmark search:

> **H1.** The two-pillar combination helps when site nuisance is *substituted wholesale*
> (different scanner, protocol, population), and not when the shift is a low-level
> photometric corruption for which a preprocessing remedy already exists.
>
> **H2.** Pillar I requires an unlabelled pre-training corpus that *spans* the sites
> whose nuisance must be rendered unlearnable. A single-site pool cannot teach
> invariance to a nuisance it never varies.

H2 is the confound the CRC experiment could not separate: its unlabelled pool came from a
single release *and* was tumour-poor. This design separates the pool from the shift by
varying the pool while holding the shift fixed.

## Registered prediction

1. On cross-institution shift (NIH → CheXpert), the SSL+SAM arm pre-trained on the
   **multi-site** pool beats the supervised baseline in AUROC, with a non-overlapping
   95 % bootstrap CI.
2. The same arm pre-trained on the **single-site** pool does *not* beat baseline, or beats
   it by less. Formally: Δ(multi-site) > Δ(single-site), tested as a paired contrast.
3. The interaction term is positive on both pools (this reproduced twice already), but
   only exceeds the negative main effects on the multi-site pool.
4. Flatness will again fail to predict transfer rank-wise. We have no positive prediction
   here; two datasets already disagree on the sign.

**What falsifies the programme.** If the multi-site SSL+SAM arm does not beat baseline on
a genuine cross-institution shift with a site-spanning pool, then H1 and H2 are both
unsupported and the Camelyon17 result should be treated as a single-split finding that did
not replicate. That is the outcome we will report if we get it.

## Data

| role | cohort | notes |
|---|---|---|
| train + ID test | NIH ChestX-ray14 (institution A) | frontal; patient-disjoint splits |
| OOD test | CheXpert (institution B) | frontal only; different institution, scanners, population |
| unlabelled pool S | NIH only | single-site |
| unlabelled pool M | NIH + CheXpert | site-spanning, patients disjoint from OOD test |

- **Task:** pleural effusion, binary. The one finding labelled in both cohorts
  (NIH `Effusion`, CheXpert `Pleural Effusion`).
- **View filter:** frontal only. CheXpert is ~80 % frontal and NIH is frontal throughout;
  leaving laterals in would make view, not institution, the shift.
- **Label convention:** CheXpert blank → negative (the standard convention); explicit
  uncertain → **excluded**. Recorded because it changes prevalence materially.
- **Prevalence:** NIH ≈ 9 %, CheXpert ≈ 41 % under that convention. All **test** sets are
  **balanced 50/50 by subsampling** so AUROC is comparable across cohorts and prevalence
  is not a confound.
- **Training set is not balanced by subsampling.** Balancing it 50/50 at 9.5 % prevalence
  would discard ~80 % of available negatives and cap training at ~5k images. Instead the
  train split keeps **every** positive and fills to the target size with random negatives,
  and the loss is **class-weighted** by the negative/positive ratio — prevalence-independent
  without throwing data away. Identical for all six arms, so not a between-arm confound.
- **Patient disjointness is enforced everywhere.** Both cohorts have multiple images per
  patient (NIH `00000945_000.png` → patient `00000945`; CheXpert
  `train/patient00001/study1/...` → `patient00001`). Splitting by image would leak the same
  patient across train and test and inflate every arm. Splits are computed on patient IDs
  and images follow their patient.

## Declared caveat: pool M is transductive

Pool M contains unlabelled images from the target institution, so arms using it are doing
unsupervised domain adaptation, not pure domain generalisation. This is **not** hidden — it
is the point of the H2 contrast, and it is the clinically realistic case: a hospital
deploying a model usually does have unlabelled images of its own. Two constraints keep it
honest:

- pool M patients are disjoint from the OOD test patients (no image and no patient overlap);
- **no CheXpert label is read at any point** during pre-training, training, or model
  selection.

Arms on pool S remain a pure domain-generalisation comparison. The manuscript must report
the two pools as answering different questions and must not present a pool-M number as a
domain-generalisation result.

## Protocol frozen from the previous two experiments

Architecture, optimiser, SAM ρ, batch size, LR schedule, seeds and all analysis code are
carried over unchanged. Deviations, with reasons:

- **Input resolution 96×96** rather than 64×64. Effusion is a costophrenic-angle finding;
  at 64×64 the relevant anatomy is a few pixels. This is a task requirement, applied
  identically to every arm.
- **Model selection is in-distribution** (NIH held-out patients), as before. No arm sees
  CheXpert labels, and the OOD set is scored once per checkpoint.
- **Augmentations replaced** (`sup_augment_cxr`, `simclr_view_cxr` in `train_lib.py`).
  The histopathology augmentations encode dihedral symmetry — a tissue patch is valid under
  any flip or 90° rotation — and stain colour jitter. Neither holds for radiographs: they
  have a canonical upright orientation, and they are grayscale, so hue jitter would inject
  colour that does not exist in the data. The substitutes use horizontal flip only, small
  affine jitter (±8°, ±6 % shift, ±8 % scale), and brightness/contrast jitter; the SimCLR
  view adds mild blur and keeps the crop-scale floor at 0.5 because effusion evidence sits
  at the peripheral costophrenic angles and aggressive crops would remove it from both
  views. Intensity jitter and blur are the radiograph analogue of the stain shortcut the
  histopathology views targeted (exposure/scanner variation). **The originals are left
  untouched**, so the two completed experiments remain reproducible. Identical for all six
  arms, so this is not a between-arm confound.
- **Grayscale storage.** Radiographs are stored single-channel and replicated to three
  channels at load (`to_float_gray`), keeping the encoder unchanged.

## Arms (6) × seeds (2) = 12 runs, plus 4 pre-trainings

`baseline`, `sam`, `ssl_S`, `ssl_M`, `ssl_S+sam`, `ssl_M+sam`.
Baseline and `sam` need no pre-training; `ssl_S`/`ssl_M` share encoders with their `+sam`
counterparts, so pre-training is 2 pools × 2 seeds.

## Analysis (identical to the CRC experiment)

Paired bootstrap, B = 10,000, joint resamples of the test set shared across arms; percentile
CIs; two-sided p as twice the smaller tail. 2×2 interaction decomposition per pool.
Adversarial sharpness at ρ ∈ {0.05, 0.2}, 10 PGD-ascent steps, Spearman against OOD AUROC.
