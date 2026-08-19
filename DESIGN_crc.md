# Second-dataset replication: design specification (NCT-CRC-HE)

Frozen before any model was trained. Deviations from this document, if any, are
recorded at the bottom.

## Purpose

The Camelyon17-WILDS experiment found that the cross-hospital gain from the two
proposed mechanisms is an **interaction**: neither self-supervised pre-training
(Pillar I) nor sharpness-aware minimisation (Pillar II) helps alone, but the
combination does. A single dataset cannot distinguish a general property of the
two mechanisms from a property of Camelyon17. This replication asks whether the
interaction reproduces on an independent histopathology cohort with a different
organ, a different task, and a different kind of distribution shift.

## Dataset

NCT-CRC-HE (Kather, Halama & Marx 2018), Zenodo DOI 10.5281/zenodo.1214456,
CC-BY-4.0, via the HuggingFace mirror `1aurent/NCT-CRC-HE`.

| Split | Patches | Description |
|---|---|---|
| `NCT_CRC_HE_100K` | 100,000 | 86 slides, NCT Heidelberg + UMM Mannheim; Macenko colour-normalised |
| `NCT_CRC_HE_100K_NONORM` | 100,000 | same raw material, **no** colour normalisation |
| `CRC_VAL_HE_7K` | 7,180 | 50 patients, no patient overlap with the 100K set |

9 tissue classes: ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM.

## Task

Binary **TUM** (colorectal adenocarcinoma epithelium) vs **non-TUM**, class
balanced. The `BACK` (background) class is **excluded** from all splits: it is
empty slide glass and trivially separable, so including it in the negative set
would inflate every arm's score and compress the between-arm differences the
experiment is measuring.

Patches are centre-cropped from 224x224 to 64x64 to match the Camelyon17
preprocessing and the encoder's input size. This is a resolution reduction
relative to the source data (0.5 MPP) and is a stated limitation, not an
oversight: it is what makes 8 training runs feasible on 10 CPU cores.

## Two shift axes

Neither axis is a cross-hospital shift. This is stated explicitly because the
Camelyon17 experiment *was* cross-hospital and the distinction matters for what
can be claimed.

1. **Stain/acquisition shift (primary).** Train on colour-normalised patches,
   test on `NONORM` patches of the same tissue material. Patient identity is
   held approximately constant while staining intensity and colour vary. This is
   the real-data analogue of the paper's synthetic colour-tint appendix, and it
   is precisely the acquisition nuisance the position argument identifies.
2. **Patient shift (secondary).** Test on `CRC_VAL_HE_7K`: 50 patients with no
   overlap with the training set. Tissue came from the **same NCT tissue bank**,
   so this is a patient-disjoint external cohort, *not* a second hospital.

## Arms

Four arms, identical to the Camelyon17 ablation:

| Arm | Pillar I (SSL) | Pillar II (SAM) |
|---|---|---|
| `baseline` | - | - |
| `ssl` | yes | - |
| `sam` | - | yes |
| `ssl_sam` | yes | yes |

## Hyperparameters (held identical to the Camelyon17 run)

Supervised: 5 epochs, batch 128, SGD lr 0.02, momentum 0.9, weight decay 1e-4,
cosine annealing, SAM rho 0.05. SSL: SimCLR/NT-Xent, 6 epochs, batch 256,
lr 0.05, temperature 0.5. Encoder: 4-stage CNN, ~0.58M parameters.
Seeds: 0 and 1 (two per arm, 8 runs total).

Nothing is tuned on this dataset. Reusing the Camelyon17 hyperparameters
verbatim is the point: a replication that required re-tuning per dataset would
not test the generality of the mechanisms.

## Model selection — departure from the WILDS protocol

The Camelyon17 run selected checkpoints on an **OOD validation hospital**
(center 1), leaving the test hospital untouched until the end. NCT-CRC-HE has no
third site, so that protocol cannot be reproduced.

Checkpoints here are selected on **in-distribution validation** (held-out
colour-normalised patches). Both shift test sets are scored **once**, with the
already-selected checkpoint. No shifted data influences training or selection.

Consequence, stated plainly: ID-based selection is the weaker protocol, and
absolute numbers are not comparable to the Camelyon17 table. Between-arm
contrasts within this dataset remain valid because every arm gets the identical
selection rule.

## No-leakage constraints

- SSL pre-training uses **only** colour-normalised unlabeled tiles, disjoint
  from train and id_val. The encoder never sees NONORM appearance; letting it
  would leak the primary test condition.
- SSL tiles are drawn from the 100K cohort only, never from `CRC_VAL_HE_7K`.
- Labels are never used during pre-training.

## Pre-registered analysis

Primary estimand: the **interaction term** in the 2x2 decomposition of shifted
AUROC, defined exactly as for Camelyon17 —
`interaction = (ssl_sam - baseline) - (ssl - baseline) - (sam - baseline)`,
computed on seed-matched runs.

Uncertainty: paired bootstrap over test patches (10,000 resamples), identical
resampled indices scored for both arms of every contrast.

Reported per arm and per shift axis: AUROC, average precision, accuracy, F1,
balanced accuracy, sensitivity, specificity, ECE; plus the ID-to-shift gap and
adversarial sharpness at rho in {0.05, 0.2}.

Comparability rule carried over from the Camelyon17 analysis: an arm that fails
to reach the in-distribution ceiling is **not** credited with a small
generalisation gap, because a small gap under a weak ID fit does not indicate
better transfer.

## Outcomes that would count as a failure to replicate

Declared in advance so the result cannot be reinterpreted after the fact:

- Interaction term not positive, or CI comfortably spanning zero -> the
  Camelyon17 interaction does not generalise.
- `ssl_sam` not the best arm on the shifted axis -> the combined
  recommendation is not supported on this dataset.
- Either mechanism helping substantially *alone* -> the "neither alone" claim is
  Camelyon17-specific.

Any of these will be reported as such in the manuscript.

## Deviations from this specification

None recorded yet.
