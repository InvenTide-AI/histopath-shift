# Current state: multi-hospital sweep, blocked on GPU

Everything below the next heading concerns the earlier two-arm seed gap and is
retained for provenance. The active work is the five-hospital sweep.

## Ready to launch

- **Splits built and verified.** Five leave-one-hospital-out folds, each holding
  out one centre entirely; 40k train / 8k in-dist. val / 20k held-out test per
  fold, 50/50 class balance. `python src/verify_cache.py` checks all five folds
  for site purity, patient disjointness and index disjointness — 20 fold checks,
  all pass. Patient/node/slide identifiers come from a mirror retaining them
  (the image-only distributions drop them); per-split row counts match the
  published release exactly (302,436 / 68,464 / 85,054).
- **Pixel cache complete.** 251,496 patches at native 96x96 (`src/cache/decoded_96.npy`,
  7.0 GB, ~25 min to rebuild via `decode_patches.py`). Coverage is exact — every
  row the five folds index is present, nothing extra. 42 patches sampled across
  all 21 source shards re-fetched and compared byte-for-byte: all identical.
  Zero all-zero rows (an interrupted decode leaves patches that train as black
  images without erroring). The array is gitignored and not archived; the
  row-index manifest `decoded_96_index.json` IS archived, since that records
  which rows the array holds.
- **All arms smoke-run on real pixels**, each with a distinct loss confirming the
  mechanism actually alters the training step: baseline 0.544, sam 0.992,
  stainaug 0.592, swa 0.544 (matches baseline as expected — SWA averages weights
  without touching the objective), dann 1.027, simclr pre-training NT-Xent 6.184.

## What is blocked

No compute target is configured, so the sweep has not run and **no model-based
number exists for the multi-hospital design**. The full design is ~1,280 CPU-hours
locally versus roughly 1-2 days on one A100. Add an SSH target to a GPU host
(Customize -> Compute), then:

```bash
cd src && python run_sweep.py        # see --help for fold/arm subsetting
```

Consequently the manuscript's title and abstract still frame the two-mechanism
study. They were left deliberately untouched: retargeting them commits to a claim
about mechanism ranking that the sweep has not produced and which may come back
negative.

## What is already final

Two findings are model-free measurements and do not depend on the sweep. Both are
written into the manuscript with figures:

- Site shift is **graded**: stain distance varies 1.6-fold across folds (0.342 to
  0.563), while domain-classifier AUROC is near ceiling everywhere (0.926-0.996)
  and rank-correlates with distance at only rho = 0.30. Distance is therefore the
  usable difficulty axis; separability is close to uninformative here.
- The **patient-nesting control** qualifies that: cross-centre patient pairs
  separate at median AUROC 1.000 versus 0.933 *within* a single centre. The site
  effect exceeds patient variation, but 0.933 means much of what a colour-based
  classifier detects at this resolution is patient-level, not institutional.
- Class balance is exactly 50/50 at every centre, so no shift difference is
  confounded with label prevalence. (An earlier per-site tumour-fraction range of
  0 to 0.755 was an artifact of class-ordered parquet row groups, not a property
  of the sites.)

## Build note

Use `tectonic`, not `pdflatex` — see README "Building the manuscript". A TeX Live
install missing `unsrtnat.bst` does not fail the build; it writes a zero-byte
`.bbl` and produces a complete-looking PDF with **no reference list at all**.

---

# Outstanding run: Camelyon17 seed 1, `sam` and `ssl_sam`

## Why these two arms

The Camelyon17 experiment released here has both seeds for `baseline` and `ssl`
but only seed 0 for `sam` and `ssl_sam`. The paper's central quantity — the
SSL x SAM interaction — therefore rests on a one-seed estimate *on that dataset*
(NCT-CRC-HE has all four arms at both seeds and shows the same sign at each seed
independently). Completing these two arms converts the Camelyon17 interaction
from one seed to two. It is the single most useful experiment left, and the
limitations section of the paper says so.

**No claim in the paper depends on the outcome.** The argument rests on the
five-of-five sign consistency across both datasets, not on the Camelyon17
magnitude. If these runs come back and the interaction is smaller, the honest
revision is to the magnitude and to the `I > c` margin at the held-out hospital
(currently `I=0.161` vs `c=0.142` — a thin margin, and worth restating plainly if
the second seed narrows it).

## State as of the end of the last session

- Cache rebuilt and **verified sample-identical** to the published runs:
  `python src/verify_cache.py` -> 30 checks, all pass. The new seeds are
  therefore directly comparable to the released seed-0 numbers.
- SimCLR pre-training for seed 1 **completed** (6 epochs, NT-Xent 5.271 -> 4.600
  against a chance level of 6.236; 3121 s total). Encoder written to
  `src/cache/ssl_encoder_seed1.pt` (2,351,813 bytes) and archived as a checkpoint
  artifact alongside `ssl_history_seed1.json`.
- `sam` seed 1 had just started. `ssl_sam` seed 1 had not started.

## Resuming

```bash
cd src && python run_missing_seeds.py
```

`run_missing_seeds.py` is idempotent: it skips pre-training when
`cache/ssl_encoder_seed1.pt` exists, and skips any arm whose record is already in
`runs_camelyon17/`. So an interrupted resume costs only the arm in flight, and the
~52 min of pre-training is never repeated. If the encoder is missing, restore it
from the checkpoint artifact rather than re-running pre-training.

Expect ~18 h per SAM arm on 8 threads, but budget from your own first epoch:
wall-clock here is dominated by machine contention, not by the arm (two runs of
the *identical* `ssl` configuration took 16.7 h and 4.4 h).

## After both arms land

```bash
python src/bootstrap_ci.py      # per-patch intervals, both seeds
python src/flatness.py          # sharpness at the SAM radius, new checkpoints
python src/make_figures.py      # regenerates fig1 decomposition + fig2/3
```

Then update, in this order:

1. `results/camelyon17/results_raw.csv` gains two rows; re-derive the
   decomposition (`I`, `c`, net) for `ood_test_auroc` and `id_val_auroc` as the
   mean over both seeds.
2. Paper abstract and Section "Results": the Camelyon17 interaction, `c`, and net
   effect, plus the shift-axis interaction range (currently "+0.004 to +0.161").
3. Paper Limitations: delete the one-seed caveat for these arms and replace it
   with the two-seed spread. The `rho` sweep remains outstanding.
4. `README.md`: the headline numbers block and the compute table.

Every number in the paper and README was verified against
`results/*/`*`.csv`* at the time of writing; re-verify after regenerating rather
than editing figures by hand.
