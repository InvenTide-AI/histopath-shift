# Pool-size test on the CRC stain-shift axis

Pre-registered before any arm was run. Written after the analysis in
`shift_characteristics.csv` raised, and then partly falsified, a
pool-coverage explanation for why SSL+SAM clears baseline on Camelyon17 but
not on NCT-CRC-HE.

## What prompted this, and what was wrong with the first framing

The original reading was that Camelyon17's SSL pool "spans the shift axis"
while the CRC pool does not. **That is false.** Camelyon17's
`unlabeled_train` is hospitals 0/3/4 and explicitly excludes test hospital 2
(`data_manifest.json`: `"unlabeled_train": "0,3,4 (no test-hospital
leakage)"`). Its encoder sees appearance *diversity*, never the *target*
appearance.

So the obvious experiment — add NONORM tiles to the CRC pool — is the wrong
one. It would grant the encoder the test appearance, which is unsupervised
domain adaptation, not domain generalization, and would not be comparable to
Camelyon17 in either direction.

The surviving difference that is testable is **pool size**: 40,000 tiles on
Camelyon17 against 10,226 on CRC, a 3.9x gap that was confounded with the
dataset comparison from the start.

## Hypothesis

H1. At matched pool size (40k), the SSL+SAM interaction on the CRC NONORM
axis grows enough that SSL+SAM exceeds the supervised baseline, as it does on
Camelyon17.

H0. Pool size is not the operative variable; SSL+SAM remains below baseline
on the NONORM axis at 40k.

## Design

| | original | this test |
|---|---|---|
| labeled train | 20,000 (seed 20240) | **identical** |
| id_val | 4,000 | **identical** |
| SSL pool | 10,226 tiles, 21.8% TUM | 40,000 tiles, ~5.8% TUM |
| NONORM test | 4,000 | **identical** |
| arms re-run | — | `ssl`, `ssl_sam` (seeds 0, 1) |
| arms reused | — | `baseline`, `sam` (pool-independent) |

`prepare_crc_pool40k.py` replays the original sequential row-group allocation
under the same seed before allocating the new pool, so train/id_val are
unchanged and the new pool is disjoint from all three original splits. Only
the pool varies.

## The confound this test cannot avoid

**Pool size and pool composition are inseparable in this dataset.** Class 8
(TUM) holds 14,317 tiles; train + id_val + the original pool already consumed
14,226, leaving 91. Therefore:

- a size-matched 40k pool is at most 5.8% tumour (down from 21.8%);
- a composition-matched pool caps at 10,644 tiles — 1.04x the original, far
  too small to test a size effect.

Two mitigations, neither complete. First, SimCLR is label-free, so dilution
changes the tissue *mix* the encoder sees rather than any supervision signal.
Second, Camelyon17's own pool was uniform random over `unlabeled_train` with
no class balancing, so an unbalanced pool is the closer analogue of the
condition being replicated. Even so, a positive result would not distinguish
"more tiles" from "more diverse non-tumour tissue," and this must be stated
in any writeup.

## Analysis, fixed in advance

Primary quantity: `net_vs_baseline` = mean OOD AUROC(`ssl_sam`) −
mean OOD AUROC(`baseline`) on `shift_nonorm`, over matched seeds {0, 1}.

Secondary: the interaction term
`ssl_sam − ssl − sam + baseline`, reported per seed as well as averaged,
because the two-seed spread on this axis was 0.051–0.123 originally —
sign stable, magnitude not.

Decision rule stated now to prevent post-hoc reading. All quantities are
computed **within** this six-arm set, on its own `baseline`; the published
numbers are not a comparator (different splits, different train size).

Let `net(p) = ssl_sam_p{p} − baseline` on `shift_nonorm_auroc`.

- **H1 (pool coverage) supported** only if `net(20k) > 0` **and**
  `net(20k) > net(10k)`, **for both seeds**.
- **H0 (pool size is not the operative variable)** if `net(20k) ≤ 0`, even
  with a positive interaction term. That is the original finding — interaction
  +0.087 with net −0.067 — and reproducing it here, at a *more* SSL-favourable
  train size, strengthens the paper's negative replication.
- **Ambiguous** if the two seeds disagree in sign on `net(20k) − net(10k)`.
  This will be reported as ambiguous, not resolved by adding a third seed
  after seeing the first two.

Two seeds cannot support an inferential claim, and no p-value will be
computed from n=2. This is a directional check on a specific confound, not a
hypothesis test. The 10k arms additionally serve as an internal replication:
if they do not reproduce the paper's qualitative pattern (positive
interaction, negative net), that itself is the finding and the pool-size
contrast becomes uninterpretable.

## Execution notes

The workspace cache did not survive, so the labeled splits are rebuilt from
the public shards before the pool is built. `build_rg_index.py` was recovered
from the session archive (it had been gitignored as reproducible) and its
rebuilt index reproduces the archived per-shard label inventory exactly —
100,000 tiles, all nine class totals identical — which confirms the split
allocation will reproduce the original train/id_val bit-for-bit under seed
20240.

`--pool`/`--tag` on `pretrain_ssl_crc.py` and `--enc-tag` on
`run_arm_crc.py` were added for this test; all three default to the original
filenames, so the published runs still reproduce with the unmodified command
lines.

Unlike the original runs, this rerun retains per-run prediction files
(`{arm}_seed{seed}_preds.npz`), so paired bootstrap contrasts are computable
here even though they are not recomputable for the published numbers.

## Design correction: the 40k pool was infeasible, pools are now nested

The original design proposed a 40,000-tile pool against the published ~10k,
and flagged pool size / pool composition confounding as a *risk*. Building it
showed the risk was in fact a hard arithmetic barrier, and worse than assumed.

The NCT-CRC-HE cohort contains **14,317 TUM tiles**, and
`train`(20k) + `id_val`(4k) + the original pool already consume **14,252** of
them; the 65 that remain sit inside already-reserved row groups. Any larger
pool can therefore be filled only with non-tumour tissue. The pool actually
built came out **20,000 tiles at 0% tumour** — 1.95x the size, but with the
target class entirely absent. An SSL encoder pre-trained on it would never see
tumour morphology, so a drop in transfer would be attributable to composition,
not size, and the test would manufacture the negative result it is meant to
probe. That build was discarded.

**Corrected design — nested, composition-matched pools.**

| | tiles | tumour | source |
|---|---|---|---|
| `pool10k` | 10,000 | 21.96% | class-stratified subset of `pool20k` |
| `pool20k` | 20,000 | 21.96% | drawn disjoint from train/id_val |

The small pool is a strict subset of the large one, so composition is
identical *by construction* rather than by matched sampling, and the contrast
isolates size at 2.0x. Nesting is verified on tile content (SHA-1 per tile,
multiset containment), not on indices.

To free the tumour headroom this requires, **`train` shrinks from 20,000 to
15,000**. Budget: 7,500 (train) + 2,000 (id_val) + 4,392 (pool20k) = 13,892 of
14,317, leaving 425 spare. `id_val`, `shift_nonorm`, and `shift_external` are
unchanged in size; the two shift splits are byte-identical to those already
built, and were carried over after verifying their row-group plans match.

Consequences to state plainly when reporting:

- These six arms are **not** a replication of the published setup — the
  labeled training set is 25% smaller. They are a self-contained experiment
  whose internal comparisons are valid; cross-reading them against the
  paper's absolute numbers is not.
- A smaller labeled set widens the headroom in which SSL can help, so the
  10k-pool arms here are a *more* favourable setting for SSL than the paper's.
  If SSL still fails to convert on the stain axis, that strengthens rather
  than weakens the original finding.
- Checkpoint files are keyed on a hash of the row-group plan, not the split
  name, because `id_val` is allocated after a smaller `train` and would
  otherwise silently restore tiles belonging to the old plan.

Arms: `baseline`, `sam`, `ssl_p10k`, `ssl_sam_p10k`, `ssl_p20k`,
`ssl_sam_p20k`, seeds 0 and 1.

## Split reproducibility bug found while rebuilding (and its consequence)

Rebuilding the lost cache exposed a real defect in `prepare_crc_data.py`.
Splits were allocated from a single shared `RandomState`. `plan_row_groups`
runs before the per-split checkpoint restore, but `fetch` consumes three
further draws from the same stream — so a run resumed from checkpoint skipped
those draws and every *later* split was planned from a shifted rng state. The
splits therefore depended on the interruption history of the run, not only on
the seed.

Evidence: an uninterrupted rebuild produced `unlabeled` n=10,252 and
`shift_nonorm` with 42 row groups, against the original manifest's n=10,226
and 44. `train` (204 row groups) and `id_val` (43) matched exactly, which is
what pins the cause to a downstream stream shift rather than a different seed.
Since the original run's own interruption pattern was never recorded, the
original stream is unrecoverable and the original tile-level splits cannot be
reproduced.

**Consequence for this test.** The stain-shift test set is a different sample
of tiles than the published runs used, so the published `baseline` (0.7908) and
`sam` (0.7807) numbers are not a valid comparison for newly trained arms —
a difference would confound pool size with test-set resampling. All six arms
are therefore rerun on one internally consistent split set:
`baseline`, `sam`, `ssl@10k`, `ssl_sam@10k`, `ssl@40k`, `ssl_sam@40k`,
seeds 0 and 1. The 10k arms serve as the within-splits reference, and they
double as an independent replication of the paper's original negative finding.

**Fix.** Each split now draws from its own stream derived from
`sha256(seed:split_name)` (`split_rng`), making allocation invariant to how
many splits were restored from checkpoint. Verified directly: planning with
nothing restored and with `train`+`id_val` restored yields identical row-group
sets for all four splits. This changes the splits relative to *both* the
original and my first rebuild, so the cache was rebuilt once more under the
fixed planner; the shipped splits are reproducible from the shipped script.

**Published results are unaffected.** The paper's numbers stand as measured;
what is lost is only the ability to regenerate their exact tile samples. The
manifest records n and class balance for each split, which is what the
reported comparisons depend on.

## Reference values to beat (original 10k pool, matched seeds)

```
baseline  shift_nonorm AUROC  0.7908
ssl_sam   shift_nonorm AUROC  0.7242
net_vs_baseline                -0.0666
interaction                    +0.0871   (per seed: 0.0513, 0.1229)
```
