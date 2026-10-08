# v2 interface contracts

`V2=$HISTOPATH_DATA/v2`. Code lives in `src/v2/` (new files only; v1 code under `src/` is not modified).

## A. Fold cache (written by the data track, read by the training track)

Directory: `$V2/cache{64,96}/{cohort}/fold{t}/` with cohort ∈ {`c17`, `canine`, `midog21`, `midogpp`}.
Every array is a plain `.npy` file (uncompressed, so `np.load(..., mmap_mode='r')` works):

| file | dtype / shape | meaning |
|---|---|---|
| `Xtr.npy` | uint8 (N,P,P,3) | labelled training patches, training domains only |
| `ytr.npy` | int64 (N,) | binary label (1 = tumour / mitotic figure) |
| `dtr.npy` | int64 (N,) | training-domain index, **0..K_train-1** (contiguous) |
| `gtr.npy` | int64 (N,) | group id (patient / slide / image) — used only for disjointness checks |
| `Xiv.npy`,`yiv.npy`,`div.npy`,`giv.npy` | | in-distribution validation: training domains, **group-disjoint from train** |
| `Xov.npy`,`yov.npy` | | leave-one-domain-out validation domain; **absent** for `midog21` |
| `Xot.npy`,`yot.npy` | | held-out test domain; **row order shuffled** with a fixed seed |
| `Xu.npy` | uint8 (M,P,P,3) | unlabelled pool from training domains only, excluding id-val groups, val domain and test domain |
| `manifest.json` | | see below |

All splits are row-shuffled with a fixed seed (no label-sorted order anywhere).
`manifest.json` keys: `cohort, fold, patch_px, test_domain, val_domain (or null), train_domains (list; index = dtr value),
n_{tr,iv,ov,ot,u}, pos_rate_{tr,iv,ov,ot}, group_kind, disjointness: {train_vs_iv: true, ...}, source, seed, created, notes`.
The same patches (same centres) are used in `cache64` and `cache96` wherever possible; P=64 or 96.

## B. Run record (written by the training track, read by the stats track)

Path: `$V2/runs/{tier}/{cohort}/fold{t}/{tag}.json` (+ `{tag}_preds.npz`, + `{tag}.pt` for the ID-selected checkpoint).
`tier` ∈ {`C`, `S`}. `tag = {arm}__{hpkey}__s{seed}` where `hpkey` is a short stable string of non-default hyperparameters (or `default`).

```json
{
  "schema": "v2",
  "arm": "erm", "tier": "C", "backbone": "cnn4|resnet50|vit_s16", "init": "random|imagenet|lunit_bt|lunit_dino|...",
  "cohort": "c17", "fold": 0, "test_domain": "...", "val_domain": "... or null", "seed": 0,
  "hparams": {"lr": 0.02, "steps": 1200, "bs": 128, "...": "every hyperparameter, including defaults"},
  "phase": "tune|final", "input_res": 64, "n_params": 583394, "n_train": 30000, "wall_s": 0.0,
  "checkpoints": [ {"step": 120, "train_loss": 0.0,
                    "id_val": {"auroc": 0, "ap": 0, "acc": 0, "bacc": 0, "f1": 0, "ece": 0, "brier": 0, "nll": 0},
                    "ood_val": {"...": 0} , "ood_test": {"...": 0} } ],
  "selected": {"id": 600, "ood": 360, "last": 1200, "oracle": 840},
  "test_at": {"id": {"auroc": 0, "...": 0}, "ood": {}, "last": {}, "oracle": {}},
  "tta": {"bnadapt": {"auroc": 0, "...": 0}, "tent": {"auroc": 0, "...": 0}},
  "pcam": {"id": {"auroc": 0, "...": 0}},
  "code_version": "sha256 of src/v2/*.py", "torch": "2.7.0", "device": "cuda"
}
```
`ood_val` is `null` and `selected.ood`/`test_at.ood` are absent when the cohort has no validation domain.
`tta` and `pcam` are present only when requested (TTA uses the ID-selected checkpoint; PCam only for `c17`).
`{tag}_preds.npz`: `y_ood_test`, `p_ood_test_id`, `p_ood_test_ood` (if any), `p_ood_test_last`, and `p_tta_bnadapt` / `p_tta_tent` if run.
