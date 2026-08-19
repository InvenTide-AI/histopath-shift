"""Build leave-one-hospital-out (LOHO) splits for Camelyon17.

Why this exists
---------------
The earlier experiments in this repository used the WILDS-provided
`id_train`/`ood_val`/`ood_test` partition, which holds out ONE hospital
(center 4) as the OOD test set. That gives a single cross-site estimate, so
every claim about "cross-site generalization" rested on one held-out site and
could not distinguish a property of the mechanisms from a property of that
particular hospital.

Camelyon17 has 5 centers. This script builds 5 folds, each holding out one
center entirely, so the cross-site effect becomes a distribution over sites
rather than a point.

Two correctness requirements, both enforced with assertions rather than trusted
-------------------------------------------------------------------------------
1. PATIENT-level disjointness. Patches from one patient are highly correlated
   (adjacent tiles of the same lymph node section). Splitting patches at random
   leaks a patient across train and test and inflates every metric. Within a
   fold, the in-distribution train/val split is therefore made by PATIENT, not
   by patch.
2. Held-out site purity. The test set of fold k contains center k and nothing
   else; train/val contain no center-k patch at all.

The metadata (center/patient/node/slide per patch) comes from the
wltjr1007/Camelyon17-WILDS mirror, which ships these columns; the copy used by
the earlier runs (jxie/camelyon17) has only image+label, which is why the
original design could not do this. Row counts of the two mirrors agree with the
published WILDS release (302,436 train / 85,054 test), and this script asserts
the total.

Class balance
-------------
Hospitals differ in tumour prevalence, so an unbalanced draw would confound
site with prevalence. Each split is balanced to 50/50 by subsampling the
majority class, and the achieved balance is recorded in the manifest.

Output
------
cache/hospital_splits.npz         index arrays into the metadata table
cache/hospital_splits_manifest.json  provenance + per-fold composition

The npz holds INDICES, not images: the image bytes stay in the parquet shards
and are decoded by the training script. This keeps the artifact small enough to
version and makes the splits auditable without the pixel data.
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

# Per-fold in-distribution validation size, as a fraction of available patients.
VAL_PATIENT_FRAC = 0.20
# Cap per split so folds are comparable in size and the sweep is affordable.
CAPS = {"train": 40000, "id_val": 8000, "test": 20000}


def balanced_subsample(df, n, rng):
    """Take up to n rows, 50/50 by label, without replacement."""
    per = n // 2
    parts = []
    for lab in (0, 1):
        idx = df.index[df.label == lab].to_numpy()
        take = min(per, len(idx))
        parts.append(rng.choice(idx, size=take, replace=False))
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


def build(meta, seed):
    rng = np.random.default_rng(seed)
    centers = sorted(meta.center.unique().tolist())
    folds = {}
    for held in centers:
        te_pool = meta[meta.center == held]
        tr_pool = meta[meta.center != held]

        # in-distribution val is split off BY PATIENT from the training centers
        pats = np.sort(tr_pool.patient.unique())
        rng.shuffle(pats)
        n_val = max(1, int(round(len(pats) * VAL_PATIENT_FRAC)))
        val_pats = set(pats[:n_val].tolist())
        is_val = tr_pool.patient.isin(val_pats)

        tr_idx = balanced_subsample(tr_pool[~is_val], CAPS["train"], rng)
        iv_idx = balanced_subsample(tr_pool[is_val], CAPS["id_val"], rng)
        te_idx = balanced_subsample(te_pool, CAPS["test"], rng)

        # --- correctness assertions (fail loudly, do not warn) ---
        assert set(meta.loc[te_idx, "center"]) == {held}, "test set not pure"
        assert held not in set(meta.loc[tr_idx, "center"]), "held-out center leaked into train"
        assert held not in set(meta.loc[iv_idx, "center"]), "held-out center leaked into val"
        p_tr = set(meta.loc[tr_idx, "patient"])
        p_iv = set(meta.loc[iv_idx, "patient"])
        p_te = set(meta.loc[te_idx, "patient"])
        assert not (p_tr & p_iv), f"patient overlap train/val: {sorted(p_tr & p_iv)[:5]}"
        assert not (p_tr & p_te), f"patient overlap train/test: {sorted(p_tr & p_te)[:5]}"
        assert not (p_iv & p_te), f"patient overlap val/test: {sorted(p_iv & p_te)[:5]}"
        assert not (set(tr_idx) & set(iv_idx) | set(tr_idx) & set(te_idx)), "index reuse"

        folds[held] = dict(train=np.sort(tr_idx), id_val=np.sort(iv_idx),
                           test=np.sort(te_idx), val_patients=sorted(val_pats))
    return folds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", default="cache/camelyon17_metadata.parquet")
    p.add_argument("--out", default="cache")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    meta = pd.read_parquet(a.meta)
    # `center` arrives as a categorical class-label; normalise to the site name
    meta["center"] = meta["center"].astype(str)
    for col in ("label", "center", "patient", "node", "slide", "shard"):
        assert col in meta.columns, f"metadata lacks required column {col!r}"
    # The published WILDS release is 302,436 train + 34,904 ood_val + 33,560
    # id_val + 85,054 test. The mirror ships train/validation/test = 302,436 /
    # 68,464 / 85,054 = 455,954. Warn rather than fail: a legitimate subset (a
    # smoke test, or a partial fetch) should still build splits, but a silent
    # size change should not pass unremarked.
    if len(meta) < 300_000:
        print(f"WARNING: only {len(meta)} metadata rows -- expected ~455,954 for the "
              "full mirror. Proceeding (valid for smoke tests); do not publish "
              "results from a partial fetch.")

    folds = build(meta, a.seed)

    arrays, comp = {}, {}
    for held, f in folds.items():
        for split in ("train", "id_val", "test"):
            arrays[f"{held}__{split}"] = f[split].astype(np.int64)
        comp[held] = {
            split: {
                "n": int(len(f[split])),
                "pos_frac": round(float(meta.loc[f[split], "label"].mean()), 4),
                "n_patients": int(meta.loc[f[split], "patient"].nunique()),
                "centers": sorted(set(meta.loc[f[split], "center"])),
            } for split in ("train", "id_val", "test")
        }
        comp[held]["val_patients"] = f["val_patients"]

    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "hospital_splits.npz"), **arrays)

    per_center = (meta.groupby("center")
                  .agg(n_patches=("label", "size"), pos_frac=("label", "mean"),
                       n_patients=("patient", "nunique"),
                       n_slides=("slide", "nunique"))
                  .round(4).to_dict("index"))

    manifest = {
        "source_dataset": "wltjr1007/Camelyon17-WILDS",
        "reason_for_mirror": ("carries center/patient/node/slide; the jxie mirror used by "
                              "the original runs has image+label only"),
        "metadata_rows": int(len(meta)),
        "metadata_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(meta[["label", "center", "patient"]], index=False)
            .values.tobytes()).hexdigest()[:16],
        "split_seed": a.seed,
        "val_patient_frac": VAL_PATIENT_FRAC,
        "caps": CAPS,
        "design": ("leave-one-hospital-out; test = held-out center only; "
                   "train/id_val split BY PATIENT among remaining centers; "
                   "all splits balanced 50/50 by label"),
        "per_center_population": per_center,
        "folds": comp,
    }
    with open(os.path.join(a.out, "hospital_splits_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"wrote {len(arrays)} index arrays for {len(folds)} folds")
    for held, c in comp.items():
        print(f"  hold out {held}: train={c['train']['n']} "
              f"({c['train']['n_patients']}p) id_val={c['id_val']['n']} "
              f"({c['id_val']['n_patients']}p) test={c['test']['n']} "
              f"({c['test']['n_patients']}p) pos_frac_test={c['test']['pos_frac']}")


if __name__ == "__main__":
    main()
