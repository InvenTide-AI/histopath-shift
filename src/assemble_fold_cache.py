#!/usr/bin/env python3
"""Assemble one protocol-matched fold's cache from the decoded union store.

Writes, into <out>/fold{t}/ :
  splits.npz         Xtr,ytr (60000) Xiv,yiv (12000) Xov,yov (12000) Xot,yot (25000)
  unlabeled.npz      Xu (40000)
  data_manifest.json provenance + this fold's hospital roles

Key names, role sizes and the 64x64 center crop match prepare_data.py's PLAN exactly,
so run_arm.py / pretrain_ssl.py consume this cache unmodified via SPLITS_FILE / --data.

Verifies before writing: exact role sizes, exact 50/50 class balance in every labeled
role, no index reuse, ood_test pure in the held-out center, ood_val pure in the
validation center, and neither center present in train or the unlabeled pool.
"""
import argparse, json, os, time
import numpy as np, pandas as pd

ROLES = {"train": ("Xtr", "ytr", 60000), "id_val": ("Xiv", "yiv", 12000),
         "ood_val": ("Xov", "yov", 12000), "ood_test": ("Xot", "yot", 25000)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, help="held-out test center 0-4")
    ap.add_argument("--store", default="union_store")
    ap.add_argument("--folds", default="protocol_matched_folds.npz")
    ap.add_argument("--metadata", default="camelyon17_metadata.parquet")
    ap.add_argument("--out", default="cache")
    ap.add_argument("--compress", action="store_true",
                    help="savez_compressed as prepare_data.py did (slow; ~1.8 GB uncompressed)")
    a = ap.parse_args()
    t = a.fold

    md = pd.read_parquet(a.metadata)
    Z = np.load(a.folds)
    gidx = np.load(os.path.join(a.store, "union_gidx.npy"))
    X = np.load(os.path.join(a.store, "patches_union.npy"), mmap_mode="r")
    assert len(gidx) == len(X), (len(gidx), len(X))
    where = {int(g): i for i, g in enumerate(gidx)}          # global row -> store row

    idx = {r: Z[f"t{t}__{r}"] for r in list(ROLES) + ["unlabeled"]}
    test_c = int(md.loc[idx["ood_test"], "center"].iloc[0])
    val_c = int(md.loc[idx["ood_val"], "center"].iloc[0])

    # ---- verify the fold definition -------------------------------------------------
    for r, (_, _, n) in ROLES.items():
        assert len(idx[r]) == n, (r, len(idx[r]), n)
        y = md.loc[idx[r], "label"].to_numpy()
        assert y.sum() * 2 == n, (r, "class balance", int(y.sum()), n)
    assert len(idx["unlabeled"]) == 40000, len(idx["unlabeled"])
    allix = np.concatenate([idx[r] for r in ROLES])
    assert len(np.unique(allix)) == len(allix), "index reused across labeled roles"
    assert set(md.loc[idx["ood_test"], "center"]) == {test_c}
    assert set(md.loc[idx["ood_val"], "center"]) == {val_c}
    for r in ("train", "id_val", "unlabeled"):
        cs = set(md.loc[idx[r], "center"].tolist())
        assert test_c not in cs and val_c not in cs, (r, "leak", cs)

    d = os.path.join(a.out, f"fold{t}")
    os.makedirs(d, exist_ok=True)
    save = np.savez_compressed if a.compress else np.savez
    t0 = time.time()

    lab = {}
    for r, (xk, yk, _) in ROLES.items():
        rows = np.array([where[int(g)] for g in idx[r]], dtype=np.int64)
        lab[xk] = np.ascontiguousarray(X[rows])
        lab[yk] = md.loc[idx[r], "label"].to_numpy().astype(np.int64)
    save(os.path.join(d, "splits.npz"), **lab)

    rows = np.array([where[int(g)] for g in idx["unlabeled"]], dtype=np.int64)
    save(os.path.join(d, "unlabeled.npz"), Xu=np.ascontiguousarray(X[rows]))

    man = {"source": "huggingface.co/datasets/wltjr1007/Camelyon17-WILDS",
           "crop": "center 64x64 of 96x96", "label": "tumor in central 32x32 region",
           "protocol": "matched to published run: 3 train centers, epoch selected on a "
                       "held-out OOD validation hospital, test hospital touched once",
           "fold": t, "train_centers": sorted(set(md.loc[idx["train"], "center"].tolist())),
           "ood_val_center": val_c, "ood_test_center": test_c,
           "shapes": {k: list(v.shape) for k, v in lab.items()},
           "reproduces_published_run": bool(test_c == 2 and val_c == 1)}
    json.dump(man, open(os.path.join(d, "data_manifest.json"), "w"), indent=1)

    mb = sum(os.path.getsize(os.path.join(d, f)) for f in ("splits.npz", "unlabeled.npz")) / 1e9
    print(f"fold{t}: train {sorted(man['train_centers'])} | ood_val {val_c} | ood_test {test_c}")
    print(f"  wrote {d} ({mb:.2f} GB) in {time.time()-t0:.0f}s; all checks passed")

if __name__ == "__main__":
    main()
