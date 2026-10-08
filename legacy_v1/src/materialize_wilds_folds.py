"""Convert a downloaded Camelyon17-WILDS directory into the paper's fold cache.

Input:   WILDS's native layout under <root>/camelyon17_v1.0/
            metadata.csv
            patches/patient_XXX_node_Y/patch_patient_XXX_node_Y_x_A_y_B.png

Output:  <out_root>/fold{t}/splits.npz  for t in 0..4
         each containing: Xtr, ytr, Xiv, yiv, Xov, yov, Xot, yot
         and a companion str (site index of each training patch, for
         GroupDRO / CORAL / IRM in run_arm_ext.py).

Fold t = held-out test center. OOD-validation center v = (t-1) mod 5 (matches
the paper's protocol; fold 2 exactly reproduces the original WILDS design).
Training draws from the remaining three centers.

Patient-disjoint by construction: metadata.csv records the source patient of
each patch and we filter on that; no patient ever appears in more than one
role of a given fold.

Sample budgets match the paper (30k train / 12k id_val / 12k ood_val / 25k
ood_test, class-stratified 50/50). Sub-sampling is seeded so a rebuild
produces the same cache modulo file system ordering.
"""
from __future__ import annotations
import argparse, io, json, os
import numpy as np, pandas as pd
from PIL import Image

CROP = 64  # center-crop each 96x96 WILDS patch to 64x64


def load_patch(root, patient, node, x, y):
    path = os.path.join(root, "patches",
                        f"patient_{patient}_node_{node}",
                        f"patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png")
    im = Image.open(path).convert("RGB")
    w, h = im.size
    l, t = (w - CROP) // 2, (h - CROP) // 2
    return np.asarray(im.crop((l, t, l + CROP, t + CROP)), dtype=np.uint8)


def stratified_sample(idx, y, n, rng):
    if n is None or n >= len(idx):
        return idx
    per = n // 2
    out = []
    for c in (0, 1):
        c_idx = idx[y[idx] == c]
        take = min(per, len(c_idx))
        out.append(rng.choice(c_idx, size=take, replace=False))
    return np.sort(np.concatenate(out))


def load_batch(root, meta_rows):
    imgs = np.empty((len(meta_rows), CROP, CROP, 3), dtype=np.uint8)
    for i, (p, n, x, y) in enumerate(meta_rows[["patient", "node",
                                                "x_coord", "y_coord"]].itertuples(index=False)):
        imgs[i] = load_patch(root, p, n, x, y)
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wilds_dir", default="./data/wilds/camelyon17_v1.0")
    ap.add_argument("--out_root", default="cache")
    ap.add_argument("--n_train", type=int, default=30000)
    ap.add_argument("--n_iv", type=int, default=12000)
    ap.add_argument("--n_ov", type=int, default=12000)
    ap.add_argument("--n_ot", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=20240)
    ap.add_argument("--folds", default="0 1 2 3 4")
    args = ap.parse_args()

    meta = pd.read_csv(os.path.join(args.wilds_dir, "metadata.csv"),
                       index_col=0, dtype={"patient": "str"})
    print(f"metadata: {len(meta)} patches; centers "
          f"{sorted(meta['center'].unique())}")
    print(meta.groupby("center")["tumor"].agg(["count", "mean"]).round(3))

    folds = [int(x) for x in args.folds.split()]
    for t in folds:
        v = (t - 1) % 5
        train_centers = [c for c in range(5) if c not in (t, v)]
        rng = np.random.default_rng(args.seed + t)
        out = os.path.join(args.out_root, f"fold{t}")
        os.makedirs(out, exist_ok=True)
        target = os.path.join(out, "splits.npz")
        if os.path.exists(target):
            print(f"fold {t}: exists, skipping"); continue

        idx = np.arange(len(meta))
        y_all = meta["tumor"].values.astype(np.int64)
        c_all = meta["center"].values.astype(np.int64)
        p_all = meta["patient"].values

        # candidate pools by role
        tr_pool = idx[np.isin(c_all, train_centers)]
        ov_pool = idx[c_all == v]
        ot_pool = idx[c_all == t]
        # id_val is drawn from the training centers but from *different*
        # patients than training. Get patient-disjoint holdout first.
        tr_patients = sorted(set(p_all[tr_pool]))
        rng.shuffle(tr_patients)
        iv_patients = set(tr_patients[: max(1, len(tr_patients) // 5)])
        iv_pool = tr_pool[np.isin(p_all[tr_pool], list(iv_patients))]
        tr_pool = tr_pool[~np.isin(p_all[tr_pool], list(iv_patients))]

        # class-stratified sampling per role
        tr_sel = stratified_sample(tr_pool, y_all, args.n_train, rng)
        iv_sel = stratified_sample(iv_pool, y_all, args.n_iv, rng)
        ov_sel = stratified_sample(ov_pool, y_all, args.n_ov, rng)
        ot_sel = stratified_sample(ot_pool, y_all, args.n_ot, rng)

        print(f"fold {t}: train c={train_centers} n={len(tr_sel)}, "
              f"id_val n={len(iv_sel)}, ood_val c={v} n={len(ov_sel)}, "
              f"ood_test c={t} n={len(ot_sel)}")

        # decode images
        Xtr = load_batch(args.wilds_dir, meta.iloc[tr_sel])
        ytr = y_all[tr_sel]
        str_ = c_all[tr_sel]  # site label per training patch, remapped 0..K-1
        remap = {c: i for i, c in enumerate(sorted(set(str_.tolist())))}
        str_ = np.array([remap[s] for s in str_.tolist()], dtype=np.int64)
        Xiv = load_batch(args.wilds_dir, meta.iloc[iv_sel]); yiv = y_all[iv_sel]
        Xov = load_batch(args.wilds_dir, meta.iloc[ov_sel]); yov = y_all[ov_sel]
        Xot = load_batch(args.wilds_dir, meta.iloc[ot_sel]); yot = y_all[ot_sel]

        np.savez_compressed(target,
                            Xtr=Xtr, ytr=ytr, str=str_,
                            Xiv=Xiv, yiv=yiv,
                            Xov=Xov, yov=yov,
                            Xot=Xot, yot=yot)
        # unlabeled pool = all remaining patches in the training centers,
        # capped for CPU budget. Feeds src/pretrain_ssl.py.
        unl_pool = idx[np.isin(c_all, train_centers)]
        unl_pool = np.setdiff1d(unl_pool, np.concatenate([tr_sel, iv_sel]))
        n_unl = min(40000, len(unl_pool))
        unl_sel = rng.choice(unl_pool, size=n_unl, replace=False); unl_sel.sort()
        Xu = load_batch(args.wilds_dir, meta.iloc[unl_sel])
        np.savez_compressed(os.path.join(out, "unlabeled.npz"), Xu=Xu)
        print(f"fold {t}: wrote {target} + unlabeled n={n_unl}")

        with open(os.path.join(out, "manifest.json"), "w") as f:
            json.dump({
                "fold": t, "val_center": v, "train_centers": train_centers,
                "n_train": int(len(tr_sel)), "n_id_val": int(len(iv_sel)),
                "n_ood_val": int(len(ov_sel)), "n_ood_test": int(len(ot_sel)),
                "n_unlabeled": int(n_unl), "seed": args.seed + t,
                "train_class_balance": [int((ytr == c).sum()) for c in (0, 1)],
                "ood_test_class_balance": [int((yot == c).sum()) for c in (0, 1)],
            }, f, indent=2)


if __name__ == "__main__":
    main()
