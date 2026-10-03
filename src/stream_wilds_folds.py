"""Stream Camelyon17-WILDS archive.tar.gz straight into fold-cache npz files.

Motivation: on GPFS-backed scratch, extracting the 455k-patch tarball to disk
is disk-io-bound on file-creation and takes multiple hours.  A single-pass
streaming decode -- read the tar, decode selected PNGs into arrays, buffer per
fold, drop the buffers to compressed npz at end -- writes only 11 files total
(5 splits.npz + 5 unlabeled.npz + 1 metadata copy), so the file-creation cost
disappears.

Requires the tarball at <wilds_dir>/archive.tar.gz and the extracted
metadata.csv alongside it (WILDS auto-extracts the CSV first).  All patch
selection is metadata-driven, so no external index is needed.
"""
from __future__ import annotations
import argparse, io, json, os, tarfile, time
import numpy as np, pandas as pd
from PIL import Image

CROP = 64


def per_fold_indices(meta: pd.DataFrame, t: int, seed: int,
                     n_train=30000, n_iv=12000, n_ov=12000, n_ot=25000,
                     n_unl=40000):
    """Return dict[fold] -> {role -> set of row indices in metadata order}."""
    v = (t - 1) % 5
    train_centers = [c for c in range(5) if c not in (t, v)]
    rng = np.random.default_rng(seed + t)
    idx = np.arange(len(meta))
    y = meta["tumor"].values.astype(np.int64)
    c = meta["center"].values.astype(np.int64)
    p = meta["patient"].values

    def strat(pool, cap):
        if cap is None or cap >= len(pool):
            return pool
        per = cap // 2
        out = []
        for cls in (0, 1):
            c_idx = pool[y[pool] == cls]
            out.append(rng.choice(c_idx, size=min(per, len(c_idx)), replace=False))
        return np.sort(np.concatenate(out))

    tr_pool = idx[np.isin(c, train_centers)]
    ov_pool = idx[c == v]
    ot_pool = idx[c == t]
    tr_patients = sorted(set(p[tr_pool].tolist()))
    rng.shuffle(tr_patients)
    iv_p = set(tr_patients[: max(1, len(tr_patients) // 5)])
    iv_pool = tr_pool[np.isin(p[tr_pool], list(iv_p))]
    tr_pool = tr_pool[~np.isin(p[tr_pool], list(iv_p))]

    tr = strat(tr_pool, n_train)
    iv = strat(iv_pool, n_iv)
    ov = strat(ov_pool, n_ov)
    ot = strat(ot_pool, n_ot)
    unl_pool = np.setdiff1d(idx[np.isin(c, train_centers)],
                            np.concatenate([tr, iv]))
    unl = rng.choice(unl_pool, size=min(n_unl, len(unl_pool)), replace=False)
    unl.sort()
    remap = {ci: k for k, ci in enumerate(sorted(train_centers))}
    site = np.array([remap[ci] for ci in c[tr].tolist()], dtype=np.int64)
    return {"train": tr, "id_val": iv, "ood_val": ov, "ood_test": ot,
            "unlabeled": unl, "site_of_train": site, "val_center": v,
            "train_centers": train_centers}


def path_of(row):
    return (f"./patches/patient_{row.patient}_node_{row.node}/"
            f"patch_patient_{row.patient}_node_{row.node}_x_{row.x_coord}"
            f"_y_{row.y_coord}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wilds_dir", default="./data/wilds/camelyon17_v1.0")
    ap.add_argument("--out_root", default="cache")
    ap.add_argument("--folds", default="0 1 2 3 4")
    ap.add_argument("--seed", type=int, default=20240)
    args = ap.parse_args()

    meta = pd.read_csv(os.path.join(args.wilds_dir, "metadata.csv"),
                       index_col=0, dtype={"patient": "str"})
    folds = [int(x) for x in args.folds.split()]
    print(f"{len(meta)} patches; folds {folds}")

    # 1) Choose all indices per fold, per role. Build a global path -> [(fold, role, pos)] table.
    plan = {t: per_fold_indices(meta, t, args.seed) for t in folds}
    all_needed = {}
    for t, roles in plan.items():
        for role in ("train", "id_val", "ood_val", "ood_test", "unlabeled"):
            for pos, ridx in enumerate(roles[role]):
                p = path_of(meta.iloc[ridx])
                all_needed.setdefault(p, []).append((t, role, pos, ridx))
    print(f"unique paths to fetch: {len(all_needed)}")

    # 2) Allocate per-fold buffers.
    buffers = {}
    for t, roles in plan.items():
        buffers[t] = {
            "Xtr": np.empty((len(roles["train"]), CROP, CROP, 3), dtype=np.uint8),
            "Xiv": np.empty((len(roles["id_val"]), CROP, CROP, 3), dtype=np.uint8),
            "Xov": np.empty((len(roles["ood_val"]), CROP, CROP, 3), dtype=np.uint8),
            "Xot": np.empty((len(roles["ood_test"]), CROP, CROP, 3), dtype=np.uint8),
            "Xu":  np.empty((len(roles["unlabeled"]), CROP, CROP, 3), dtype=np.uint8),
        }
    role_to_key = {"train": "Xtr", "id_val": "Xiv", "ood_val": "Xov",
                   "ood_test": "Xot", "unlabeled": "Xu"}

    # 3) Stream the tarball, matching each member against the wanted set.
    tar_path = os.path.join(args.wilds_dir, "archive.tar.gz")
    print(f"scanning {tar_path} ({os.path.getsize(tar_path)/1e9:.1f} GB)")
    t0 = time.time(); found = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar:
            if not m.isfile():
                continue
            hits = all_needed.pop(m.name, None)
            if not hits:
                continue
            f = tar.extractfile(m)
            if f is None: continue
            buf = f.read()
            im = Image.open(io.BytesIO(buf)).convert("RGB")
            w, h = im.size
            l, tp = (w - CROP) // 2, (h - CROP) // 2
            arr = np.asarray(im.crop((l, tp, l + CROP, tp + CROP)), dtype=np.uint8)
            for (t, role, pos, ridx) in hits:
                buffers[t][role_to_key[role]][pos] = arr
            found += 1
            if found % 5000 == 0:
                print(f"  {found} patches decoded ({time.time() - t0:.0f}s)", flush=True)
            if not all_needed:
                break
    print(f"decoded {found} unique patches, {time.time() - t0:.0f}s wall")
    if all_needed:
        print(f"WARNING: {len(all_needed)} paths not found in tarball")

    # 4) Write per-fold caches.
    y = meta["tumor"].values.astype(np.int64)
    for t, roles in plan.items():
        out = os.path.join(args.out_root, f"fold{t}")
        os.makedirs(out, exist_ok=True)
        np.savez_compressed(os.path.join(out, "splits.npz"),
                            Xtr=buffers[t]["Xtr"], ytr=y[roles["train"]], str=roles["site_of_train"],
                            Xiv=buffers[t]["Xiv"], yiv=y[roles["id_val"]],
                            Xov=buffers[t]["Xov"], yov=y[roles["ood_val"]],
                            Xot=buffers[t]["Xot"], yot=y[roles["ood_test"]])
        np.savez_compressed(os.path.join(out, "unlabeled.npz"), Xu=buffers[t]["Xu"])
        with open(os.path.join(out, "manifest.json"), "w") as fh:
            json.dump({"fold": t, "val_center": roles["val_center"],
                       "train_centers": roles["train_centers"],
                       "n_train": len(roles["train"]), "n_id_val": len(roles["id_val"]),
                       "n_ood_val": len(roles["ood_val"]), "n_ood_test": len(roles["ood_test"]),
                       "n_unlabeled": len(roles["unlabeled"]),
                       "seed": args.seed + t}, fh, indent=2)
        print(f"fold{t}: wrote splits.npz + unlabeled.npz")


if __name__ == "__main__":
    main()
