"""v2 Camelyon17-WILDS fold caches (64 and 96 px from one streaming pass).

ood_val / ood_test: v1 src/stream_wilds_folds.per_fold_indices is run verbatim
(seed 20240+t), so Xov / Xot hold exactly the v1 patches.
train / id_val / unlabelled are re-drawn (v2 changes):
  * id-val is STRATIFIED BY TRAINING CENTRE: k = max(1, n_c // 5) patients of
    every training centre c are held out (n_c = 7..10 patients -> k = 1..2),
    drawn by seeded rejection sampling until the held-out patients have
    >= 1000 tumour and >= 1000 normal patches and <= 40 % of the centre's
    tumour patches (so id-val has both classes from every training centre and
    train keeps most of each centre's tumour).
    v1 held out 1/5 of the pooled training patients, so in folds 0, 1 and 4
    the id-val split had no patient of one training centre (incl. the largest
    centre in folds 0 and 1) -- bad for ID-val model selection;
  * D11: the unlabelled pool excludes every patch of the id-val patients and
    the selected Xtr rows (v1 excluded only the *selected* id-val rows);
  * these draws use a separate RNG default_rng([seed, t, 404]), class-balanced
    caps as v1 (Xtr 30000, Xiv 12000, Xu 40000).

96 px = native WILDS patch (no crop); 64 px = centre crop [16:80, 16:80] of the
same patch, identical to v1's crop. Domains: centre index c -> "center{c}".
Group id = patient number. Fold t: test centre t, val centre (t-1) mod 5.
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import tarfile
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

NATIVE = 96
# id-val patient set per training centre: k = max(1, n_c // 5) patients drawn at random
# (rejection sampling with the fold RNG) until the held-out patients have >= MIN_IV_CLASS
# patches of each class and <= MAX_IV_POS_SHARE of the centre's tumour patches (Camelyon17
# patients are extremely heterogeneous: 5 .. 56666 tumour patches per patient).
MIN_IV_CLASS = 1000
MAX_IV_POS_SHARE = 0.40


def per_fold_indices(meta: pd.DataFrame, t: int, seed: int,
                     n_train=30000, n_iv=12000, n_ov=12000, n_ot=25000,
                     n_unl=40000):
    """v1 logic (stream_wilds_folds.per_fold_indices) with the D11 fix."""
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
    # --- D11: pool = training-centre rows of NON-id-val patients, minus selected train rows
    unl_pool = np.setdiff1d(tr_pool, tr)
    unl = rng.choice(unl_pool, size=min(n_unl, len(unl_pool)), replace=False)
    unl.sort()
    return {"train": tr, "id_val": iv, "ood_val": ov, "ood_test": ot,
            "unlabeled": unl, "val_center": v, "train_centers": train_centers,
            "iv_patients": sorted(iv_p), "unl_pool_size": int(len(unl_pool))}


def per_fold_indices_v2(meta: pd.DataFrame, t: int, seed: int,
                        n_train=30000, n_iv=12000, n_unl=40000):
    """ov/ot = v1 (verbatim); tr/iv/u re-drawn with centre-stratified id-val patients."""
    r1 = per_fold_indices(meta, t, seed)
    v = r1["val_center"]
    train_centers = r1["train_centers"]
    rng = np.random.default_rng([seed, t, 404])
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
    iv_p, iv_tries = {}, {}
    for cc in sorted(train_centers):
        pats = np.array(sorted(set(p[idx[c == cc]].tolist())))
        k = max(1, len(pats) // 5)
        pos_c = int(y[c == cc].sum())
        for attempt in range(1, 10001):
            cand = sorted(rng.permutation(pats)[:k].tolist())
            m = (c == cc) & np.isin(p, cand)
            npos, nneg = int(y[m].sum()), int((1 - y[m]).sum())
            if npos >= MIN_IV_CLASS and nneg >= MIN_IV_CLASS and npos <= MAX_IV_POS_SHARE * pos_c:
                break
        else:
            raise RuntimeError("no admissible id-val patient set for centre %d" % cc)
        iv_p[cc] = cand
        iv_tries[cc] = attempt
    iv_all = [x for cc in sorted(iv_p) for x in iv_p[cc]]
    iv_pool = tr_pool[np.isin(p[tr_pool], iv_all)]
    tr_pool = tr_pool[~np.isin(p[tr_pool], iv_all)]
    tr = strat(tr_pool, n_train)
    iv = strat(iv_pool, n_iv)
    unl_pool = np.setdiff1d(tr_pool, tr)
    unl = np.sort(rng.choice(unl_pool, size=min(n_unl, len(unl_pool)), replace=False))
    return {"train": tr, "id_val": iv, "ood_val": r1["ood_val"], "ood_test": r1["ood_test"],
            "unlabeled": unl, "val_center": v, "train_centers": train_centers,
            "iv_patients": iv_all, "iv_patients_by_center": {"center%d" % k: vv for k, vv in iv_p.items()},
            "iv_patients_v1": r1["iv_patients"], "unl_pool_size": int(len(unl_pool)),
            "iv_draw_attempts_by_center": {"center%d" % k: vv for k, vv in iv_tries.items()}}


def path_of(row):
    return (f"./patches/patient_{row.patient}_node_{row.node}/"
            f"patch_patient_{row.patient}_node_{row.node}_x_{row.x_coord}"
            f"_y_{row.y_coord}.png")


def decode(buf):
    im = Image.open(io.BytesIO(buf)).convert("RGB")
    a = np.asarray(im, dtype=np.uint8)
    assert a.shape == (NATIVE, NATIVE, 3), a.shape
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wilds_dir", default=os.path.join(os.environ.get("HISTOPATH_DATA", "data"), "wilds/camelyon17_v1.0"))
    ap.add_argument("--v2", default=C.V2)
    ap.add_argument("--patch", type=int, nargs="+", default=[64, 96], choices=[64, 96])
    ap.add_argument("--folds", default="0 1 2 3 4")
    ap.add_argument("--seed", type=int, default=20240)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    meta = pd.read_csv(os.path.join(a.wilds_dir, "metadata.csv"), index_col=0, dtype={"patient": "str"})
    folds = [int(x) for x in a.folds.split()]
    plan = {t: per_fold_indices_v2(meta, t, a.seed) for t in folds}
    needed = set()
    for t, r in plan.items():
        for role in ("train", "id_val", "ood_val", "ood_test", "unlabeled"):
            needed.update(r[role].tolist())
    needed = np.array(sorted(needed))
    path2row = {path_of(meta.iloc[i]): int(i) for i in needed}
    print("%d patches in metadata; %d unique rows needed" % (len(meta), len(needed)), flush=True)

    # stream tar (pigz for decompression), keep raw PNG bytes of needed members
    tar_path = os.path.join(a.wilds_dir, "archive.tar.gz")
    t0 = time.time()
    raw = {}
    proc = subprocess.Popen(["pigz", "-dc", tar_path], stdout=subprocess.PIPE, bufsize=1 << 24)
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
        for m in tar:
            if not m.isfile():
                continue
            r = path2row.get(m.name)
            if r is None:
                continue
            raw[r] = tar.extractfile(m).read()
            if len(raw) % 50000 == 0:
                print("  %d read (%.0fs)" % (len(raw), time.time() - t0), flush=True)
            if len(raw) == len(path2row):
                break
    proc.kill()
    print("read %d/%d PNGs in %.0fs" % (len(raw), len(path2row), time.time() - t0), flush=True)
    assert len(raw) == len(path2row), "missing members in tar"

    rows_sorted = np.array(sorted(raw))
    with Pool(a.workers) as pool:
        arrs = pool.map(decode, [raw[r] for r in rows_sorted], chunksize=512)
    del raw
    ALL = np.stack(arrs); del arrs
    pos_of = {int(r): k for k, r in enumerate(rows_sorted)}
    print("decoded %s in %.0fs" % (ALL.shape, time.time() - t0), flush=True)

    y = meta["tumor"].values.astype(np.int64)
    c = meta["center"].values.astype(np.int64)
    pat = meta["patient"].astype(int).values.astype(np.int64)
    role2s = {"train": "tr", "id_val": "iv", "ood_val": "ov", "ood_test": "ot", "unlabeled": "u"}
    for t in folds:
        r = plan[t]
        v = r["val_center"]
        for P in a.patch:
            off = (NATIVE - P) // 2
            parts = {}
            for role, s in role2s.items():
                rows = r[role]
                k = np.array([pos_of[int(i)] for i in rows])
                X = ALL[k][:, off:off + P, off:off + P, :] if P != NATIVE else ALL[k]
                parts[s] = dict(X=X, y=y[rows] if s != "u" else None,
                                d=np.array(["center%d" % ci for ci in c[rows]]),
                                g=pat[rows], rows=rows)
            out = os.path.join(a.v2, "cache%d" % P, "c17", "fold%d" % t)
            man = C.write_fold(
                out, "c17", t, P, a.seed + t, "center%d" % t, "center%d" % v,
                ["center%d" % ci for ci in sorted(r["train_centers"])], parts,
                group_kind="patient", source=a.wilds_dir + "/archive.tar.gz",
                notes=[
                    "ov/ot identical to v1 stream_wilds_folds.per_fold_indices (seed 20240+t)",
                    "id-val stratified by training centre: max(1, n_patients_c // 5) patients per training centre (iv_patients_by_center); tr/iv/u re-drawn with default_rng([seed, t, 404]); Xtr/Xiv therefore differ from v1 (v1 id-val missed one training centre in folds 0, 1, 4)",
                    "D11: unlabelled pool excludes all patches of id-val patients and the selected Xtr rows",
                    "96 px = native WILDS patch; 64 px = centre crop [16:80,16:80] (== v1 crop)",
                    "all splits row-shuffled (common.shuffle_perm); rows.npz holds metadata.csv row ids",
                ],
                extra={"iv_patients": r["iv_patients"], "iv_patients_by_center": r["iv_patients_by_center"],
                       "iv_patients_v1": r["iv_patients_v1"], "unl_pool_size": r["unl_pool_size"],
                       "iv_draw_attempts_by_center": r["iv_draw_attempts_by_center"],
                       "iv_rule": {"k": "max(1, n_patients_c // 5)", "min_per_class": MIN_IV_CLASS,
                                   "max_pos_share": MAX_IV_POS_SHARE},
                       "code_sha": C.code_sha(__file__)})
            print("fold%d P=%d: tr=%d iv=%d ov=%d ot=%d u=%d pos tr/iv/ov/ot=%.3f/%.3f/%.3f/%.3f" % (
                t, P, man["n_tr"], man["n_iv"], man["n_ov"], man["n_ot"], man["n_u"],
                man["pos_rate_tr"], man["pos_rate_iv"], man["pos_rate_ov"], man["pos_rate_ot"]), flush=True)
    print("done %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
