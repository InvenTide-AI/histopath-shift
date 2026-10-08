"""Build a small smoke-test cache in the CONTRACTS.md A format from the v1 Camelyon17
fold-0 cache (plumbing only; the 96-px copy is a bilinear upsample of the 64-px tiles).

  python src/v2/make_smoke_cache.py  [--n_tr 4000 --n_split 1000 --n_u 4000]
writes $V2/smoke/cache64/c17/fold0 and $V2/smoke/cache96/c17/fold0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

V1 = os.path.join(C.DATA_ROOT, "cache", "fold0")


def up96(X):
    out = []
    for k in range(0, len(X), 1000):
        x = torch.from_numpy(X[k:k + 1000]).permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(96, 96), mode="bilinear", align_corners=False)
        out.append(x.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tr", type=int, default=4000)
    ap.add_argument("--n_split", type=int, default=1000)
    ap.add_argument("--n_u", type=int, default=4000)
    ap.add_argument("--out_root", default=os.path.join(C.V2, "smoke"))
    a = ap.parse_args()
    t0 = time.time()
    d = np.load(os.path.join(V1, "splits.npz"))
    m1 = json.load(open(os.path.join(V1, "manifest.json")))
    rng = np.random.RandomState(0)
    arr = {}
    n = len(d["ytr"])
    idx = rng.choice(n, a.n_tr, replace=False)
    arr["Xtr"], arr["ytr"] = d["Xtr"][idx], d["ytr"][idx].astype(np.int64)
    s = d["str"][idx].astype(np.int64)
    u = np.unique(s)
    arr["dtr"] = np.searchsorted(u, s).astype(np.int64)        # remap str -> contiguous dtr
    arr["gtr"] = np.full(a.n_tr, -1, np.int64)
    for sp in ("iv", "ov", "ot"):
        y = d["y" + sp]
        idx = rng.choice(len(y), a.n_split, replace=False)
        arr["X" + sp], arr["y" + sp] = d["X" + sp][idx], y[idx].astype(np.int64)
    arr["div"] = np.full(a.n_split, -1, np.int64)
    arr["giv"] = np.full(a.n_split, -1, np.int64)
    Xu = np.load(os.path.join(V1, "unlabeled.npz"))["Xu"]
    arr["Xu"] = Xu[rng.choice(len(Xu), a.n_u, replace=False)]
    print(f"subsampled in {time.time() - t0:.0f}s", flush=True)
    tr_dom = [m1["train_centers"][i] for i in u]
    for P in (64, 96):
        out = os.path.join(a.out_root, f"cache{P}", "c17", "fold0")
        os.makedirs(out, exist_ok=True)
        for k, v in arr.items():
            if k.startswith("X") and P == 96:
                v = up96(v)
            np.save(os.path.join(out, k + ".npy"), np.ascontiguousarray(v))
        man = dict(cohort="c17", fold=0, patch_px=P, test_domain=0, val_domain=m1["val_center"],
                   train_domains=tr_dom, n_tr=a.n_tr, n_iv=a.n_split, n_ov=a.n_split, n_ot=a.n_split,
                   n_u=a.n_u, **{f"pos_rate_{sp}": float(arr["y" + sp].mean()) for sp in ("tr", "iv", "ov", "ot")},
                   group_kind="unknown (smoke)", disjointness={}, source=V1, seed=0,
                   created=time.strftime("%Y-%m-%d %H:%M:%S"),
                   notes="SMOKE cache for plumbing tests only; subsampled from v1 fold0"
                         + ("; 96 px = bilinear upsample of 64 px" if P == 96 else ""))
        json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
