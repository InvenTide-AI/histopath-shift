"""Build scanner-complete fold caches from the multi-scanner canine SCC dataset.

The dataset (Wilm et al., arXiv:2301.04423; Zenodo 7418555) is 44 canine
cutaneous squamous cell carcinoma slides, each digitized on five scanning
systems, with tumour and six skin-tissue classes annotated as polygons on one
scanner and transferred to the other four by registration. Five scanners means
this cohort supports the identical K = 5 leave-one-domain-out protocol used for
Camelyon17, which Camelyon17's five hospitals do and MIDOG's three scanners do
not.

Two design points matter.

Scale. The five scanners differ in native resolution (0.22-0.26 um/px), so a
fixed pixel crop would cover different physical areas and the "domain effect"
would partly be a trivial zoom difference. Because the annotations are
registered, the total annotated area per slide gives a precise empirical scale
ratio: taking the square root yields linear factors of 1.000 (cs2), 1.020
(p1000), 1.116 (nz20), 1.147 (nz210) and 0.963 (gt450), with a coefficient of
variation of 0.001 across slides, and matching the scanners' published um/px to
within a few percent. We crop each scanner at its own pixel size and resize to
64, so every patch covers the same physical extent and what remains between
domains is colour, sharpness and contrast.

Patient disjointness. The same 44 slides appear under all five scanners, so
holding out a scanner alone would leave the test tissue present in training
under a different scanner. To mirror Camelyon17, where a held-out hospital also
brings held-out patients, we partition the slides into five disjoint groups and
hold out both a scanner and a slide group per fold.

Output matches the Camelyon17 cache contract so the existing arm runner is
unchanged.
"""
from __future__ import annotations
import argparse
import collections
import json
import os

import numpy as np
from matplotlib.path import Path as MplPath
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SCANNERS = ["cs2", "p1000", "nz20", "nz210", "gt450"]
SCALE_NORM = True   # set False by --no_scale_norm; recorded in the manifest
PATCH = 64
TUMOUR_CATEGORY = 11          # "SCC"
TISSUE_CATEGORIES = {1, 2, 3, 4, 5, 6}   # bone, cartilage, dermis, epidermis, subcutis, inflamm/necrosis


def scale_factors(ann) -> dict:
    """Linear pixel-size ratio of each scanner relative to cs2, from registered
    polygon areas. Robust: cv across slides is ~0.001."""
    meta = {im["id"]: im["file_name"][:-4].rsplit("_", 1) for im in ann["images"]}
    tot = collections.defaultdict(float)
    for a in ann["annotations"]:
        slide, sc = meta[a["image_id"]]
        tot[(slide, sc)] += a["area"]
    slides = sorted({k[0] for k in tot})
    out = {}
    for sc in SCANNERS:
        r = [np.sqrt(tot[(s, sc)] / tot[(s, "cs2")]) for s in slides
             if (s, sc) in tot and tot.get((s, "cs2"), 0) > 0]
        out[sc] = float(np.median(r)) if r else 1.0
    return out


def polygon_points(seg, n, rng):
    """Sample up to n points uniformly inside a COCO polygon by rejection."""
    if isinstance(seg, str):
        seg = json.loads(seg)
    if seg and isinstance(seg[0], list):
        seg = seg[0]
    xy = np.asarray(seg, dtype=float).reshape(-1, 2)
    if len(xy) < 3:
        return np.zeros((0, 2))
    path = MplPath(xy)
    lo, hi = xy.min(0), xy.max(0)
    got = []
    for _ in range(12):
        cand = rng.uniform(lo, hi, size=(max(n * 4, 32), 2))
        inside = cand[path.contains_points(cand)]
        got.append(inside)
        if sum(len(g) for g in got) >= n:
            break
    pts = np.concatenate(got) if got else np.zeros((0, 2))
    return pts[:n]


def extract(src, ann_path, per_image=420, verbose=True, scale_norm=True):
    ann = json.load(open(ann_path))
    scales = scale_factors(ann) if scale_norm else {s: 1.0 for s in SCANNERS}
    if verbose:
        print("scanner scale factors (linear, vs cs2):",
              {k: round(v, 3) for k, v in scales.items()}, flush=True)
    by_image = collections.defaultdict(list)
    for a in ann["annotations"]:
        by_image[a["image_id"]].append(a)

    recs = []          # (patch, label, scanner_idx, slide)
    rng = np.random.default_rng(0)
    for im_rec in ann["images"]:
        fn = im_rec["file_name"]
        path = os.path.join(src, fn)
        if not os.path.exists(path):
            continue
        slide, sc = fn[:-4].rsplit("_", 1)
        s_idx = SCANNERS.index(sc)
        crop_px = int(round(PATCH * scales[sc]))
        half = crop_px // 2
        img = np.asarray(Image.open(path).convert("RGB"))
        H, W = img.shape[:2]

        anns = by_image.get(im_rec["id"], [])
        tum = [a for a in anns if a["category_id"] == TUMOUR_CATEGORY]
        tis = [a for a in anns if a["category_id"] in TISSUE_CATEGORIES]
        n_each = per_image // 2
        kept = 0
        for group, label in ((tum, 1), (tis, 0)):
            if not group:
                continue
            area = np.array([a["area"] for a in group], dtype=float)
            share = np.maximum((area / area.sum() * n_each).astype(int), 1)
            for a, k in zip(group, share):
                pts = polygon_points(a["segmentation"], int(k), rng)
                for cx, cy in pts:
                    x = int(cx) - half
                    y = int(cy) - half
                    if x < 0 or y < 0 or x + crop_px > W or y + crop_px > H:
                        continue
                    p = img[y:y + crop_px, x:x + crop_px]
                    if p.mean() > 225 or p.std() < 10:
                        continue
                    if crop_px != PATCH:
                        p = np.asarray(Image.fromarray(p).resize(
                            (PATCH, PATCH), Image.BILINEAR))
                    recs.append((p, label, s_idx, slide))
                    kept += 1
        if verbose:
            print("  %-22s scanner=%-6s crop=%dpx kept=%d"
                  % (fn, sc, crop_px, kept), flush=True)
        del img
    return recs


def build_folds(out_root, recs, unlab_cap=40000, seed=0):
    rng = np.random.default_rng(seed)
    X = np.asarray([r[0] for r in recs], dtype=np.uint8)
    y = np.asarray([r[1] for r in recs], dtype=np.int64)
    s = np.asarray([r[2] for r in recs], dtype=np.int64)
    slide = np.asarray([r[3] for r in recs])

    # Partition slides into five balanced, disjoint groups so that a held-out
    # fold withholds both a scanner and a set of slides.
    slides = sorted(set(slide.tolist()))
    groups = {slides[int(idx)]: k % 5
              for k, idx in enumerate(rng.permutation(len(slides)))}
    g = np.asarray([groups[v] for v in slide], dtype=np.int64)

    for t in range(5):
        v = (t - 1) % 5
        test = (s == t) & (g == t)
        oodv = (s == v) & (g == v)
        train = (~np.isin(s, [t, v])) & (~np.isin(g, [t, v]))
        # a held-out slice of the training rows, for reference only
        tr_idx = np.where(train)[0]
        rng.shuffle(tr_idx)
        n_iv = int(0.1 * len(tr_idx))
        iv_idx, tr_idx = tr_idx[:n_iv], tr_idx[n_iv:]

        pool = np.where(train)[0]
        Xu = X[pool[rng.permutation(len(pool))[:unlab_cap]]] if len(pool) else \
            np.zeros((0, PATCH, PATCH, 3), np.uint8)

        d = os.path.join(out_root, "fold%d" % t)
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(
            os.path.join(d, "splits.npz"),
            Xtr=X[tr_idx], ytr=y[tr_idx], str=s[tr_idx],
            Xiv=X[iv_idx], yiv=y[iv_idx],
            Xov=X[oodv], yov=y[oodv],
            Xot=X[test], yot=y[test])
        np.savez_compressed(os.path.join(d, "unlabeled.npz"), Xu=Xu)
        json.dump({
            "dataset": "CanineSCC-multiscanner",
            "domain_variable": "slide scanner (same 44 slides on all five)",
            "fold": t,
            "test_scanner": SCANNERS[t], "ood_val_scanner": SCANNERS[v],
            "train_scanners": [SCANNERS[i] for i in range(5) if i not in (t, v)],
            "n_train": int(len(tr_idx)), "n_id_val": int(len(iv_idx)),
            "n_ood_val": int(oodv.sum()), "n_ood_test": int(test.sum()),
            "n_unlabeled": int(len(Xu)),
            "pos_rate_train": float(y[tr_idx].mean()) if len(tr_idx) else None,
            "pos_rate_test": float(y[test].mean()) if test.sum() else None,
            "patch": PATCH,
            "task": "SCC tumour vs non-tumour skin tissue",
            "slide_disjoint": True,
            "model_selection": "held-out scanner (ood_val), as in Camelyon17",
            "scale_normalized": SCALE_NORM,
        }, open(os.path.join(d, "manifest.json"), "w"), indent=1)
        print("fold %d: test=%s(%d) oodval=%s(%d) train=%s n_train=%d n_unlab=%d"
              % (t, SCANNERS[t], test.sum(), SCANNERS[v], oodv.sum(),
                 [SCANNERS[i] for i in range(5) if i not in (t, v)],
                 len(tr_idx), len(Xu)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="./data/canine_scc")
    ap.add_argument("--out", default="./data/canine_cache")
    ap.add_argument("--per_image", type=int, default=420)
    ap.add_argument("--no_scale_norm", action="store_true",
                    help="crop a fixed 64px on every scanner, leaving the "
                         "scanners' 0.22-0.26 um/px difference in the data. "
                         "Isolates the resolution component of scanner shift.")
    a = ap.parse_args()
    global SCALE_NORM
    SCALE_NORM = not a.no_scale_norm
    recs = extract(a.src, os.path.join(a.src, "scc.json"), per_image=a.per_image,
                   scale_norm=SCALE_NORM)
    print("total patches: %d" % len(recs))
    os.makedirs(a.out, exist_ok=True)
    build_folds(a.out, recs)


if __name__ == "__main__":
    main()
