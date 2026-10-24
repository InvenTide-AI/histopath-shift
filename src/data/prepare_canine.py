"""v2 canine cutaneous SCC multi-scanner fold caches (64 and 96 px, same centres).

Follows v1 src/prepare_canine.py:
  * candidate centres sampled inside the COCO polygons with the same RNG
    (default_rng(0), same image/annotation order) -> identical candidates to v1;
  * scale normalisation: per-scanner crop of round(P * s_scanner) px resized
    (PIL bilinear) to P, s from registered polygon areas (cs2 = 1);
  * rejection: crop must lie inside the image at BOTH patch sizes, and the
    64-px-equivalent crop must pass v1's filter (mean <= 225, std >= 10);
  * v2 label precedence (new): tissue polygons (categories 1-6) overlap the
    tumour polygons (no polygon has holes), so v1 labelled some centres that
    lie inside an SCC polygon as non-tumour. Rule: TUMOUR TAKES PRECEDENCE --
    a negative candidate centre inside any tumour-supercategory polygon
    (categories 7-13) of the same image is dropped (count in manifest
    n_neg_dropped_in_tumour); positives are unaffected;
  * exact duplicate centres (same image, same integer centre, which polygon
    sampling can produce) are removed (first occurrence kept);
  * slide groups: v1 build_folds (default_rng(0).permutation of the 44 sorted
    slides, k mod 5); fold t: test = scanner t & slide-group t, ood_val =
    scanner v=(t-1) mod 5 & slide-group v, train = other scanners & other groups.
v2 changes (D5, D6, D11):
  * id-val = all training-scanner patches of a held-out 15 % of the training
    slides (slide-disjoint from train, for every scanner);
  * training-domain labels remapped to 0..2 (contiguous);
  * unlabelled pool (new, label-free, same scheme as MIDOG): random tissue crops
    from the non-id-val training slides of the training scanners. Per image up
    to --n_unlab centres drawn with an image-specific RNG
    (default_rng([seed, image_id, 505])) inside the image with a margin of
    ceil(96*s)+1 px (every size fits), accepted iff the 64-px-equivalent crop
    passes v1's MIDOG tissue filter (mean < 220, std > 12), no duplicate
    centres, and not exactly on a labelled centre of that image (which would
    reproduce an Xtr patch); random subsample to 40000 per fold. (v1 / first v2 build used the
    labelled training patches themselves, i.e. Xu == Xtr without labels.)
  * scale factors: computed from the annotation areas of all slides (one
    extraction pass); for every fold they are recomputed from the slides
    outside the fold's test and val slide groups, and the crop sizes
    round(P*s) are asserted to be identical (so the cache is exactly what the
    fold-restricted factors would give); both are recorded in the manifest.
Group id = slide number (scc_XX -> XX); the same slide on different scanners
shares the group id.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from matplotlib.path import Path as MplPath
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
SCANNERS = ["cs2", "p1000", "nz20", "nz210", "gt450"]
TUMOUR_CATEGORY = 11
TISSUE_CATEGORIES = {1, 2, 3, 4, 5, 6}
TUMOUR_SUPER = {7, 8, 9, 10, 11, 12, 13}   # every tumour-supercategory polygon wins over tissue
SIZES = (64, 96)


def scale_factors(ann, exclude_slides=()) -> dict:
    meta = {im["id"]: im["file_name"][:-4].rsplit("_", 1) for im in ann["images"]}
    tot = collections.defaultdict(float)
    for a in ann["annotations"]:
        slide, sc = meta[a["image_id"]]
        if slide in exclude_slides:
            continue
        tot[(slide, sc)] += a["area"]
    slides = sorted({k[0] for k in tot})
    out = {}
    for sc in SCANNERS:
        r = [np.sqrt(tot[(s, sc)] / tot[(s, "cs2")]) for s in slides
             if (s, sc) in tot and tot.get((s, "cs2"), 0) > 0]
        out[sc] = float(np.median(r)) if r else 1.0
    return out


def seg_path(seg):
    if isinstance(seg, str):
        seg = json.loads(seg)
    if seg and isinstance(seg[0], list):
        assert len(seg) == 1, "multi-ring segmentation not supported"
        seg = seg[0]
    xy = np.asarray(seg, dtype=float).reshape(-1, 2)
    return MplPath(xy) if len(xy) >= 3 else None


def drop_neg_in_tumour(ann, jobs):
    """Tumour precedence: drop label-0 centres inside any tumour polygon of the same image."""
    by_fn = {im["file_name"]: im["id"] for im in ann["images"]}
    tum = collections.defaultdict(list)
    for a in ann["annotations"]:
        if a["category_id"] in TUMOUR_SUPER:
            pth = seg_path(a["segmentation"])
            if pth is not None:
                tum[a["image_id"]].append(pth)
    out, n_drop = [], collections.Counter()
    for fn, pts, lab in jobs:
        bad = np.zeros(len(pts), bool)
        neg = lab == 0
        for pth in tum.get(by_fn[fn], []):
            if neg.any():
                bad |= neg & pth.contains_points(pts)
        n_drop[fn[:-4].rsplit("_", 1)[1]] += int(bad.sum())
        out.append((fn, pts[~bad], lab[~bad]))
    return out, dict(n_drop)


def polygon_points(seg, n, rng):
    """Verbatim v1."""
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


def candidates(ann, per_image):
    """Same RNG consumption order as v1 extract(): per image, tumour group then tissue group."""
    by_image = collections.defaultdict(list)
    for a in ann["annotations"]:
        by_image[a["image_id"]].append(a)
    rng = np.random.default_rng(0)
    jobs = []
    for im_rec in ann["images"]:
        anns = by_image.get(im_rec["id"], [])
        tum = [a for a in anns if a["category_id"] == TUMOUR_CATEGORY]
        tis = [a for a in anns if a["category_id"] in TISSUE_CATEGORIES]
        n_each = per_image // 2
        pts_all, lab_all = [], []
        for group, label in ((tum, 1), (tis, 0)):
            if not group:
                continue
            area = np.array([a["area"] for a in group], dtype=float)
            share = np.maximum((area / area.sum() * n_each).astype(int), 1)
            for a, k in zip(group, share):
                pts = polygon_points(a["segmentation"], int(k), rng)
                pts_all.append(pts); lab_all.append(np.full(len(pts), label))
        pts = np.concatenate(pts_all) if pts_all else np.zeros((0, 2))
        lab = np.concatenate(lab_all) if lab_all else np.zeros(0, int)
        jobs.append((im_rec["file_name"], pts, lab))
    return jobs


def crop_one(img, cx, cy, P, s):
    cp = int(round(P * s))
    half = cp // 2
    x, y = int(cx) - half, int(cy) - half
    H, W = img.shape[:2]
    if x < 0 or y < 0 or x + cp > W or y + cp > H:
        return None
    p = img[y:y + cp, x:x + cp]
    return p, cp


def unlabelled_crops(img, s, n, rng_key, tries_per):
    """Random tissue crops (label-free), MIDOG scheme; returns {P: (k,P,P,3)}, centres, tries."""
    H, W = img.shape[:2]
    rng = np.random.default_rng(rng_key)
    m = int(np.ceil(max(SIZES) * s)) + 1
    out = {P: [] for P in SIZES}
    xy, seen, tries = [], set(), 0
    while len(xy) < n and tries < n * tries_per:
        tries += 1
        cx = int(rng.integers(m, W - m)); cy = int(rng.integers(m, H - m))
        if (cx, cy) in seen:
            continue
        c64 = crop_one(img, cx, cy, 64, s)
        p64 = c64[0] if c64[1] == 64 else np.asarray(Image.fromarray(c64[0]).resize((64, 64), Image.BILINEAR))
        if not C.is_tissue(p64):
            continue
        seen.add((cx, cy))
        for P in SIZES:
            p, cp = crop_one(img, cx, cy, P, s)
            out[P].append(p if cp == P else np.asarray(Image.fromarray(p).resize((P, P), Image.BILINEAR)))
        xy.append((cx, cy))
    res = {P: (np.stack(out[P]) if out[P] else np.zeros((0, P, P, 3), np.uint8)) for P in SIZES}
    return res, np.asarray(xy, np.int64).reshape(-1, 2), tries


def work(args):
    src, fn, pts, lab, s, n_unlab, rng_key, tries_per = args
    img = np.asarray(Image.open(os.path.join(src, fn)).convert("RGB"))
    out = {P: [] for P in SIZES}
    keep_lab, keep_xy = [], []
    n_bound = n_filter = 0
    for (cx, cy), l in zip(pts, lab):
        crops = {P: crop_one(img, cx, cy, P, s) for P in SIZES}
        if any(c is None for c in crops.values()):
            n_bound += 1
            continue
        p64 = crops[64][0]
        if p64.mean() > 225 or p64.std() < 10:   # v1 filter, on the 64-px crop
            n_filter += 1
            continue
        for P in SIZES:
            p, cp = crops[P]
            if cp != P:
                p = np.asarray(Image.fromarray(p).resize((P, P), Image.BILINEAR))
            out[P].append(p)
        keep_lab.append(l); keep_xy.append((int(cx), int(cy)))
    res = {P: (np.stack(out[P]) if out[P] else np.zeros((0, P, P, 3), np.uint8)) for P in SIZES}
    unl, uxy, tries = unlabelled_crops(img, s, n_unlab, rng_key, tries_per)
    return (fn, res, np.asarray(keep_lab, np.int64), np.asarray(keep_xy, np.int64).reshape(-1, 2), n_bound, n_filter,
            unl, uxy, tries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(os.environ.get("HISTOPATH_DATA", "data"), "canine_scc"))
    ap.add_argument("--v2", default=C.V2)
    ap.add_argument("--patch", type=int, nargs="+", default=[64, 96], choices=[64, 96])
    ap.add_argument("--per_image", type=int, default=420)
    ap.add_argument("--iv_frac", type=float, default=0.15)
    ap.add_argument("--cap_tr", type=int, default=30000)
    ap.add_argument("--cap_u", type=int, default=40000)
    ap.add_argument("--n_unlab", type=int, default=800, help="max random tissue crops per image")
    ap.add_argument("--tries_per", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    ann = json.load(open(os.path.join(a.src, "scc.json")))
    scales = scale_factors(ann)
    print("scale factors:", {k: round(v, 4) for k, v in scales.items()}, flush=True)
    jobs = candidates(ann, a.per_image)
    n_cand = sum(len(j[1]) for j in jobs)
    jobs, n_neg_drop = drop_neg_in_tumour(ann, jobs)
    print("%d candidate centres; tumour precedence dropped %d negative centres inside tumour polygons %s" % (
        n_cand, sum(n_neg_drop.values()), n_neg_drop), flush=True)
    img_id = {im["file_name"]: im["id"] for im in ann["images"]}
    args = []
    for fn, pts, lab in jobs:
        sc = fn[:-4].rsplit("_", 1)[1]
        assert os.path.exists(os.path.join(a.src, fn)), fn
        args.append((a.src, fn, pts, lab, scales[sc], a.n_unlab, [a.seed, int(img_id[fn]), 505], a.tries_per))
    print("%d images, %d candidate centres after precedence rule" % (len(args), sum(len(j[1]) for j in jobs)),
          flush=True)
    X = {P: [] for P in SIZES}
    XU = {P: [] for P in SIZES}
    Y, S, SL, XY, IMG = [], [], [], [], []
    US, USL, UXY, UIMG = [], [], [], []
    nb = nf = 0
    u_stats = {}
    with Pool(a.workers) as pool:
        for fn, res, lab, xy, b, f, unl, uxy, tries in pool.imap(work, args):
            slide, sc = fn[:-4].rsplit("_", 1)
            for P in SIZES:
                X[P].append(res[P]); XU[P].append(unl[P])
            Y.append(lab); S += [sc] * len(lab); SL += [int(slide.split("_")[1])] * len(lab)
            XY.append(xy); IMG += [fn] * len(lab)
            US += [sc] * len(uxy); USL += [int(slide.split("_")[1])] * len(uxy); UXY.append(uxy)
            UIMG += [fn] * len(uxy)
            u_stats[fn] = (len(uxy), tries)
            nb += b; nf += f
    X = {P: np.concatenate(X[P]) for P in SIZES}
    XU = {P: np.concatenate(XU[P]) for P in SIZES}
    y = np.concatenate(Y); s = np.asarray(S); slide = np.asarray(SL, np.int64)
    XY = np.concatenate(XY); IMG = np.asarray(IMG)
    us = np.asarray(US); uslide = np.asarray(USL, np.int64); UXY = np.concatenate(UXY); UIMG = np.asarray(UIMG)
    print("kept %d patches (rejected: %d boundary, %d white/flat) %.0fs" % (len(y), nb, nf, time.time() - t0),
          flush=True)
    n_short = sum(1 for k, _ in u_stats.values() if k < a.n_unlab)
    print("unlabelled: %d crops from %d images (%d images below %d; acceptance %.3f)" % (
        len(us), len(u_stats), n_short, a.n_unlab,
        sum(k for k, _ in u_stats.values()) / max(1, sum(t_ for _, t_ in u_stats.values()))), flush=True)
    # exact duplicate centres (same image, same integer centre) -> keep first occurrence
    key = np.char.add(np.char.add(IMG.astype(str), "|"), np.char.add(XY[:, 0].astype(str), np.char.add(",", XY[:, 1].astype(str))))
    _, first = np.unique(key, return_index=True)
    keep = np.sort(first)
    n_dup = len(y) - len(keep)
    _, inv = np.unique(key, return_inverse=True)
    n_dup_conflict = int(sum(len(set(y[inv == k].tolist())) > 1 for k in np.unique(inv[np.bincount(inv)[inv] > 1])))
    X = {P: X[P][keep] for P in SIZES}
    y, s, slide, XY, IMG = y[keep], s[keep], slide[keep], XY[keep], IMG[keep]
    print("removed %d exact duplicate centres (%d with conflicting labels); %d patches" % (
        n_dup, n_dup_conflict, len(y)), flush=True)
    # an unlabelled crop at exactly a labelled centre (same image) would be an Xtr patch -> drop it
    lab_keys = set(zip(IMG.tolist(), XY[:, 0].tolist(), XY[:, 1].tolist()))
    u_ok = np.array([k not in lab_keys for k in zip(UIMG.tolist(), UXY[:, 0].tolist(), UXY[:, 1].tolist())], bool)
    n_u_coincide = int((~u_ok).sum())
    XU = {P: XU[P][u_ok] for P in SIZES}
    us, uslide, UXY, UIMG = us[u_ok], uslide[u_ok], UXY[u_ok], UIMG[u_ok]
    print("dropped %d unlabelled crops centred exactly on a labelled centre; %d unlabelled crops" % (
        n_u_coincide, len(us)), flush=True)

    # slide groups exactly as v1 build_folds (rng(0) first call)
    rng0 = np.random.default_rng(0)
    slides = sorted(set(("scc_%02d" % v) for v in slide.tolist()))
    assert slides == sorted({im["file_name"][:-4].rsplit("_", 1)[0] for im in ann["images"]})
    groups = {int(slides[int(idx)].split("_")[1]): k % 5 for k, idx in enumerate(rng0.permutation(len(slides)))}
    crop_px = {sc: {P: int(round(P * scales[sc])) for P in SIZES} for sc in SCANNERS}
    g5 = np.asarray([groups[v] for v in slide], np.int64)
    sidx = np.asarray([SCANNERS.index(v) for v in s])
    rows_all = np.arange(len(y))

    usidx = np.asarray([SCANNERS.index(v) for v in us])
    for t in range(5):
        v = (t - 1) % 5
        excl = ["scc_%02d" % sl for sl, gg in groups.items() if gg in (t, v)]
        scales_fold = scale_factors(ann, exclude_slides=set(excl))
        crop_fold = {sc: {P: int(round(P * scales_fold[sc])) for P in SIZES} for sc in SCANNERS}
        assert crop_fold == crop_px, ("fold-restricted scale factors change the crop size", t, crop_fold, crop_px)
        train_sc = [i for i in range(5) if i not in (t, v)]
        test = (sidx == t) & (g5 == t)
        oodv = (sidx == v) & (g5 == v)
        train = np.isin(sidx, train_sc) & ~np.isin(g5, [t, v])
        tr_slides = sorted(set(slide[train].tolist()))
        rng = np.random.default_rng([a.seed, t, 101])
        n_iv = max(1, int(round(a.iv_frac * len(tr_slides))))
        iv_slides = sorted(rng.choice(tr_slides, size=n_iv, replace=False).tolist())
        iv = train & np.isin(slide, iv_slides)
        tr = train & ~np.isin(slide, iv_slides)
        tr_idx = rows_all[tr]
        strata = sidx[tr_idx] * 2 + y[tr_idx]
        tr_idx = C.stratified_cap(tr_idx, strata, a.cap_tr, rng)
        iv_idx = rows_all[iv]
        # label-free random tissue crops of the non-id-val training slides, training scanners
        u_pool = np.where(np.isin(usidx, train_sc) & np.isin(uslide, tr_slides) & ~np.isin(uslide, iv_slides))[0]
        assert not np.isin(uslide[u_pool], iv_slides).any() and np.isin(usidx[u_pool], train_sc).all()
        u_idx = np.sort(rng.choice(u_pool, size=min(a.cap_u, len(u_pool)), replace=False))
        ov_idx, ot_idx = rows_all[oodv], rows_all[test]
        for P in a.patch:
            XP = X[P]
            def part(ix):
                return dict(X=XP[ix], y=y[ix], d=s[ix], g=slide[ix], rows=ix)
            parts = {"tr": part(tr_idx), "iv": part(iv_idx), "ov": part(ov_idx), "ot": part(ot_idx),
                     "u": dict(X=XU[P][u_idx], y=None, d=us[u_idx], g=uslide[u_idx], rows=u_idx)}
            out = os.path.join(a.v2, "cache%d" % P, "canine", "fold%d" % t)
            man = C.write_fold(
                out, "canine", t, P, a.seed, SCANNERS[t], SCANNERS[v], [SCANNERS[i] for i in train_sc],
                parts, group_kind="slide", source=a.src,
                notes=[
                    "candidate centres identical to v1 prepare_canine (rng(0)); patch kept iff inside image at 64 and 96 px and v1 filter passes on the 64-px crop",
                    "label precedence: tumour wins -- negative centres inside any tumour polygon dropped (n_neg_dropped_in_tumour, per scanner); exact duplicate centres removed (n_dup_centres_removed)",
                    "scale-normalised: crop round(P*s) resized bilinear to P; s per scanner in manifest (scale_factors: all-slide annotation areas; scale_factors_fold: excluding test+val slide groups; identical crop_px asserted)",
                    "slide groups identical to v1; test = scanner t & group t; ood_val = scanner (t-1)%5 & group (t-1)%5",
                    "D5: id-val = training-scanner patches of %d held-out training slides (%.0f%% of training slides); slide-disjoint from train" % (n_iv, 100 * a.iv_frac),
                    "D11: Xu = label-free random tissue crops (MIDOG scheme: per-image RNG, margin ceil(96*s)+1, v1 MIDOG tissue filter on the 64-px crop, no duplicate centres) from the non-id-val training slides of the training scanners only, random subsample to cap_u; rows_u = positions in the all-image crop pool; NOT the Xtr patches (changed from v1)",
                    "D6: dtr/div are 0..2 over train_domains",
                ],
                extra={"scale_factors": scales, "scale_factors_fold": scales_fold, "crop_px": crop_px,
                       "scale_normalized": True, "iv_slides": iv_slides,
                       "n_neg_dropped_in_tumour": n_neg_drop, "label_precedence": "tumour > tissue",
                       "n_dup_centres_removed": int(n_dup), "n_dup_centres_conflicting_labels": n_dup_conflict,
                       "u_pool_size": int(len(u_pool)), "n_unlab_per_image": a.n_unlab,
                       "n_u_dropped_on_labelled_centre": n_u_coincide,
                       "u_scheme": "random tissue crops, MIDOG filter",
                       "slide_group_of_test": t, "slide_group_of_val": v,
                       "slides_train": sorted(set(slide[tr_idx].tolist())),
                       "slides_ov": sorted(set(slide[ov_idx].tolist())),
                       "slides_ot": sorted(set(slide[ot_idx].tolist())),
                       "n_rejected_boundary": nb, "n_rejected_filter": nf, "per_image": a.per_image,
                       "tr_capped": bool(tr.sum() > a.cap_tr), "code_sha": C.code_sha(__file__)})
            print("fold%d P=%d test=%s val=%s tr=%d iv=%d ov=%d ot=%d u=%d pos=%.3f/%.3f/%.3f/%.3f iv_slides=%s" % (
                t, P, SCANNERS[t], SCANNERS[v], man["n_tr"], man["n_iv"], man["n_ov"], man["n_ot"], man["n_u"],
                man["pos_rate_tr"], man["pos_rate_iv"], man["pos_rate_ov"], man["pos_rate_ot"], iv_slides), flush=True)
    print("done %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
