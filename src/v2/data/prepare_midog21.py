"""v2 MIDOG 2021 fold caches (64 and 96 px, same centres).

Follows v1 src/prepare_midog.py: images 001-150 (three annotated scanners,
one lab), positive = category 1 (mitotic figure), negative = category 2
(non-mitotic look-alike); patch centred on the bbox centre (near-border
objects cut from a reflect-padded image so they stay centred -- v1 clamped
the crop instead; see midog_lib); unlabelled = up to 300 random tissue crops
per image (v1 filter, no duplicate centres).
Scale: by default NO scale normalisation (as v1 and the MIDOG 2021 challenge):
the acquisition shift therefore includes the scanners' ~11 % magnification
difference (XR/S360 ~0.227-0.230 um/px vs CS2 0.2533 um/px per the figshare
TIFF tags, recorded in the manifest), whereas canine and MIDOG++ are
normalised to 0.25 um/px. --scale_norm builds the normalised variant
(crop round(P*0.25/mpp) px, bilinear to P, as MIDOG++) under cohort name
`midog21sn` for a sensitivity analysis.
v2 changes (D5, D6, D11):
  * NO ood-val split at all (Xov/yov absent): K=3, fold t tests scanner t and
    trains on the other two;
  * id-val = all objects of 15 % of the images of each training scanner
    (image-disjoint); the held-out images are fixed per scanner
    (default_rng([seed, scanner])), so a scanner's id-val images are the same in
    both folds where it is a training scanner;
  * training-domain labels 0..1 (contiguous);
  * unlabelled pool from the non-id-val images of the training scanners only.
Group id = image number (1..150).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import midog_lib as L  # noqa: E402

SCANNERS = ["HamamatsuXR", "HamamatsuS360", "AperioCS2"]


def scanner_of(n):
    return (n - 1) // 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(os.environ.get("HISTOPATH_DATA", "data"), "midog"))
    ap.add_argument("--v2", default=C.V2)
    ap.add_argument("--patch", type=int, nargs="+", default=[64, 96], choices=[64, 96])
    ap.add_argument("--iv_frac", type=float, default=0.15)
    ap.add_argument("--n_unlab", type=int, default=300)
    ap.add_argument("--cap_u", type=int, default=40000)
    ap.add_argument("--cap_tr", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scale_norm", action="store_true",
                    help="normalise to 0.25 um/px with the per-image figshare TIFF tags; writes cohort midog21sn")
    a = ap.parse_args()
    cohort = "midog21sn" if a.scale_norm else "midog21"
    TARGET_MPP = 0.25
    pp = os.path.join(os.environ.get("HISTOPATH_DATA", "data"), "midogpp/image_meta.json")
    mpp_img = {}
    if os.path.exists(pp):
        for m in json.load(open(pp)):
            r = m["resolution"] or {}
            if r.get("mpp_x") and r.get("mpp_y"):
                mpp_img[m["file_name"]] = (r["mpp_x"] * r["mpp_y"]) ** 0.5
    if a.scale_norm:
        assert all(("%03d.tiff" % n) in mpp_img for n in range(1, 151)), "need midogpp/image_meta.json"
    t0 = time.time()
    ann = json.load(open(os.path.join(a.src, "MIDOG.json")))
    num_of = {im["id"]: int(im["file_name"].split(".")[0]) for im in ann["images"]}
    objs = {n: [] for n in range(1, 151)}
    for x in ann["annotations"]:
        n = num_of[x["image_id"]]
        if n > 150:
            continue
        x0, y0, x1, y1 = x["bbox"]
        objs[n].append(((x0 + x1) / 2.0, (y0 + y1) / 2.0, 1 if x["category_id"] == 1 else 0, x["id"]))
    jobs = [dict(path=os.path.join(a.src, "%03d.tiff" % n), num=n, objs=objs[n],
                 s=(TARGET_MPP / mpp_img["%03d.tiff" % n]) if a.scale_norm else 1.0,
                 seed=a.seed, n_unlab=a.n_unlab) for n in range(1, 151)]
    res = L.run_all(jobs, a.workers)
    print("extracted %d labelled, %d unlabelled in %.0fs" % (
        sum(len(r["y"]) for r in res.values()), sum(len(r["ucxy"]) for r in res.values()), time.time() - t0),
        flush=True)

    # fixed per-scanner id-val image hold-out
    iv_imgs = {}
    for s in range(3):
        imgs = [n for n in range(1, 151) if scanner_of(n) == s]
        rng = np.random.default_rng([a.seed, s, 202])
        k = int(round(a.iv_frac * len(imgs)))
        iv_imgs[s] = sorted(rng.choice(imgs, size=k, replace=False).tolist())
    res_mpp = None
    if os.path.exists(pp):
        res_mpp = {m["file_name"]: (m["resolution"] or {}).get("mpp_x") for m in json.load(open(pp))}
        mpp_by_scanner = {SCANNERS[s]: sorted({round(res_mpp.get("%03d.tiff" % n) or -1, 4)
                                                for n in range(1, 151) if scanner_of(n) == s}) for s in range(3)}
    else:
        mpp_by_scanner = None

    for t in range(3):
        train_sc = [s for s in range(3) if s != t]
        tr_imgs = [n for n in range(1, 151) if scanner_of(n) in train_sc and n not in iv_imgs[scanner_of(n)]]
        iv_list = [n for s in train_sc for n in iv_imgs[s]]
        ot_imgs = [n for n in range(1, 151) if scanner_of(n) == t]
        for P in a.patch:
            parts = {}
            for name, imgs in (("tr", tr_imgs), ("iv", iv_list), ("ot", ot_imgs)):
                X, y, img = L.assemble(res, imgs, P, "lab")
                d = np.array([SCANNERS[scanner_of(n)] for n in img])
                parts[name] = dict(X=X, y=y, d=d, g=img, rows=np.concatenate(
                    [res[n]["ann_id"] for n in imgs]) if imgs else np.zeros(0, np.int64),
                    pad=L.assemble_pad(res, imgs, P))
            Xu, _, imgu = L.assemble(res, tr_imgs, P, "unl")
            rng = np.random.default_rng([a.seed, t, 303])
            keep = np.sort(rng.choice(len(Xu), size=min(a.cap_u, len(Xu)), replace=False))
            parts["u"] = dict(X=Xu[keep], y=None, d=np.array([SCANNERS[scanner_of(n)] for n in imgu[keep]]),
                              g=imgu[keep], rows=keep)
            assert len(parts["tr"]["X"]) <= a.cap_tr
            out = os.path.join(a.v2, "cache%d" % P, cohort, "fold%d" % t)
            man = C.write_fold(
                out, cohort, t, P, a.seed, SCANNERS[t], None, [SCANNERS[s] for s in train_sc], parts,
                group_kind="image", source=a.src + " (MIDOG.json, images 001-150)",
                notes=[
                    "no ood_val split (K=3): Xov/yov absent by design (D5)",
                    "id-val image-disjoint: %d images per scanner held out (fixed per scanner)" % len(iv_imgs[0]),
                    "labelled patch centred on the bbox centre at both 64 and 96 px (same centre); objects whose crop leaves the image are cut from a reflect-padded image (v1 clamped instead); per-row padding px in rows.npz pad_<split>, counts in n_edge_padded_<split>",
                    "unlabelled: up to %d random tissue crops per non-id-val training image (v1 filter on the 64-px crop, no duplicate centres), cap %d" % (a.n_unlab, a.cap_u),
                    ("scale-normalised to 0.25 um/px with per-image figshare TIFF tags: crop round(P*0.25/mpp) px, bilinear resize to P (sensitivity variant)"
                     if a.scale_norm else
                     "NOT scale-normalised (as v1 / MIDOG 2021 challenge): the scanner shift includes the ~11% magnification difference (XR/S360 ~0.227-0.230 vs CS2 0.2533 um/px, see mpp_by_scanner); canine and MIDOG++ ARE normalised to 0.25 um/px; the normalised variant is cohort midog21sn"),
                    "rows_* in rows.npz = MIDOG.json annotation ids (labelled) / pool positions (Xu)",
                ],
                extra={"iv_images": {SCANNERS[s]: iv_imgs[s] for s in train_sc},
                       "n_images": {"tr": len(tr_imgs), "iv": len(iv_list), "ot": len(ot_imgs)},
                       "mpp_by_scanner_figshare_tiff_tags": mpp_by_scanner, "scale_normalized": bool(a.scale_norm),
                       "crop_px": {P_: sorted({res[n]["crop_px"][P_] for n in range(1, 151)}) for P_ in L.SIZES},
                       "category_map": {"1 mitotic figure": 1, "2 not mitotic figure": 0},
                       "code_sha": C.code_sha(__file__)})
            print("fold%d P=%d test=%s tr=%d iv=%d ot=%d u=%d pos=%.3f/%.3f/%.3f" % (
                t, P, SCANNERS[t], man["n_tr"], man["n_iv"], man["n_ot"], man["n_u"],
                man["pos_rate_tr"], man["pos_rate_iv"], man["pos_rate_ot"]), flush=True)
    print("done %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
