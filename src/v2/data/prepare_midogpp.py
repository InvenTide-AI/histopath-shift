"""v2 MIDOG++ fold caches (64 and 96 px, same centres), K = 7 tumour-type domains.

Data: MIDOG++ (Aubreville et al., Sci Data 10:484, 2023; figshare collection
6615571), 503 annotated ROI images (2 mm^2 each), annotations MIDOG++.json.
Task: mitotic figure (category 1) vs non-mitotic look-alike (category 2), the
consensus label `category_id`.
Domains: `tumor_type`, canonical order = alphabetical (DOMAINS below).
Fold t: test domain t, val domain (t-1) mod 7, train the other 5.
  * id-val: all objects of 15 % of the images of each training domain
    (image-disjoint), fixed per domain (default_rng([seed, domain, 202]));
  * Xtr capped at 30000 and Xov at 12000 by proportional stratified
    subsampling over (domain, label) when larger (recorded in the manifest);
  * unlabelled: up to 300 random tissue crops per non-id-val training image
    (v1 MIDOG filter), random subsample to 40000;
  * training-domain labels 0..4 (contiguous); group id = MIDOG++ image id.
Scale normalisation: per Table 1 of Aubreville et al. (2023), the domains were
scanned with the Hamamatsu NanoZoomer XR / S360 (~0.23 um/px), the Aperio
ScanScope CS2 (listed as "Leica ScanScope CS2"; 0.25 um/px) or the 3DHistech
Pannoramic Scan II (0.25 um/px); the figshare TIFF tags give the exact
per-image value (image_meta.json), which is what the code uses. Every patch is cropped at
round(P * 0.25 / mpp_image) px and resized (bilinear) to P, so that every patch
covers P * 0.25 um, as in the canine prep.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import midog_lib as L  # noqa: E402

DOMAINS = sorted([
    "canine cutaneous mast cell tumor", "canine lung cancer", "canine lymphosarcoma",
    "canine soft tissue sarcoma", "human breast cancer", "human melanoma", "human neuroendocrine tumor"])
TARGET_MPP = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(os.environ.get("HISTOPATH_DATA", "data"), "midogpp"))
    ap.add_argument("--v2", default=C.V2)
    ap.add_argument("--patch", type=int, nargs="+", default=[64, 96], choices=[64, 96])
    ap.add_argument("--iv_frac", type=float, default=0.15)
    ap.add_argument("--n_unlab", type=int, default=300)
    ap.add_argument("--cap_u", type=int, default=40000)
    ap.add_argument("--cap_tr", type=int, default=30000)
    ap.add_argument("--cap_ov", type=int, default=12000)
    ap.add_argument("--no_scale_norm", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    ann = json.load(open(os.path.join(a.src, "MIDOG++.json")))
    meta = {m["file_name"]: m for m in json.load(open(os.path.join(a.src, "image_meta.json")))}
    ims = {im["id"]: im for im in ann["images"]}
    objs = collections.defaultdict(list)
    for x in ann["annotations"]:
        x0, y0, x1, y1 = x["bbox"]
        assert x["category_id"] in (1, 2)
        objs[x["image_id"]].append(((x0 + x1) / 2.0, (y0 + y1) / 2.0, 1 if x["category_id"] == 1 else 0, x["id"]))
    ids = sorted(i for i in ims if objs.get(i))
    assert set(ims[i]["tumor_type"] for i in ids) == set(DOMAINS)
    dom = {i: DOMAINS.index(ims[i]["tumor_type"]) for i in ids}
    jobs, mpp = [], {}
    for i in ids:
        fn = ims[i]["file_name"]
        r = meta[fn]["resolution"]
        assert r and r["mpp_x"] and abs(r["mpp_x"] / r["mpp_y"] - 1) < 1e-2, (fn, r)  # 3DHistech tags: x/y differ by 0.4%
        mpp[i] = (r["mpp_x"] * r["mpp_y"]) ** 0.5   # geometric mean; square crops
        s = 1.0 if a.no_scale_norm else TARGET_MPP / mpp[i]
        jobs.append(dict(path=os.path.join(a.src, fn), num=i, objs=objs[i], s=s, seed=a.seed, n_unlab=a.n_unlab))
    res = L.run_all(jobs, a.workers)
    print("extracted %d labelled, %d unlabelled from %d images in %.0fs" % (
        sum(len(r["y"]) for r in res.values()), sum(len(r["ucxy"]) for r in res.values()), len(res),
        time.time() - t0), flush=True)
    domain_facts = {}
    for k, dname in enumerate(DOMAINS):
        di = [i for i in ids if dom[i] == k]
        yy = np.concatenate([res[i]["y"] for i in di])
        domain_facts[dname] = {"n_images": len(di), "n_mitotic": int(yy.sum()), "n_imposter": int((1 - yy).sum()),
                               "mpp_values": sorted({round(mpp[i], 4) for i in di}),
                               "crop_px_64": sorted({res[i]["crop_px"][64] for i in di}),
                               "crop_px_96": sorted({res[i]["crop_px"][96] for i in di})}
    print(json.dumps(domain_facts, indent=1), flush=True)

    iv_imgs = {}
    for k in range(7):
        di = [i for i in ids if dom[i] == k]
        rng = np.random.default_rng([a.seed, k, 202])
        iv_imgs[k] = sorted(rng.choice(di, size=int(round(a.iv_frac * len(di))), replace=False).tolist())

    for t in range(7):
        v = (t - 1) % 7
        train_d = [k for k in range(7) if k not in (t, v)]
        tr_imgs = [i for i in ids if dom[i] in train_d and i not in iv_imgs[dom[i]]]
        iv_list = [i for k in train_d for i in iv_imgs[k]]
        ov_imgs = [i for i in ids if dom[i] == v]
        ot_imgs = [i for i in ids if dom[i] == t]
        rng = np.random.default_rng([a.seed, t, 303])
        caps = {}
        sel = {}
        for name, imgs, cap in (("tr", tr_imgs, a.cap_tr), ("iv", iv_list, None), ("ov", ov_imgs, a.cap_ov),
                                ("ot", ot_imgs, None)):
            y = np.concatenate([res[i]["y"] for i in imgs])
            img = np.concatenate([np.full(len(res[i]["y"]), i) for i in imgs])
            idx = np.arange(len(y))
            strata = np.array([dom[i] for i in img]) * 2 + y
            keep = C.stratified_cap(idx, strata, cap, rng) if cap else idx
            caps[name] = {"before": int(len(idx)), "after": int(len(keep)), "capped": bool(len(keep) < len(idx))}
            sel[name] = (imgs, keep)
        nu = sum(len(res[i]["ucxy"]) for i in tr_imgs)
        ukeep = np.sort(rng.choice(nu, size=min(a.cap_u, nu), replace=False))
        caps["u"] = {"before": int(nu), "after": int(len(ukeep)), "capped": bool(len(ukeep) < nu)}
        for P in a.patch:
            parts = {}
            for name, (imgs, keep) in sel.items():
                X, y, img = L.assemble(res, imgs, P, "lab")
                aid = np.concatenate([res[i]["ann_id"] for i in imgs])
                pad = L.assemble_pad(res, imgs, P)
                parts[name] = dict(X=X[keep], y=y[keep], d=np.array([DOMAINS[dom[i]] for i in img[keep]]),
                                   g=img[keep], rows=aid[keep], pad=pad[keep])
            Xu, _, imgu = L.assemble(res, tr_imgs, P, "unl")
            parts["u"] = dict(X=Xu[ukeep], y=None, d=np.array([DOMAINS[dom[i]] for i in imgu[ukeep]]),
                              g=imgu[ukeep], rows=ukeep)
            out = os.path.join(a.v2, "cache%d" % P, "midogpp", "fold%d" % t)
            man = C.write_fold(
                out, "midogpp", t, P, a.seed, DOMAINS[t], DOMAINS[v], [DOMAINS[k] for k in train_d], parts,
                group_kind="image", source=a.src + " (MIDOG++.json, figshare collection 6615571)",
                notes=[
                    "domain = tumor_type; canonical order alphabetical (domain_order)",
                    "fold t: test domain t, val domain (t-1)%7, train the other 5",
                    "id-val image-disjoint: 15% of images per training domain, fixed per domain",
                    "caps: Xtr 30000, Xov 12000 (proportional stratified over domain x label), Xu 40000 random; see caps",
                    "scale-normalised to %.2f um/px using per-image figshare TIFF resolution tags: crop round(P*0.25/mpp) px, bilinear resize to P" % TARGET_MPP
                    if not a.no_scale_norm else "no scale normalisation",
                    "labelled patch centred on the bbox centre at both 64 and 96 px (same centre); objects whose crop leaves the image are cut from a reflect-padded image (no clamping); per-row padding px in rows.npz pad_<split>, counts in n_edge_padded_<split>",
                    "unlabelled: up to 300 tissue crops per non-id-val training image (v1 MIDOG filter on the 64-px crop, no duplicate centres)",
                    "rows_* in rows.npz = MIDOG++.json annotation ids (labelled) / pool positions (Xu)",
                ],
                extra={"domain_order": DOMAINS, "caps": caps,
                       "iv_images": {DOMAINS[k]: iv_imgs[k] for k in train_d},
                       "n_images": {"tr": len(tr_imgs), "iv": len(iv_list), "ov": len(ov_imgs), "ot": len(ot_imgs)},
                       "scale_normalized": not a.no_scale_norm, "target_mpp": TARGET_MPP,
                       "domain_facts": domain_facts,
                       "category_map": {"1 mitotic figure": 1, "2 not mitotic figure": 0},
                       "code_sha": C.code_sha(__file__)})
            print("fold%d P=%d test=%s val=%s tr=%d iv=%d ov=%d ot=%d u=%d pos=%.3f/%.3f/%.3f/%.3f" % (
                t, P, DOMAINS[t], DOMAINS[v], man["n_tr"], man["n_iv"], man["n_ov"], man["n_ot"], man["n_u"],
                man["pos_rate_tr"], man["pos_rate_iv"], man["pos_rate_ov"], man["pos_rate_ot"]), flush=True)
    print("done %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
