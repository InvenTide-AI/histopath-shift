"""Verify every v2 fold cache against CONTRACTS.md section A.

Per fold and patch size:
  * required files present (Xov/yov absent for midog21), dtypes, shapes, P, row counts consistent
  * labels binary, both classes present, pos rate; dtr/div contiguous 0..K_train-1, len(train_domains)
  * group disjointness recomputed from disk (tr/iv/ov/ot pairwise; u vs iv/ov/ot), du subset of training domains
  * manifest counts / disjointness flags agree with disk
  * no all-white / flat patches (fails on std==0 or min pixel >= 235); reports near-white counts
  * rows are not label-sorted (adjacent-label-change rate vs 2p(1-p))
  * 64 vs 96: identical source rows in identical order; for c17 cache64 == centre crop of cache96
  * id-val covers every training domain (div == 0..K_train-1)
  * pixel-hash duplicates: no exact duplicate patch within a split; no patch shared between
    labelled splits or between Xu and iv/ov/ot; Xtr/Xu shared patches reported (error if > 1 % of Xu)
  * midog21: cache64 == centre crop [16:80,16:80] of cache96 (s = 1, same centre, reflect padding)
  * c17: ov/ot identical to the v1 cache (same patches); Xu disjoint from id-val patients
  * canine / midog21: v2 64-px ood_test patches are a subset of the v1 ood_test patches (hash multiset;
    midog21: rows without edge padding only -- v1 clamped near-border crops instead)
Writes PNG montages (one per cohort x size x fold; one row per split, 8 neg + 8 pos with
red = positive / green = negative borders, unlabelled grey) and verify_report.json under
$V2/cache_previews/.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

COHORTS = {"c17": 5, "canine": 5, "midog21": 3, "midogpp": 7, "midog21sn": 3}
DEFAULT_COHORTS = ["c17", "canine", "midog21", "midogpp"]
NO_OV = ("midog21", "midog21sn")
V1 = {"c17": os.path.join(os.environ.get("HISTOPATH_DATA", "data"), "cache"),
      "canine": os.path.join(os.environ.get("HISTOPATH_DATA", "data"), "canine_cache"),
      "midog21": os.path.join(os.environ.get("HISTOPATH_DATA", "data"), "midog_cache")}


def ld(d, n):
    p = os.path.join(d, n + ".npy")
    return np.load(p, mmap_mode="r") if os.path.exists(p) else None


def hash_list(X, chunk=8192):
    out = []
    for i in range(0, len(X), chunk):
        c = np.asarray(X[i:i + chunk])
        out += [hashlib.md5(np.ascontiguousarray(x).tobytes()).hexdigest() for x in c]
    return out


def hashes(X):
    return collections.Counter(hash_list(X))


def montage(path, d, splits, P, title):
    n = 16
    pad = 3
    rows = []
    for s in splits:
        X = ld(d, "X" + s)
        if X is None or len(X) == 0:
            continue
        y = ld(d, "y" + s)
        rng = np.random.default_rng(0)
        if y is not None:
            neg = rng.choice(np.where(np.asarray(y) == 0)[0], size=min(8, int((np.asarray(y) == 0).sum())), replace=False)
            pos = rng.choice(np.where(np.asarray(y) == 1)[0], size=min(8, int((np.asarray(y) == 1).sum())), replace=False)
            sel = [(i, 0) for i in neg] + [(i, 1) for i in pos]
        else:
            sel = [(i, -1) for i in rng.choice(len(X), size=min(n, len(X)), replace=False)]
        rows.append((s, [(np.asarray(X[i]), l) for i, l in sel]))
    W = 60 + n * (P + 2 * pad)
    H = 20 + len(rows) * (P + 2 * pad)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    dr.text((4, 4), title, fill=(0, 0, 0))
    for r, (s, items) in enumerate(rows):
        y0 = 20 + r * (P + 2 * pad)
        dr.text((4, y0 + P // 2), s, fill=(0, 0, 0))
        for k, (x, l) in enumerate(items):
            x0 = 60 + k * (P + 2 * pad)
            col = {1: (220, 0, 0), 0: (0, 160, 0), -1: (128, 128, 128)}[l]
            dr.rectangle([x0, y0, x0 + P + 2 * pad - 1, y0 + P + 2 * pad - 1], fill=col)
            canvas.paste(Image.fromarray(x), (x0 + pad, y0 + pad))
    canvas.save(path)


def check_fold(cohort, t, P, root, prev):
    d = os.path.join(root, "cache%d" % P, cohort, "fold%d" % t)
    rep = {"ok": True, "errors": [], "warnings": []}

    def err(m):
        rep["ok"] = False; rep["errors"].append(m)

    if not os.path.isdir(d):
        err("missing dir"); return rep
    man = json.load(open(os.path.join(d, "manifest.json")))
    has_ov = cohort not in NO_OV
    req = ["Xtr", "ytr", "dtr", "gtr", "Xiv", "yiv", "div", "giv", "Xot", "yot", "Xu"] + (["Xov", "yov"] if has_ov else [])
    for f in req:
        if not os.path.exists(os.path.join(d, f + ".npy")):
            err("missing %s" % f)
    if not has_ov and (os.path.exists(os.path.join(d, "Xov.npy")) or os.path.exists(os.path.join(d, "yov.npy"))):
        err("%s must not have Xov/yov" % cohort)
    for k in ("cohort", "fold", "patch_px", "test_domain", "val_domain", "train_domains", "group_kind",
              "disjointness", "source", "seed", "created", "notes"):
        if k not in man:
            err("manifest missing %s" % k)
    K = len(man["train_domains"])
    splits = ["tr", "iv"] + (["ov"] if has_ov else []) + ["ot", "u"]
    G, D, Y = {}, {}, {}
    rep["counts"] = {}
    for s in splits:
        X = ld(d, "X" + s)
        if X is None:
            continue
        if X.dtype != np.uint8 or X.ndim != 4 or X.shape[1:] != (P, P, 3):
            err("X%s bad dtype/shape %s %s" % (s, X.dtype, X.shape))
        n = len(X)
        rep["counts"][s] = n
        if man.get("n_" + s) != n:
            err("manifest n_%s %s != %d" % (s, man.get("n_" + s), n))
        g = ld(d, "g" + s)
        if g is None or len(g) != n or g.dtype != np.int64:
            err("g%s bad" % s)
        G[s] = np.asarray(g)
        if s != "u":
            y = np.asarray(ld(d, "y" + s))
            if y.dtype != np.int64 or len(y) != n or not set(np.unique(y).tolist()) <= {0, 1}:
                err("y%s bad" % s)
            if len(set(np.unique(y).tolist())) < 2:
                err("y%s single class" % s)
            Y[s] = y
            rep["pos_rate_" + s] = float(y.mean())
            p = y.mean(); chg = float(np.mean(y[1:] != y[:-1])); exp = 2 * p * (1 - p)
            rep["label_change_rate_" + s] = [round(chg, 3), round(exp, 3)]
            if abs(chg - exp) > 0.05:
                err("y%s looks sorted (change rate %.3f vs %.3f)" % (s, chg, exp))
        if s in ("tr", "iv", "u"):
            dd = ld(d, "d" + s)
            if dd is None or dd.dtype != np.int64 or len(dd) != n:
                err("d%s bad" % s); continue
            dd = np.asarray(dd)
            D[s] = dd
            if dd.min() < 0 or dd.max() > K - 1:
                err("d%s out of 0..K-1" % s)
            if s == "tr" and sorted(np.unique(dd).tolist()) != list(range(K)):
                err("dtr not contiguous 0..%d: %s" % (K - 1, np.unique(dd)))
            if s == "iv" and sorted(np.unique(dd).tolist()) != list(range(K)):
                err("id-val does not cover every training domain: %s" % np.unique(dd))
        # white / flat patches (vectorised, in chunks)
        flat = white = near = 0
        for i in range(0, n, 4096):
            c = np.asarray(X[i:i + 4096]).reshape(min(4096, n - i), -1)
            sd = c.std(1); mn = c.min(1); mu = c.mean(1)
            flat += int((sd == 0).sum()); white += int((mn >= 235).sum()); near += int(((mu > 230) & (sd < 10)).sum())
        rep.setdefault("white", {})[s] = {"flat": flat, "all_px_ge_235": white, "near_white": near}
        if flat:
            err("X%s has %d flat (std 0) patches" % (s, flat))
        if white > 0.001 * n:
            err("X%s has %d all-white patches (> 0.1%%)" % (s, white))
        elif white:
            rep["warnings"].append("X%s has %d all-white patch(es) (min px >= 235), kept as source data" % (s, white))
    # disjointness from disk
    pairs = [("tr", "iv"), ("tr", "ov"), ("tr", "ot"), ("iv", "ov"), ("iv", "ot"), ("ov", "ot"),
             ("u", "iv"), ("u", "ov"), ("u", "ot")]
    disj = {}
    for a_, b_ in pairs:
        if a_ in G and b_ in G:
            inter = set(np.unique(G[a_]).tolist()) & set(np.unique(G[b_]).tolist())
            disj["%s_vs_%s" % (a_, b_)] = len(inter) == 0
            if inter:
                err("groups overlap %s/%s: %s" % (a_, b_, sorted(inter)[:10]))
    if "u" in D and "tr" in D:
        disj["du_subset_dtr"] = set(np.unique(D["u"]).tolist()) <= set(np.unique(D["tr"]).tolist())
        if not disj["du_subset_dtr"]:
            err("du not subset of training domains")
    rep["disjointness_from_disk"] = disj
    # pixel-hash duplicates within and across splits
    H = {s: hash_list(ld(d, "X" + s)) for s in splits if ld(d, "X" + s) is not None}
    dup = {}
    for s, h in H.items():
        nd = len(h) - len(set(h))
        dup[s] = nd
        if nd:
            err("X%s has %d exact duplicate patches" % (s, nd))
    cross = {}
    names = list(H)
    for i, a_ in enumerate(names):
        for b_ in names[i + 1:]:
            n_sh = len(set(H[a_]) & set(H[b_]))
            cross["%s&%s" % (a_, b_)] = n_sh
            if n_sh and {a_, b_} == {"tr", "u"}:
                msg = "Xtr and Xu share %d patches" % n_sh
                if n_sh > 0.01 * len(H["u"]):
                    err(msg + " (> 1% of Xu)")
                else:
                    rep["warnings"].append(msg)
            elif n_sh:
                err("X%s and X%s share %d identical patches" % (a_, b_, n_sh))
    rep["pixel_duplicates_within"] = dup
    rep["pixel_shared_across"] = cross
    del H
    if not all(man["disjointness"].values()):
        err("manifest disjointness flag false")
    rows = np.load(os.path.join(d, "rows.npz"))
    for s in splits:
        if s != "u" and len(np.unique(rows["rows_" + s])) != len(rows["rows_" + s]):
            err("duplicate source rows in %s" % s)
    # 64 vs 96 consistency
    if prev is not None:
        rows64 = np.load(os.path.join(prev, "rows.npz"))
        for s in splits:
            if not np.array_equal(rows64["rows_" + s], rows["rows_" + s]):
                err("rows differ between cache64 and cache96 for %s" % s)
            if not np.array_equal(np.load(os.path.join(prev, "g%s.npy" % s)), G[s]):
                err("groups differ between cache64 and cache96 for %s" % s)
        if cohort in ("c17", "midog21"):   # no resizing -> 64 px must be the exact centre crop of 96 px
            for s in splits:
                a64 = ld(prev, "X" + s); a96 = ld(d, "X" + s)
                for i in range(0, len(a64), 8192):
                    if not np.array_equal(np.asarray(a64[i:i + 8192]), np.asarray(a96[i:i + 8192])[:, 16:80, 16:80]):
                        err("%s cache64 != centre crop of cache96 (%s)" % (cohort, s)); break
            rep["centre_crop_64_of_96_checked"] = True
        for s in splits:
            if "pad_" + s in rows.files:
                rep.setdefault("edge_padded", {})[s] = {"p64": int((rows64["pad_" + s] > 0).sum()) if "pad_" + s in rows64.files else None,
                                                        "p96": int((rows["pad_" + s] > 0).sum())}
        rep["same_rows_64_96"] = True
    # cross-check against v1 (only once, on the 64 cache)
    if P == 64 and cohort in V1:
        v1d = os.path.join(V1[cohort], "fold%d" % t)
        z = np.load(os.path.join(v1d, "splits.npz"))
        if cohort == "c17":
            for s, k in (("ov", "Xov"), ("ot", "Xot")):   # tr/iv/u re-drawn in v2 (centre-stratified id-val)
                r = rows["rows_" + s]
                pos = np.searchsorted(np.sort(r), r)
                v1X = z[k]
                same = v1X.shape[0] == len(r) and np.array_equal(v1X[pos], np.asarray(ld(d, "X" + s)))
                rep.setdefault("v1_identical", {})[s] = bool(same)
                if not same:
                    err("c17 %s differs from v1" % s)
            ivp = set(json.load(open(os.path.join(d, "manifest.json")))["iv_patients"])
            if set(np.unique(G["u"]).tolist()) & set(int(x) for x in ivp):
                err("Xu contains id-val patients")
        else:
            Xot = np.asarray(ld(d, "Xot"))
            if "pad_ot" in rows.files:          # v1 clamped near-border crops; compare unpadded rows only
                Xot = Xot[rows["pad_ot"] == 0]
            h1 = hashes(z["Xot"]); h2 = hashes(Xot)
            sub = all(h1[k] >= c for k, c in h2.items())
            rep["v1_ot_subset"] = {"v1": sum(h1.values()), "v2": sum(h2.values()), "subset": bool(sub),
                                   "identical": h1 == h2}
            if not sub:
                err("v2 ood_test is not a subset of v1 ood_test")
    os.makedirs(os.path.join(root, "cache_previews"), exist_ok=True)
    montage(os.path.join(root, "cache_previews", "%s_p%d_fold%d.png" % (cohort, P, t)), d, splits, P,
            "%s P=%d fold%d test=%s val=%s (red=pos, green=neg, grey=unlab)" % (
                cohort, P, t, man["test_domain"], man["val_domain"]))
    rep["manifest"] = {k: man[k] for k in ("test_domain", "val_domain", "train_domains")}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default=C.V2)
    ap.add_argument("--cohorts", nargs="+", default=DEFAULT_COHORTS + [
        c for c in COHORTS if c not in DEFAULT_COHORTS and os.path.isdir(os.path.join(C.V2, "cache64", c))])
    a = ap.parse_args()
    report = {}
    allok = True
    for c in a.cohorts:
        for t in range(COHORTS[c]):
            prev = None
            for P in (64, 96):
                r = check_fold(c, t, P, a.v2, prev)
                prev = os.path.join(a.v2, "cache64", c, "fold%d" % t)
                report["%s/p%d/fold%d" % (c, P, t)] = r
                allok &= r["ok"]
                print("%-8s P=%d fold%d %s counts=%s pos=%s %s" % (
                    c, P, t, "OK " if r["ok"] else "FAIL", r.get("counts"),
                    {k[9:]: round(v, 3) for k, v in r.items() if k.startswith("pos_rate_")},
                    "; ".join(r["errors"] + ["WARN " + w for w in r["warnings"]])), flush=True)
                if r.get("v1_identical") or r.get("v1_ot_subset"):
                    print("          v1 check:", r.get("v1_identical") or r.get("v1_ot_subset"), flush=True)
    du = subprocess.run(["du", "-sh"] + [os.path.join(a.v2, "cache%d" % P, c) for P in (64, 96) for c in a.cohorts],
                        capture_output=True, text=True).stdout
    report["_disk"] = du
    report["_all_ok"] = allok
    print(du)
    out = os.path.join(a.v2, "cache_previews", "verify_report.json")
    json.dump(report, open(out, "w"), indent=1, default=str)
    print("ALL OK" if allok else "SOME CHECKS FAILED", "->", out)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
