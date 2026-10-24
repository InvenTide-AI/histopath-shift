"""Shared helpers for the v2 fold caches (CONTRACTS.md section A).

Every fold directory gets plain .npy arrays (mmap-able) plus manifest.json:
  Xtr ytr dtr gtr | Xiv yiv div giv | Xov yov gov (absent for midog21) | Xot yot got |
  Xu gu du | rows.npz (source-row ids per split, for provenance; pad_<split> = per-row
  edge-padding px for MIDOG labelled patches) | manifest.json
Extra files beyond the contract (gov, got, gu, du, rows.npz) are kept so the
disjointness checks can be recomputed from disk by verify_v2.py.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil

import numpy as np

V2 = os.path.join(os.environ.get("HISTOPATH_DATA", "data"), "v2")
SPLIT_CODES = {"tr": 1, "iv": 2, "ov": 3, "ot": 4, "u": 5}


def is_tissue(p: np.ndarray, max_mean=220.0, min_std=12.0) -> bool:
    """MIDOG tissue filter (v1 prepare_midog.is_tissue)."""
    return p.mean() < max_mean and p.std() > min_std


def shuffle_perm(n: int, seed: int, fold: int, split: str) -> np.ndarray:
    """Fixed, split-specific row permutation (independent of the selection RNG)."""
    rng = np.random.default_rng([int(seed), int(fold), SPLIT_CODES[split], 7919])
    return rng.permutation(n)


def stratified_cap(idx: np.ndarray, strata: np.ndarray, cap: int, rng) -> np.ndarray:
    """Proportional stratified subsample of idx to `cap` rows (strata per idx row).
    Returns idx unchanged when len(idx) <= cap."""
    if cap is None or len(idx) <= cap:
        return idx
    keys, inv = np.unique(strata, return_inverse=True, axis=0) if strata.ndim > 1 else \
        np.unique(strata, return_inverse=True)
    inv = inv.reshape(-1)
    counts = np.bincount(inv)
    quota = np.floor(counts / counts.sum() * cap).astype(int)
    # distribute the remainder by largest fractional part
    rem = cap - quota.sum()
    frac = counts / counts.sum() * cap - quota
    for k in np.argsort(-frac)[:rem]:
        quota[k] += 1
    out = []
    for k in range(len(counts)):
        members = idx[inv == k]
        out.append(rng.choice(members, size=min(quota[k], len(members)), replace=False))
    return np.sort(np.concatenate(out))


def _save(d, name, arr):
    np.save(os.path.join(d, name + ".npy"), np.ascontiguousarray(arr))


def _inter(a, b):
    return sorted(set(np.unique(a).tolist()) & set(np.unique(b).tolist()))


def disjointness(splits: dict) -> dict:
    """splits: name -> dict(g=group ids, d=domain names array). Computes real set
    intersections of group ids for every pair; returns {pair: bool} + details."""
    out, detail = {}, {}
    names = [n for n in ("tr", "iv", "ov", "ot", "u") if n in splits]
    longname = {"tr": "train", "iv": "iv", "ov": "ov", "ot": "ot", "u": "u"}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a == "tr" and b == "u":
                continue  # unlabelled pool is allowed to share training groups
            inter = _inter(splits[a]["g"], splits[b]["g"])
            key = "%s_vs_%s" % (longname[a], longname[b])
            out[key] = len(inter) == 0
            if inter:
                detail[key] = inter[:20]
    # domain-level: unlabelled pool only from training domains
    if "u" in splits and "tr" in splits:
        out["u_domains_subset_of_train_domains"] = set(np.unique(splits["u"]["d"]).tolist()) <= \
            set(np.unique(splits["tr"]["d"]).tolist())
    if "iv" in splits:
        out["iv_domains_subset_of_train_domains"] = set(np.unique(splits["iv"]["d"]).tolist()) <= \
            set(np.unique(splits["tr"]["d"]).tolist())
    return out, detail


def write_fold(out_dir: str, cohort: str, fold: int, patch: int, seed: int,
               test_domain: str, val_domain, train_domains: list,
               parts: dict, group_kind: str, source: str, notes, extra: dict | None = None):
    """parts: split -> dict(X=uint8 (n,P,P,3), y=int64 (labelled only), d=domain-name array,
    g=int64 group ids, rows=source row ids, pad=optional per-row edge-padding px).
    Splits: tr, iv, ov (optional), ot, u.
    Shuffles every split with a fixed seed, remaps training domains to 0..K_train-1,
    computes disjointness and writes everything.
    Atomic: the fold is written to <out_dir>.tmp and swapped in by rename, so a
    process that has the previous cache mmap-ed keeps reading consistent (old) data."""
    final_dir = out_dir
    out_dir = final_dir.rstrip("/") + ".tmp"
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    dmap = {name: k for k, name in enumerate(train_domains)}
    shuffled = {}
    for s, p in parts.items():
        n = len(p["X"])
        perm = shuffle_perm(n, seed, fold, s)
        q = {k: (np.asarray(v)[perm] if v is not None else None) for k, v in p.items()}
        shuffled[s] = q
    rows = {}
    for s, q in shuffled.items():
        X = q["X"]
        assert X.dtype == np.uint8 and X.ndim == 4 and X.shape[1:] == (patch, patch, 3), (s, X.shape)
        _save(out_dir, "X" + s, X)
        g = np.asarray(q["g"], dtype=np.int64)
        _save(out_dir, "g" + s, g)
        if s != "u":
            _save(out_dir, "y" + s, np.asarray(q["y"], dtype=np.int64))
        if s in ("tr", "iv", "u"):
            dd = np.asarray([dmap[x] for x in q["d"]], dtype=np.int64)
            _save(out_dir, "d" + s, dd)
        rows["rows_" + s] = np.asarray(q["rows"])
        if q.get("pad") is not None:
            rows["pad_" + s] = np.asarray(q["pad"], dtype=np.int64)
    np.savez(os.path.join(out_dir, "rows.npz"), **rows)
    disj, detail = disjointness({s: {"g": q["g"], "d": q["d"]} for s, q in shuffled.items()})
    man = {
        "cohort": cohort, "fold": fold, "patch_px": patch,
        "test_domain": test_domain, "val_domain": val_domain, "train_domains": list(train_domains),
        "group_kind": group_kind, "source": source, "seed": seed,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "disjointness": disj, "disjointness_violations": detail,
        "notes": notes,
    }
    for s in ("tr", "iv", "ov", "ot", "u"):
        man["n_" + s] = int(len(shuffled[s]["X"])) if s in shuffled else 0
    for s in ("tr", "iv", "ov", "ot"):
        man["pos_rate_" + s] = float(np.mean(shuffled[s]["y"])) if s in shuffled and len(shuffled[s]["y"]) else None
    for s in ("tr", "iv", "u"):
        if s in shuffled:
            dd = np.asarray([dmap[x] for x in shuffled[s]["d"]])
            man["domain_counts_" + s] = {train_domains[k]: int((dd == k).sum()) for k in range(len(train_domains))}
    for s in ("tr", "iv", "ov", "ot", "u"):
        if s in shuffled:
            man["n_groups_" + s] = int(len(np.unique(shuffled[s]["g"])))
            if shuffled[s].get("pad") is not None:
                man["n_edge_padded_" + s] = int((np.asarray(shuffled[s]["pad"]) > 0).sum())
    if extra:
        man.update(extra)
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    bad = [k for k, v in disj.items() if not v]
    assert not bad, "disjointness violated in %s: %s %s" % (out_dir, bad, detail)
    # swap in atomically (old files stay alive for any process that still has them open)
    if os.path.isdir(final_dir):
        old = final_dir.rstrip("/") + ".old"
        if os.path.isdir(old):
            shutil.rmtree(old)
        os.rename(final_dir, old)
        os.rename(out_dir, final_dir)
        shutil.rmtree(old)
    else:
        os.rename(out_dir, final_dir)
    return man


def code_sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
