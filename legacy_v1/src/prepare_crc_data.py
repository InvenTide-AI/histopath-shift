"""Build the NCT-CRC-HE splits for the second-dataset replication.

Task: binary TUM (colorectal adenocarcinoma epithelium) vs non-TUM, balanced.
The BACK class is EXCLUDED (empty slide glass, trivially separable -- including
it would inflate every arm and compress between-arm differences).

Two shift axes are produced (see DESIGN_crc.md):
  primary   -- train on colour-NORMALISED, test on NONORM (stain/acquisition
               shift, same tissue material)
  secondary -- test on CRC_VAL_HE_7K (50 patients, no patient overlap, same
               tissue bank -- patient-disjoint, NOT a second hospital)

Data is streamed at ROW-GROUP granularity over HTTP range requests, so bytes
downloaded ~= bytes decoded (~500 MB) instead of the 31 GB the full release is.

Outputs (cache/):
  crc_splits.npz     Xtr/ytr, Xiv/yiv, Xsh/ysh (NONORM), Xex/yex (VAL_7K)
  crc_unlabeled.npz  Xu -- normalised tiles only, disjoint from train/id_val
  crc_data_manifest.json
"""
import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hfio import BASE, HttpFile

NAMES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
TUM, BACK = 8, 1
CROP = 64
HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="cache")
ap.add_argument("--n-train", type=int, default=20000)
ap.add_argument("--n-idval", type=int, default=4000)
ap.add_argument("--n-shift", type=int, default=4000)
ap.add_argument("--n-unlab", type=int, default=16000)
ap.add_argument("--seed", type=int, default=20240)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.RandomState(a.seed)


def split_rng(name):
    """Independent rng per split, derived from (seed, split name).

    The original single shared `rng` made the result depend on execution
    history: `plan_row_groups` runs before the per-split checkpoint restore,
    but `fetch` consumes three further draws, so a resumed run skipped those
    draws and every LATER split was planned from a shifted state. That is why
    an interrupted original run and an uninterrupted rerun produce different
    shift_nonorm row groups (44 vs 42) from the same seed. Deriving each
    split's stream from its name makes every split reproducible regardless of
    how many splits were restored from checkpoint.
    """
    h = hashlib.sha256(f"{a.seed}:{name}".encode()).digest()
    return np.random.RandomState(int.from_bytes(h[:4], "big"))


def load_index(prefix):
    p = os.path.join(HERE, f"crc_rg_index_{prefix}.json")
    if not os.path.exists(p):
        sys.exit(f"missing {p} -- run build_rg_index.py {prefix} first")
    return json.load(open(p))


def decode(cell):
    """224x224 -> centre CROPxCROP uint8, matching the Camelyon17 pipeline."""
    b = cell["bytes"] if isinstance(cell, dict) else cell
    im = Image.open(io.BytesIO(b)).convert("RGB")
    w, h = im.size
    l, t = (w - CROP) // 2, (h - CROP) // 2
    return np.asarray(im.crop((l, t, l + CROP, t + CROP)), dtype=np.uint8)


def plan_row_groups(index, want_pos, want_neg, exclude=None, rng=None):
    """Choose (shard, rg) units to satisfy positive/negative quotas.

    Row groups are the atomic download unit. Shards are label-sorted so most
    row groups are single-class; we shuffle the candidate pool per class so the
    sample is not biased toward whichever shard happens to sort first.
    """
    exclude = exclude or set()
    rng = globals()["rng"] if rng is None else rng
    pos_pool, neg_pool = [], []
    for shard, rgs in index.items():
        for rg, comp in rgs:
            key = (shard, rg)
            if key in exclude:
                continue
            npos = comp.get(str(TUM), 0)
            nneg = sum(v for k, v in comp.items()
                       if int(k) not in (TUM, BACK))
            if npos:
                pos_pool.append((key, npos))
            if nneg:
                neg_pool.append((key, nneg))
    rng.shuffle(pos_pool)
    rng.shuffle(neg_pool)

    chosen, got_p, got_n = {}, 0, 0
    for (key, n) in pos_pool:
        if got_p >= want_pos:
            break
        chosen.setdefault(key, set()).add("pos")
        got_p += n
    for (key, n) in neg_pool:
        if got_n >= want_neg:
            break
        chosen.setdefault(key, set()).add("neg")
        got_n += n
    if got_p < want_pos or got_n < want_neg:
        print(f"  WARNING: pool exhausted (pos {got_p}/{want_pos}, "
              f"neg {got_n}/{want_neg})")
    return chosen


def _fetch_shard(args):
    """Read the planned row groups of ONE shard. Runs in a worker thread:
    row-group reads are network-bound, so shards are fetched concurrently."""
    shard, rgs, seed = args
    r = np.random.RandomState(seed)
    out = []
    pf = pq.ParquetFile(HttpFile(BASE + shard))
    for rg, roles in sorted(rgs):
        t = pf.read_row_group(rg, columns=["image", "label"])
        labs = t["label"].to_numpy()
        imgs = t["image"].to_pylist()
        for i in r.permutation(len(labs)):
            lab = int(labs[i])
            if lab == BACK:
                continue
            is_pos = lab == TUM
            if (is_pos and "pos" not in roles) or (not is_pos and "neg" not in roles):
                continue
            out.append((decode(imgs[i]), 1 if is_pos else 0))
    return shard, out


def fetch(prefix, plan, cap_pos, cap_neg, workers=8, rng=None):
    """Download and decode the planned row groups, shards in parallel.

    Quotas are applied after collection rather than inside the loop: the plan
    already sizes the request, and post-hoc capping keeps the parallel workers
    independent of each other's running counts.
    """
    rng = globals()["rng"] if rng is None else rng
    by_shard = {}
    for (shard, rg), roles in plan.items():
        by_shard.setdefault(shard, []).append((rg, roles))
    jobs = [(s, rgs, a.seed + i) for i, (s, rgs) in enumerate(sorted(by_shard.items()))]

    pos, neg = [], []
    done = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for shard, out in ex.map(_fetch_shard, jobs):
            for img, lab in out:
                (pos if lab else neg).append(img)
            done += 1
            if done % 5 == 0 or done == len(jobs):
                print(f"    {prefix[:22]} shard {done}/{len(jobs)} "
                      f"pos={len(pos)} neg={len(neg)}", flush=True)
    if len(pos) < cap_pos or len(neg) < cap_neg:
        print(f"  WARNING: short of quota (pos {len(pos)}/{cap_pos}, "
              f"neg {len(neg)}/{cap_neg})")
    pos = [pos[i] for i in rng.permutation(len(pos))[:cap_pos]]
    neg = [neg[i] for i in rng.permutation(len(neg))[:cap_neg]]
    X = np.stack(pos + neg)
    y = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.int64)
    p = rng.permutation(len(y))
    return X[p], y[p]


manifest = {
    "source": "huggingface.co/datasets/1aurent/NCT-CRC-HE",
    "origin": "Kather JN, Halama N, Marx A (2018), Zenodo",
    "doi": "10.5281/zenodo.1214456",
    "license": "CC-BY-4.0",
    "task": "binary TUM (adenocarcinoma epithelium) vs non-TUM, class-balanced",
    "excluded_class": "BACK (empty slide glass; trivially separable)",
    "crop": f"centre {CROP}x{CROP} of 224x224 (0.5 MPP source)",
    "seed": a.seed,
    "model_selection": "in-distribution validation only; no third site exists "
                       "in this dataset, so the WILDS ood_val protocol used for "
                       "Camelyon17 cannot be reproduced (see DESIGN_crc.md)",
    "splits": {},
}

idx_norm = load_index("NCT_CRC_HE_100K")

# ---- train + id_val + unlabeled, all from the NORMALISED cohort ----
# Allocated from disjoint row-group pools so no tile can appear in two splits.
need = [("train", a.n_train), ("id_val", a.n_idval), ("unlabeled", a.n_unlab)]
used = set()
parts = {}
for name, n in need:
    r = split_rng(name)
    plan = plan_row_groups(idx_norm, n // 2, n // 2, exclude=used, rng=r)
    used |= set(plan.keys())
    # per-split checkpoint: a long stream must survive an interrupted session
    ck = os.path.join(a.out, f"_part_{name}.npz")
    if os.path.exists(ck):
        z = np.load(ck)
        X, y = z["X"], z["y"]
        print(f"  {name}: restored from {ck} {X.shape}", flush=True)
    else:
        X, y = fetch(f"NCT_CRC_HE_100K/{name}", plan, n // 2, n // 2, rng=r)
        np.savez_compressed(ck, X=X, y=y)
    parts[name] = (X, y)
    manifest["splits"][name] = {
        "cohort": "NCT_CRC_HE_100K (colour-normalised)",
        "n": int(len(y)), "pos": int(y.sum()), "neg": int((y == 0).sum()),
        "row_groups": len(plan),
    }
    print(f"  {name}: {X.shape} pos={int(y.sum())} neg={int((y == 0).sum())}",
          flush=True)

# ---- primary shift axis: NONORM, same tissue material, no colour normalisation
idx_non = load_index("NCT_CRC_HE_100K_NONORM")
r_sh = split_rng("shift_nonorm")
plan = plan_row_groups(idx_non, a.n_shift // 2, a.n_shift // 2, rng=r_sh)
ck = os.path.join(a.out, "_part_shift_nonorm.npz")
if os.path.exists(ck):
    z = np.load(ck)
    Xsh, ysh = z["X"], z["y"]
    print(f"  shift_nonorm: restored {Xsh.shape}", flush=True)
else:
    Xsh, ysh = fetch("NONORM/shift", plan, a.n_shift // 2, a.n_shift // 2,
                     rng=r_sh)
    np.savez_compressed(ck, X=Xsh, y=ysh)
manifest["splits"]["shift_nonorm"] = {
    "cohort": "NCT_CRC_HE_100K_NONORM (no colour normalisation)",
    "role": "PRIMARY shift axis: stain/acquisition shift, tissue material held "
            "approximately constant",
    "n": int(len(ysh)), "pos": int(ysh.sum()), "neg": int((ysh == 0).sum()),
    "row_groups": len(plan),
}
print(f"  shift_nonorm: {Xsh.shape}", flush=True)

# ---- secondary axis: external patient-disjoint cohort ----
idx_ex = load_index("CRC_VAL_HE_7K")
r_ex = split_rng("shift_external")
plan = plan_row_groups(idx_ex, 10 ** 9, 10 ** 9, rng=r_ex)  # take all available
ck = os.path.join(a.out, "_part_external.npz")
if os.path.exists(ck):
    z = np.load(ck)
    Xex, yex = z["X"], z["y"]
    print(f"  shift_external: restored {Xex.shape}", flush=True)
else:
    Xex, yex = fetch("VAL_7K/external", plan, 10 ** 9, 10 ** 9, rng=r_ex)
    np.savez_compressed(ck, X=Xex, y=yex)
# balance by subsampling the majority class
npos = int(yex.sum())
nneg = int((yex == 0).sum())
k = min(npos, nneg)
r_bal = split_rng("shift_external_balance")
sel = np.concatenate([r_bal.permutation(np.where(yex == c)[0])[:k]
                      for c in (0, 1)])
sel.sort()
Xex, yex = Xex[sel], yex[sel]
manifest["splits"]["shift_external"] = {
    "cohort": "CRC_VAL_HE_7K",
    "role": "SECONDARY axis: 50 patients, no patient overlap with the 100K set, "
            "but SAME NCT tissue bank -- patient-disjoint external cohort, NOT "
            "a second hospital",
    "n": int(len(yex)), "pos": int(yex.sum()), "neg": int((yex == 0).sum()),
    "balanced_from": {"pos": npos, "neg": nneg},
}
print(f"  shift_external: {Xex.shape}", flush=True)

Xtr, ytr = parts["train"]
Xiv, yiv = parts["id_val"]
Xu = parts["unlabeled"][0]

np.savez_compressed(os.path.join(a.out, "crc_splits.npz"),
                    Xtr=Xtr, ytr=ytr, Xiv=Xiv, yiv=yiv,
                    Xsh=Xsh, ysh=ysh, Xex=Xex, yex=yex)
np.savez_compressed(os.path.join(a.out, "crc_unlabeled.npz"), Xu=Xu)
manifest["splits"]["unlabeled"]["note"] = (
    "SSL pre-training pool. Normalised cohort ONLY -- the encoder never sees "
    "NONORM appearance, which would leak the primary test condition. Labels "
    "are discarded.")
json.dump(manifest, open(os.path.join(a.out, "crc_data_manifest.json"), "w"),
          indent=1)
print("wrote", {k: v.shape for k, v in
                dict(Xtr=Xtr, Xiv=Xiv, Xsh=Xsh, Xex=Xex, Xu=Xu).items()})
