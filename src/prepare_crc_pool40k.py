"""Build a 40k-tile UNLABELED pool for the pool-size test (CRC replication).

Motivation
----------
The original CRC replication used a 10,226-tile SSL pool; Camelyon17 used
40,000. That 3.9x gap was confounded with the dataset comparison, so the
"does SSL+SAM clear baseline under stain shift" question was never tested at
matched pool size. This script builds the size-matched pool.

Design constraints (both forced by the data, not chosen)
--------------------------------------------------------
1. NORMALISED COHORT ONLY. NONORM tiles are the test condition; including them
   would convert domain generalization into unsupervised domain adaptation and
   break comparability with Camelyon17, whose unlabeled pool is hospitals
   0/3/4 and excludes test hospital 2 -- diversity without target access.

2. TUMOUR IS EXHAUSTED, SO THE POOL IS UNBALANCED. Class 8 (TUM) holds 14,317
   tiles and train+id_val+the original pool already consumed 14,226, leaving
   91. A 40k pool is therefore 5.8% tumour vs the original 21.8%. Size and
   composition cannot be varied independently in this dataset -- a
   composition-matched pool caps at 10,644 tiles (1.04x), too small to test a
   size effect. Camelyon17's pool was itself uniform random over
   unlabeled_train with no class balancing, so an unbalanced pool is the
   closer analogue; this is a real limitation of the test, recorded here and
   in the manifest.

train and id_val are re-derived with the SAME seed (20240) and the SAME
sequential allocation order as prepare_crc_data.py, so the labeled data is
bit-identical to the original runs and the only variable is the pool.

Outputs (cache/):
  crc_unlabeled40k.npz        Xu
  crc_pool40k_manifest.json
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
ap.add_argument("--n-unlab-orig", type=int, default=16000)
ap.add_argument("--n-unlab-new", type=int, default=40000)
ap.add_argument("--seed", type=int, default=20240)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.RandomState(a.seed)


def split_rng(name):
    """Per-split stream, identical to prepare_crc_data.py's derivation.

    Must match exactly: this script replays the labeled allocation to
    reconstruct the `used` exclusion set, so any divergence would reserve
    different row groups and leak train/id_val tiles into the new pool.
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

    Identical to prepare_crc_data.plan_row_groups so the sequential allocation
    reproduces the original train/id_val splits exactly under the same seed.
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
            nneg = sum(v for k, v in comp.items() if int(k) not in (TUM, BACK))
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
    return chosen


def _fetch_shard(args):
    """Read the planned row groups of ONE shard. Byte-for-byte the same logic
    as prepare_crc_data._fetch_shard (sorted row groups, seeded within-group
    permutation, explicit column projection) so tiles are drawn identically."""
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
    by_shard = {}
    for (shard, rg), roles in plan.items():
        by_shard.setdefault(shard, []).append((rg, roles))
    jobs = [(s, rgs, a.seed + i) for i, (s, rgs) in enumerate(sorted(by_shard.items()))]

    rng = globals()["rng"] if rng is None else rng
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
        print(f"  NOTE: short of quota (pos {len(pos)}/{cap_pos}, "
              f"neg {len(neg)}/{cap_neg})", flush=True)
    pos = [pos[i] for i in rng.permutation(len(pos))[:cap_pos]]
    neg = [neg[i] for i in rng.permutation(len(neg))[:cap_neg]]
    X = np.stack(pos + neg)
    y = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.int64)
    return X, y


idx_norm = load_index("NCT_CRC_HE_100K")

# ---- replay the ORIGINAL sequential allocation so labeled data is unchanged
# and the original pool's row groups stay excluded from the new pool.
used = set()
for name, n in [("train", a.n_train), ("id_val", a.n_idval),
                ("unlabeled", a.n_unlab_orig)]:
    plan = plan_row_groups(idx_norm, n // 2, n // 2, exclude=used,
                           rng=split_rng(name))
    used |= set(plan.keys())
    print(f"  replayed {name}: {len(plan)} row groups reserved", flush=True)

# ---- the new pool: everything else in the normalised cohort, uniform over
# what remains. Tumour is nearly exhausted, so quotas are set to "take all
# available" and the resulting composition is recorded rather than imposed.
r_pool = split_rng("unlabeled40k")
plan = plan_row_groups(idx_norm, 10 ** 9, 10 ** 9, exclude=used, rng=r_pool)
print(f"  new pool: {len(plan)} row groups", flush=True)

ck = os.path.join(a.out, "_part_unlab40k.npz")
if os.path.exists(ck):
    z = np.load(ck)
    Xu, yu = z["X"], z["y"]
    print(f"  restored {ck} {Xu.shape}", flush=True)
else:
    Xu, yu = fetch("pool40k", plan, a.n_unlab_new // 2, a.n_unlab_new // 2,
                   rng=r_pool)
    np.savez_compressed(ck, X=Xu, y=yu)

# cap to the target size, shuffled so the npz is not class-ordered
if len(Xu) > a.n_unlab_new:
    keep = r_pool.permutation(len(Xu))[:a.n_unlab_new]
else:
    keep = r_pool.permutation(len(Xu))
Xu, yu = Xu[keep], yu[keep]

np.savez_compressed(os.path.join(a.out, "crc_unlabeled40k.npz"), Xu=Xu)

manifest = {
    "purpose": "pool-size test for the CRC stain-shift axis",
    "seed": a.seed,
    "cohort": "NCT_CRC_HE_100K (colour-normalised) ONLY -- no NONORM tiles",
    "n": int(len(Xu)),
    "pos": int(yu.sum()),
    "neg": int((yu == 0).sum()),
    "tumour_fraction": round(float(yu.mean()), 4),
    "row_groups": len(plan),
    "disjoint_from": ["train", "id_val", "original unlabeled pool"],
    "note": (
        "Unbalanced by necessity: TUM headroom after train+id_val+original "
        "pool is ~91 tiles, so a 40k pool cannot preserve the original 21.8% "
        "tumour fraction. Camelyon17's unlabeled pool was itself uniform "
        "random with no class balancing. Pool SIZE and COMPOSITION are "
        "therefore confounded in this test; see DESIGN_crc_pool40k.md."
    ),
}
json.dump(manifest, open(os.path.join(a.out, "crc_pool40k_manifest.json"), "w"),
          indent=1)
print(json.dumps(manifest, indent=1), flush=True)
