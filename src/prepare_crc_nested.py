"""Build composition-matched NESTED unlabeled pools for the CRC pool-size test.

See DESIGN_crc_pool40k.md. The first attempt at a larger pool failed on
arithmetic: the NCT-CRC-HE cohort holds 14,317 TUM tiles and
train(20k)+id_val(4k)+the original pool consume 14,252 of them, so any larger
pool can only be filled with non-tumour tissue (the built pool came out 0%
tumour). Size and composition were therefore confounded in the direction that
would manufacture a negative result.

This script fixes that by (a) shrinking the labeled train split to 15,000,
freeing tumour headroom, and (b) making the small pool a strict class-stratified
SUBSET of the large one, so composition is identical by construction and only
size varies:

  pool20k  20,000 tiles, 21.96% TUM   (matches the published pool's composition)
  pool10k  10,000 tiles, 21.96% TUM   (a nested subset of pool20k)

Only the pool and the labeled train split change; shift_nonorm and
shift_external are rebuilt unchanged by the same seeded planner.

Outputs (cache/):
  crc_splits_n15k.npz     Xtr/ytr (15k), Xiv/yiv, Xsh/ysh, Xex/yex
  crc_unlabeled_p10k.npz  Xu -- nested subset
  crc_unlabeled_p20k.npz  Xu -- superset
  crc_nested_manifest.json
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="cache")
ap.add_argument("--n-train", type=int, default=15000)
ap.add_argument("--n-idval", type=int, default=4000)
ap.add_argument("--n-shift", type=int, default=4000)
ap.add_argument("--n-pool-big", type=int, default=20000)
ap.add_argument("--n-pool-small", type=int, default=10000)
ap.add_argument("--tum-frac", type=float, default=0.2196,
                help="tumour fraction of the published pool, preserved in both")
ap.add_argument("--seed", type=int, default=20240)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def split_rng(name):
    """Per-split stream -- same derivation as prepare_crc_data.py."""
    h = hashlib.sha256(f"{a.seed}:{name}".encode()).digest()
    return np.random.RandomState(int.from_bytes(h[:4], "big"))


def load_helpers():
    """Reuse prepare_crc_data.py's fetch/plan machinery without re-running it.

    That module executes its pipeline at import, so only the definitions above
    the pipeline are exec'd here. This keeps ONE implementation of row-group
    planning and decoding rather than a third copy that could silently drift.
    """
    path = os.path.join(HERE, "prepare_crc_data.py")
    head = open(path).read().split("manifest = {")[0]
    ns = {"__name__": "notmain", "__file__": path}
    sys.argv = ["prepare_crc_data.py", "--out", a.out, "--seed", str(a.seed)]
    exec(compile(head, path, "exec"), ns)
    return ns


def plan_hash(plan):
    return hashlib.sha256(
        json.dumps(sorted(map(list, plan.keys()))).encode()).hexdigest()[:16]


def cached_fetch(name, prefix, plan, cap_pos, cap_neg, rng):
    """Fetch with a checkpoint keyed on the PLAN, not just the filename.

    prepare_crc_data.py keys its checkpoints on the split name alone. That is
    safe there because the plan is fixed, but this script allocates id_val
    after a SMALLER train split, so a same-named checkpoint from another run
    would restore tiles that belong to a different plan -- silently, and with
    no way to notice downstream. Storing the plan hash alongside the tiles
    turns that into a refetch instead of a wrong answer.
    """
    ck = os.path.join(a.out, f"_part_nested_{name}.npz")
    want = plan_hash(plan)
    if os.path.exists(ck):
        z = np.load(ck)
        got = str(z["plan"]) if "plan" in z else None
        if got == want:
            print(f"  {name}: restored {z['X'].shape} (plan {want})",
                  flush=True)
            return z["X"], z["y"]
        print(f"  {name}: checkpoint is for plan {got}, need {want} "
              f"-- refetching", flush=True)
    X, y = fetch(prefix, plan, cap_pos, cap_neg, rng=rng)
    np.savez_compressed(ck, X=X, y=y, plan=want)
    return X, y


H = load_helpers()
plan_row_groups, fetch, load_index = (
    H["plan_row_groups"], H["fetch"], H["load_index"])
TUM = H["TUM"]

manifest = {
    "purpose": "composition-matched nested pools for the CRC pool-size test",
    "seed": a.seed,
    "cohort": "NCT_CRC_HE_100K (colour-normalised) for train/id_val/pools",
    "design": ("pool10k is a class-stratified SUBSET of pool20k, so pool "
               "composition is identical by construction and only size varies"),
    "splits": {},
}

idx_norm = load_index("NCT_CRC_HE_100K")

# ---- labeled splits: train shrunk to free tumour headroom for the pools ----
used, parts = set(), {}
for name, n in [("train_n15k", a.n_train), ("id_val", a.n_idval)]:
    r = split_rng(name)
    plan = plan_row_groups(idx_norm, n // 2, n // 2, exclude=used, rng=r)
    used |= set(plan.keys())
    X, y = cached_fetch(name, f"NCT_CRC_HE_100K/{name}", plan,
                        n // 2, n // 2, r)
    parts[name] = (X, y)
    manifest["splits"][name] = {
        "n": int(len(y)), "pos": int(y.sum()), "row_groups": len(plan)}
    print(f"  {name}: {X.shape} pos={int(y.sum())}", flush=True)

# ---- the BIG pool, at the published tumour fraction ----
n_big = a.n_pool_big
want_pos = int(round(a.tum_frac * n_big))
want_neg = n_big - want_pos
r_pool = split_rng("pool_nested")
plan = plan_row_groups(idx_norm, want_pos, want_neg, exclude=used, rng=r_pool)
print(f"  pool20k plan: {len(plan)} row groups "
      f"(want {want_pos} TUM / {want_neg} non-TUM)", flush=True)

Xu, yu = cached_fetch("pool", "pool_nested", plan, want_pos, want_neg, r_pool)

if len(Xu) < n_big:
    sys.exit(f"pool short: got {len(Xu)} of {n_big} -- tumour headroom "
             f"insufficient; lower --n-train or --tum-frac")

# ---- the SMALL pool: class-stratified subset, so composition is identical ----
r_sub = split_rng("pool_nested_subset")
half = a.n_pool_small
sub_pos = int(round(a.tum_frac * half))
sub_neg = half - sub_pos
ipos = np.where(yu == 1)[0]
ineg = np.where(yu == 0)[0]
sel = np.concatenate([r_sub.permutation(ipos)[:sub_pos],
                      r_sub.permutation(ineg)[:sub_neg]])
sel = sel[r_sub.permutation(len(sel))]
Xs, ys = Xu[sel], yu[sel]

np.savez_compressed(os.path.join(a.out, "crc_unlabeled_p20k.npz"), Xu=Xu, yu=yu)
np.savez_compressed(os.path.join(a.out, "crc_unlabeled_p10k.npz"), Xu=Xs, yu=ys)

# Nesting is the whole point of the design, so verify it on CONTENT rather
# than on indices: hash each tile and confirm pool10k's multiset of tiles is
# contained in pool20k's, and that the class balance matches.
def _tile_hashes(X):
    return [hashlib.sha1(t.tobytes()).hexdigest() for t in X]


_big, _small = _tile_hashes(Xu), _tile_hashes(Xs)
from collections import Counter as _C
_cb, _cs = _C(_big), _C(_small)
assert all(_cs[h] <= _cb[h] for h in _cs), "pool10k is not nested in pool20k"
assert abs(ys.mean() - yu.mean()) < 0.005, (
    f"composition drift: {ys.mean():.4f} vs {yu.mean():.4f}")
print(f"  nesting verified on tile content: {len(_cs)} distinct tiles "
      f"of pool10k all present in pool20k", flush=True)
manifest["pools"] = {
    "pool20k": {"n": int(len(yu)), "pos": int(yu.sum()),
                "tumour_fraction": float(yu.mean()), "row_groups": len(plan)},
    "pool10k": {"n": int(len(ys)), "pos": int(ys.sum()),
                "tumour_fraction": float(ys.mean()),
                "nested_in": "pool20k (class-stratified subset)"},
}
print(f"  pool20k: n={len(yu)} TUM={int(yu.sum())} ({yu.mean():.4%})",
      flush=True)
print(f"  pool10k: n={len(ys)} TUM={int(ys.sum())} ({ys.mean():.4%})",
      flush=True)

# ---- shift axes: unchanged, same seeded planner as the main script ----
for name, prefix, index_name, n in [
        ("shift_nonorm", "NONORM/shift", "NCT_CRC_HE_100K_NONORM", a.n_shift),
        ("shift_external", "VAL_7K/external", "CRC_VAL_HE_7K", None)]:
    r = split_rng(name)
    idx = load_index(index_name)
    cap = (n // 2, n // 2) if n else (10 ** 9, 10 ** 9)
    plan = plan_row_groups(idx, *cap, rng=r)
    X, y = cached_fetch(name, prefix, plan, cap[0], cap[1], r)
    parts[name] = (X, y)
    manifest["splits"][name] = {"n": int(len(y)), "pos": int(y.sum())}
    print(f"  {name}: {X.shape} pos={int(y.sum())}", flush=True)

# balance the external cohort exactly as prepare_crc_data.py does
Xex, yex = parts["shift_external"]
k = int(min((yex == 0).sum(), (yex == 1).sum()))
r_bal = split_rng("shift_external_balance")
sel = np.concatenate([r_bal.permutation(np.where(yex == c)[0])[:k]
                      for c in (0, 1)])
sel = sel[r_bal.permutation(len(sel))]
Xex, yex = Xex[sel], yex[sel]
manifest["splits"]["shift_external"] = {"n": int(len(yex)),
                                        "pos": int(yex.sum())}

Xtr, ytr = parts["train_n15k"]
Xiv, yiv = parts["id_val"]
Xsh, ysh = parts["shift_nonorm"]
np.savez_compressed(os.path.join(a.out, "crc_splits_n15k.npz"),
                    Xtr=Xtr, ytr=ytr, Xiv=Xiv, yiv=yiv,
                    Xsh=Xsh, ysh=ysh, Xex=Xex, yex=yex)
json.dump(manifest, open(os.path.join(a.out, "crc_nested_manifest.json"), "w"),
          indent=1)
print("\nwrote crc_splits_n15k.npz, crc_unlabeled_p{10,20}k.npz", flush=True)
print(json.dumps(manifest["pools"], indent=1), flush=True)
