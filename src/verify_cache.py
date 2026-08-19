"""Verify a rebuilt Camelyon17 cache is sample-identical to the published runs.

Why this exists: new training runs are only comparable to the published ones if
the rebuilt cache contains the SAME samples, not merely the same split sizes. A
prepare_data.py that reads a different subset of source shards produces splits
with correct sizes and class balances that are nonetheless different data.

Checks, in increasing strength:
  1. cache/data_manifest.json agrees with the published results manifest on
     shard set, seed, per-split caps and available row counts;
  2. cached split sizes and class balances match the manifest;
  3. sample-level -- the selection indices are re-derived independently from the
     raw parquet label columns under the recorded seed, and the resulting label
     vectors must equal the cached ones exactly.

Exit status 0 if every check passes, 1 otherwise. Prints one line per check.
"""
import json, os, sys, glob
import numpy as np
import pyarrow.parquet as pq

SRC = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(SRC, "cache")
PUBLISHED = os.path.join(os.path.dirname(SRC), "results", "camelyon17",
                         "data_manifest.json")

# Must mirror PLAN in prepare_data.py exactly: same order (one shared RNG is
# advanced across splits, so order changes the draws) and same caps.
PLAN = [("id_train", "ytr", 60000), ("id_val", "yiv", 12000),
        ("ood_val", "yov", 12000), ("ood_test", "yot", 25000),
        ("unlabeled_train", None, 40000)]
SPLITS = tuple(s for s, _, _ in PLAN)

fails = []


def check(label, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not ok:
        fails.append(label)
    return ok


def main():
    if not os.path.exists(PUBLISHED):
        print(f"published manifest not found: {PUBLISHED}")
        return 1
    pub = json.load(open(PUBLISHED))
    loc_path = os.path.join(CACHE, "data_manifest.json")
    if not os.path.exists(loc_path):
        print(f"no rebuilt cache manifest at {loc_path} -- run prepare_data.py first")
        return 1
    loc = json.load(open(loc_path))

    # The provenance manifests ARE committed; the arrays they describe are not
    # (gitignored, several GB, rebuilt by prepare_data.py). A fresh clone thus
    # reaches this point with manifests but no data -- say so instead of raising
    # FileNotFoundError from inside a numpy call.
    splits_npz = os.path.join(CACHE, "splits.npz")
    if not os.path.exists(splits_npz):
        print(f"no cached arrays at {splits_npz}\n"
              "The provenance manifests are committed but the arrays are not "
              "(gigabytes, gitignored).\nRebuild them first:\n"
              "    python src/prepare_data.py       # CXR / Camelyon17 splits\n"
              "    python src/decode_patches.py     # multi-hospital patch cache\n"
              "then re-run this script to verify the rebuild is sample-identical "
              "to the published runs.")
        return 1

    # ---- 1. manifest agreement with the published run
    check("subsample_seed matches",
          pub.get("subsample_seed") == loc.get("subsample_seed"),
          f"{pub.get('subsample_seed')} vs {loc.get('subsample_seed')}")
    check("hospital assignment matches",
          pub.get("hospitals") == loc.get("hospitals"))
    for k in SPLITS:
        pk, lk = pub["splits"].get(k, {}), loc["splits"].get(k, {})
        if not pk or not lk:
            check(f"{k}: present in both manifests", False)
            continue
        check(f"{k}: n_used matches", pk["n_used"] == lk["n_used"],
              f"{pk['n_used']} vs {lk['n_used']}")
        # Decisive for provenance: same shard set AND same total rows visible.
        # If the original read 2 of 5 unlabeled shards, a rebuild reading all 5
        # draws from a larger pool and yields different samples at the same size.
        check(f"{k}: shards_read matches",
              sorted(pk["shards_read"]) == sorted(lk["shards_read"]),
              f"{len(lk['shards_read'])} shard(s)")
        check(f"{k}: n_available_in_shards matches",
              pk["n_available_in_shards"] == lk["n_available_in_shards"],
              f"{pk['n_available_in_shards']} vs {lk['n_available_in_shards']}")

    # ---- 2. cached arrays agree with their own manifest
    z = np.load(os.path.join(CACHE, "splits.npz"))
    for split, yk, _ in PLAN:
        if yk is None:
            continue
        if yk not in z:
            check(f"{split}: present in cache", False, f"missing key {yk}")
            continue
        y = z[yk]
        check(f"{split}: cached size matches manifest",
              len(y) == loc["splits"][split]["n_used"],
              f"{len(y)} vs {loc['splits'][split]['n_used']}")
        cb = loc["splits"][split].get("class_balance", {})
        check(f"{split}: class balance matches manifest",
              int((y == 1).sum()) == cb.get("pos") and int((y == 0).sum()) == cb.get("neg"),
              f"pos={int((y == 1).sum())} neg={int((y == 0).sum())}")

    # ---- 3. sample-level: re-derive the selection independently.
    # prepare_data.py advances ONE RandomState across splits in PLAN order, so the
    # draws must be replayed in that same order even for splits we do not check.
    rng = np.random.RandomState(loc["subsample_seed"])
    checked_any = False
    for split, yk, cap in PLAN:
        files = sorted(glob.glob(os.path.join(SRC, "data", f"{split}-*.parquet")))
        if not files:
            print(f"[SKIP] {split}: raw shards absent under src/data -- cannot re-derive")
            # Cannot replay this split's draws, so every later split is unverifiable.
            print("[WARN] RNG stream broken at this point; later splits not checked.")
            break
        all_lab = np.concatenate([pq.read_table(f, columns=["label"])["label"].to_numpy()
                                  for f in files])
        n_total = len(all_lab)
        if yk:
            per = cap // 2 if cap else None
            sel = []
            for c in (0, 1):
                idx_c = np.where(all_lab == c)[0]
                take = len(idx_c) if per is None else min(per, len(idx_c))
                sel.append(rng.choice(idx_c, size=take, replace=False))
            sel = np.sort(np.concatenate(sel))
            if yk in z:
                checked_any = True
                check(f"{split}: SAMPLE-LEVEL label vector reproduces",
                      np.array_equal(all_lab[sel], z[yk]), f"n={len(sel)}")
        else:
            take = n_total if cap is None else min(cap, n_total)
            sel = np.sort(rng.choice(n_total, size=take, replace=False))
            check(f"{split}: re-derived selection size matches",
                  len(sel) == loc["splits"][split]["n_used"], f"n={len(sel)}")

    if not checked_any:
        print("[WARN] no sample-level check ran. Size and balance agreement alone "
              "does NOT establish that the cache holds the same samples.")

    check_hospital_cache()

    print(f"\n{len(fails)} failed check(s)" if fails else "\nall checks passed")
    return 1 if fails else 0


def check_hospital_cache():
    """Verify the leave-one-hospital-out splits and the decoded patch cache.

    Covers the failure modes that would silently corrupt a sweep rather than
    crash it: a split that leaks patients across train/test, a decoded array
    that does not cover every row the folds index, and unfilled rows left as
    zeros by an interrupted decode. Pixel fidelity against the source shards is
    NOT checked here (it needs ~21 network reads); run --pixels for that.
    """
    meta_p = os.path.join(CACHE, "camelyon17_metadata.parquet")
    sp_p = os.path.join(CACHE, "hospital_splits.npz")
    if not (os.path.exists(meta_p) and os.path.exists(sp_p)):
        print("\n[SKIP] hospital splits not built (no hospital_splits.npz)")
        return

    print("\n-- hospital splits --")
    import pandas as pd
    m = pd.read_parquet(meta_p)
    sp = np.load(sp_p, allow_pickle=True)
    folds = sorted({k.split("__")[0] for k in sp.keys()})

    for f in folds:
        tr, va, te = (np.asarray(sp[f"{f}__{s}"]).ravel()
                      for s in ("train", "id_val", "test"))
        # Held-out site must be absent from training, and present alone in test.
        tr_c, te_c = set(m.loc[tr, "center"]), set(m.loc[te, "center"])
        check(f"fold {f}: test is one held-out centre",
              te_c == {int(f)}, f"test centres={sorted(te_c)}")
        check(f"fold {f}: held-out centre absent from train",
              int(f) not in tr_c, f"train centres={sorted(tr_c)}")
        # Patient disjointness is the property that makes the design valid.
        p_tr = set(m.loc[np.concatenate([tr, va]), "patient"])
        check(f"fold {f}: no patient in both train/val and test",
              not (p_tr & set(m.loc[te, "patient"])))
        check(f"fold {f}: row indices disjoint across splits",
              len(set(tr) | set(va) | set(te)) == len(tr) + len(va) + len(te))

    idx_p = os.path.join(CACHE, "decoded_96_index.json")
    npy_p = os.path.join(CACHE, "decoded_96.npy")
    if not (os.path.exists(idx_p) and os.path.exists(npy_p)):
        print("[SKIP] decoded patch cache absent -- run decode_patches.py")
        return

    print("-- decoded patch cache --")
    idx = json.load(open(idx_p))
    have = set(int(r) for r in idx["row_ids"])
    need = set()
    for k in sp.keys():
        need |= set(int(x) for x in np.asarray(sp[k]).ravel())
    check("cache covers every row the folds index",
          not (need - have), f"{len(need - have)} missing of {len(need)}")

    arr = np.load(npy_p, mmap_mode="r")
    check("array length matches index",
          len(arr) == len(idx["row_ids"]), f"{len(arr)} vs {len(idx['row_ids'])}")
    # An interrupted decode leaves all-zero rows that train silently as black.
    blank = int((arr.reshape(len(arr), -1).max(axis=1) == 0).sum())
    check("no unfilled (all-zero) patches", blank == 0, f"{blank} blank")


if __name__ == "__main__":
    sys.exit(main())
