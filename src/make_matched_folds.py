#!/usr/bin/env python3
"""Build protocol-matched leave-one-hospital-out folds for Camelyon17-WILDS.

Emits index sets only (no pixels): protocol_matched_folds.npz + .json.
Indices are row positions in camelyon17_metadata.parquet.

Design (matches the published center-2 run):
  test center t ; ood_val center v = (t-1) mod 5 ; train = the other three.
  Epoch selection happens on ood_val (a held-out HOSPITAL), so every fold
  needs three hospital roles.  t=2 reproduces the published fold exactly
  (train {0,3,4}, ood_val 1, test 2).

Role sizes follow prepare_data.py's PLAN: 60000/12000/12000/25000 labelled
(runs subsample train to 30000 via --ntrain) and 40000 unlabelled for SSL.
"""
import json
import numpy as np, pandas as pd

MD = "camelyon17_metadata.parquet"
TARGET = {"train": 60000, "id_val": 12000, "ood_val": 12000, "ood_test": 25000}
UNLABELED, SEED, USEED = 40000, 20240, 90240

md = pd.read_parquet(MD)
assert len(md) == 455954 and md.index.is_monotonic_increasing


def balanced(pool, n, rs):
    """n/2 patches of each class, drawn without replacement from `pool`."""
    pool = np.asarray(pool)
    lab = md.loc[pool, "label"].values
    return np.sort(np.concatenate([rs.permutation(pool[lab == c])[: n // 2] for c in (0, 1)]))


folds, manifest = {}, {}
for t in range(5):
    v = (t - 1) % 5
    trc = [c for c in range(5) if c not in (t, v)]
    rs = np.random.RandomState(SEED + t)

    # id_val patients: round-robin across training centres until BOTH classes
    # can fill 6000 slots and every training centre is represented.  Patients
    # are not individually class-balanced, so a fixed k-per-centre holdout
    # cannot guarantee this.
    per_c = {c: list(rs.permutation(np.sort(md.loc[md.center == c, "patient"].unique()))) for c in trc}
    order = []
    while any(per_c.values()):
        for c in trc:
            if per_c[c]:
                order.append((c, per_c[c].pop(0)))
    iv = []
    for c, p in order:
        iv.append((c, p))
        s = set(iv)
        sub = md[[(cc, pp) in s for cc, pp in zip(md.center, md.patient)]]
        if sub.label.value_counts().min() >= TARGET["id_val"] // 2 and len({c for c, _ in iv}) == len(trc):
            break
    iv_set = set(iv)
    m_iv = np.array([(cc, pp) in iv_set for cc, pp in zip(md.center, md.patient)])

    folds[f"t{t}__train"]    = balanced(md.index[md.center.isin(trc) & ~m_iv], TARGET["train"], rs)
    folds[f"t{t}__id_val"]   = balanced(md.index[m_iv], TARGET["id_val"], rs)
    folds[f"t{t}__ood_val"]  = balanced(md.index[md.center == v], TARGET["ood_val"], rs)
    folds[f"t{t}__ood_test"] = balanced(md.index[md.center == t], TARGET["ood_test"], rs)

    # unlabelled SSL pool: training centres only, disjoint from labelled train
    # and from id_val patients, so neither the test nor the val hospital leaks.
    ur = np.random.RandomState(USEED + t)
    pool = np.setdiff1d(md.index[md.center.isin(trc) & ~m_iv].values, folds[f"t{t}__train"])
    folds[f"t{t}__unlabeled"] = np.sort(ur.permutation(pool)[:UNLABELED])

    manifest[f"t{t}"] = dict(test_center=t, ood_val_center=v, train_centers=trc,
                             id_val_patients=[[int(c), int(p)] for c, p in sorted(iv)])

# ---- verification (all of these must hold for every fold) -------------------
for t in range(5):
    R = {r: folds[f"t{t}__{r}"] for r in TARGET}
    for r, ix in R.items():
        assert len(ix) == TARGET[r], (t, r, len(ix))
        assert abs(md.loc[ix, "label"].mean() - 0.5) < 1e-9, (t, r)
    u = folds[f"t{t}__unlabeled"]
    assert len(u) == UNLABELED
    allix = np.concatenate(list(R.values()) + [u])
    assert len(set(allix.tolist())) == len(allix), f"index reuse in t{t}"
    pats = {r: set(zip(md.loc[ix, "center"], md.loc[ix, "patient"])) for r, ix in {**R, "unlabeled": u}.items()}
    for a in pats:
        for b in pats:
            if a < b and not {a, b} <= {"train", "unlabeled"}:
                assert not (pats[a] & pats[b]), (t, a, b)
    assert set(md.loc[R["ood_test"], "center"]) == {t}
    assert set(md.loc[R["ood_val"], "center"]) == {(t - 1) % 5}
    assert not ({t, (t - 1) % 5} & set(md.loc[R["train"], "center"]))
    assert not ({t, (t - 1) % 5} & set(md.loc[u, "center"]))
assert manifest["t2"]["train_centers"] == [0, 3, 4] and manifest["t2"]["ood_val_center"] == 1

np.savez_compressed("protocol_matched_folds.npz", **folds)
sh = md.reset_index().groupby("shard")["index"].agg(["min", "max", "count"]).sort_values("min")
assert ((sh["max"] - sh["min"] + 1) == sh["count"]).all()
json.dump(dict(target_sizes={**TARGET, "unlabeled": UNLABELED}, seed_base=SEED, unlabeled_seed_base=USEED,
               rotation="ood_val = (test-1) mod 5",
               note="t2 reproduces the published design: train {0,3,4}, ood_val 1, test 2",
               global_index="row position in camelyon17_metadata.parquet; shards are contiguous blocks",
               shard_order=list(sh.index), folds=manifest),
          open("protocol_matched_folds.json", "w"), indent=1)
print("wrote protocol_matched_folds.npz (%d index sets); all assertions passed" % len(folds))
