"""Select hyperparameters from the tuning runs (seed 0; never test).

Primary rule (default, `per_fold`): for every tier x cohort x arm x fold, the config with
the highest ID-val AUROC at the ID-selected checkpoint of THAT fold's own tuning run
(DomainBed's training-domain validation: one selection per test environment). This is
what make_plan.py uses by default.

Why not the cross-fold mean (PLAN §2 as first written): in leave-one-domain-out, fold t's
test domain is a training domain of most other folds, so their ID-val sets contain
labelled patches of fold t's test domain (pixel-identical rows on c17 / MIDOG++ /
canine). Averaging ID-val AUROC over folds therefore lets fold t's test labels influence
the hyperparameters used to report fold t. The cross-fold mean is still computed and
stored under `pooled` (with `leaks_test_domain: true`) for reference only.

Diverged runs (record status 'diverged') are never selected in a fold unless every config
of that fold diverged (then the entry is flagged `all_diverged`).

Tier S: `_base` = LR of the selected `erm` config (shared by all tier-S CNN arms),
        `_vit`  = LR of the selected `erm_lunit_dino` config (shared by the ViT arms),
        both per fold (and pooled for reference).

Output structure (results/v2/selected_hparams.json):
  sel[tier][cohort][arm] = {"rule": "per_fold_id_val",
                            "folds": {"<t>": {"hp": {...}, "id_val_auroc": x, "n_configs": n,
                                              "n_diverged": n, "all_diverged": bool, "all": {hp: auroc}}},
                            "pooled": {"hp": {...}, "mean_id_val_auroc": x, "n_folds": n, "incomplete": bool,
                                       "leaks_test_domain": true, "all": {...}}}
  sel[tier][cohort]["_base" | "_vit"] = {"from": arm, "folds": {"<t>": {"lr": x}}, "pooled": {"lr": x}}
Also writes results/v2/tuning_results.csv (one row per tuning run).
  python src/v2/select_hparams.py [--runs_root $V2/runs_tune]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from make_plan import SELECTED, GRID  # noqa: E402

GRID_KEYS = {k for g in GRID.values() for d in g for k in d} | {"lr"}


def tuned_part(arm, hp):
    """The hyperparameters that were varied in tuning for this arm (+ lr)."""
    method = arm
    keys = {k for d in GRID.get(method, []) for k in d} | {"lr"}
    return {k: hp[k] for k in sorted(keys) if k in hp}


def load_rows(runs_root):
    rows = []
    for f in sorted(glob.glob(os.path.join(runs_root, "*", "*", "fold*", "*.json"))):
        r = json.load(open(f))
        if r.get("schema") != "v2" or r.get("seed") != 0:
            continue
        sid = r["selected"]["id"]
        ck = next(c for c in r["checkpoints"] if c["step"] == sid)
        rows.append(dict(tier=r["tier"], cohort=r["cohort"], fold=int(r["fold"]), arm=r["arm"], hpkey=r["hpkey"],
                         hp=json.dumps(tuned_part(r["arm"], r["hparams"]), sort_keys=True),
                         id_val_auroc=ck["id_val"]["auroc"], selected_step=sid,
                         status=r.get("status", "ok"),
                         cache_fingerprint=r.get("cache_fingerprint"),
                         ood_val_auroc_at_id=(ck["ood_val"] or {}).get("auroc"),
                         ood_test_auroc_at_id=r["test_at"]["id"]["auroc"], path=f))
    return rows


def pick(g):
    """Best config of a group of runs (one row per config): highest id_val AUROC among
    non-diverged runs; ties -> lexicographically smallest hp string."""
    ok = g[g.status != "diverged"]
    use = ok if len(ok) else g
    use = use.assign(_a=use.id_val_auroc.fillna(-1.0))
    best = use.sort_values(["_a", "hp"], ascending=[False, True]).iloc[0]
    return best, len(ok) == 0


def select(df):
    sel = {}
    for (tier, cohort, arm), g in df.groupby(["tier", "cohort", "arm"]):
        nf = C.N_FOLDS.get(cohort, g.fold.nunique())
        folds = {}
        for t, gf in g.groupby("fold"):
            gf = gf.drop_duplicates("hp", keep="last")
            best, all_div = pick(gf)
            folds[str(int(t))] = dict(hp=json.loads(best.hp), id_val_auroc=float(best.id_val_auroc),
                                      n_configs=int(len(gf)), n_diverged=int((gf.status == "diverged").sum()),
                                      all_diverged=bool(all_div),
                                      all={h: (float(a) if a == a else None) for h, a in zip(gf.hp, gf.id_val_auroc)})
        # pooled cross-fold mean (reference only): configs diverged in any fold are excluded
        div_hp = set(g.hp[g.status == "diverged"])
        agg = g.groupby("hp").agg(mean=("id_val_auroc", "mean"), n=("fold", "nunique")).reset_index()
        complete = agg[(agg.n == nf) & ~agg.hp.isin(div_hp)]
        use = complete if len(complete) else agg
        bp = use.sort_values(["mean", "hp"], ascending=[False, True]).iloc[0]
        pooled = dict(hp=json.loads(bp.hp), mean_id_val_auroc=float(bp["mean"]), n_folds=int(bp.n),
                      n_configs=int(len(agg)), incomplete=bool(len(complete) == 0), leaks_test_domain=True,
                      diverged_configs=sorted(div_hp),
                      all={h: dict(mean=float(m), n=int(n)) for h, m, n in zip(agg.hp, agg["mean"], agg.n)})
        entry = dict(rule="per_fold_id_val", folds=folds, pooled=pooled, n_folds_expected=int(nf),
                     missing_folds=sorted(set(range(nf)) - {int(t) for t in folds}))
        sel.setdefault(tier, {}).setdefault(cohort, {})[arm] = entry
        for src, key in (("erm", "_base"), ("erm_lunit_dino", "_vit")):
            if tier == "S" and arm == src:
                sel[tier][cohort][key] = {"from": src, "folds": {t: {"lr": v["hp"]["lr"]} for t, v in folds.items()},
                                          "pooled": {"lr": pooled["hp"]["lr"]}}
        miss = f" MISSING folds {entry['missing_folds']}" if entry["missing_folds"] else ""
        print(f"{tier} {cohort:8s} {arm:16s} per-fold: "
              + "; ".join(f"f{t}={v['hp']}{' (ALL DIVERGED)' if v['all_diverged'] else ''}"
                          for t, v in sorted(folds.items())) + miss)
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default=os.path.join(C.V2, "runs_tune"))
    ap.add_argument("--out", default=SELECTED)
    ap.add_argument("--csv", default=os.path.join(os.path.dirname(SELECTED), "tuning_results.csv"))
    a = ap.parse_args()
    rows = load_rows(a.runs_root)
    import pandas as pd
    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("no tuning runs found under", a.runs_root)
        return
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.csv, index=False)
    old = json.load(open(a.out)) if os.path.exists(a.out) else {}
    new = select(df)
    for tier, d in new.items():          # merge (the tier-S lr stage and arms stage are separate runs)
        for cohort, e in d.items():
            old.setdefault(tier, {}).setdefault(cohort, {}).update(e)
    C.atomic_json(old, a.out)
    print("wrote", a.out, "and", a.csv)


if __name__ == "__main__":
    main()
