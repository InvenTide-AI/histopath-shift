"""Generate v2 job plans, one shard per (tier, cohort, fold) = one SLURM array task.

Phases
  tune   seed 0, PLAN §3 tuning grid.
         Tier S is two-stage: --stage lr   (ERM base LR grid; ViT AdamW LR grid on erm_lunit_dino)
                              --stage arms (method grids at the selected tier-S base LR)
         Tier C: --stage arms (the only stage).
  final  every arm, selected hyperparameters from results/v2/selected_hparams.json,
         seeds 0-4 (tier C) / 0-2 (tier S); inline TTA and PCam (c17) as specified.
  ladder supplementary frozen-probe representation ladder (tier S, 3 seeds). Rungs that
         are also final-phase arms (probe_resnet50_imagenet, probe_resnet50_lunit_bt,
         probe_vit_s16_lunit_dino) write the same records in $V2/runs/S and are left out
         of the ladder plan unless --ladder_all (so final and ladder never race on them).

Hyperparameters are selected PER FOLD (select_hparams.py, --selection per_fold, the
default): fold t uses the config with the best ID-val AUROC on fold t's own tuning run,
and the tier-S stage-2 grids run at fold t's own selected base LR. --selection pooled
uses the cross-fold mean instead (leaks test-domain labels via other folds' ID-val; for
reference only).

Writes  slurm/v2/plans/{phase}[_lr]/{tier}_{cohort}_fold{t}.json  and  shards_{tiers}.txt (one path per line).

  python src/v2/make_plan.py --phase tune --tier C
  python src/v2/make_plan.py --phase tune --tier S --stage lr
  python src/v2/make_plan.py --phase tune --tier S --stage lr_bt      (Barlow Twins LR grid -> plans/tune_lr_bt)
  python src/v2/select_hparams.py           # after the lr stage
  python src/v2/make_plan.py --phase tune --tier S --stage arms
  python src/v2/select_hparams.py
  python src/v2/make_plan.py --phase final --tier C,S
  python src/v2/make_plan.py --phase final --tier S --arms erm_lunit_bt,he_jitter_lunit_bt --name final_bt
  slurm/v2/submit.sh slurm/v2/plans/final/shards_CS.txt [--time 08:00:00]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

REPO = os.path.dirname(os.path.dirname(C.HERE))
PLAN_ROOT = os.path.join(REPO, "slurm", "v2", "plans")
SELECTED = os.path.join(REPO, "results", "v2", "selected_hparams.json")

TIER_C_ARMS = ["erm", "sam", "simclr", "simclr_sam", "augonly", "compute_matched", "groupdro", "irm", "coral",
               "mixstyle", "fish", "lisa", "macenko", "he_jitter", "tia_style", "macenko_tile"]
TIER_S_ARMS = ["erm", "sam", "groupdro", "irm", "coral", "mixstyle", "fish", "lisa", "macenko", "he_jitter",
               "tia_style", "simclr",
               "erm_lunit_bt", "he_jitter_lunit_bt", "erm_lunit_dino", "he_jitter_lunit_dino",
               "probe_resnet50_imagenet", "probe_resnet50_lunit_bt", "probe_vit_s16_lunit_dino", "macenko_tile"]
# macenko_tile (per-tile Macenko, appendix) ran in both tiers via the final_mac plans (no tuning grid:
# default hp in tier C, the fold's selected base LR in tier S)
LADDER = ["probe_resnet18_imagenet", "probe_resnet50_imagenet", "probe_convnext_tiny_imagenet",
          "probe_vit_b16_imagenet", "probe_vit_b16_swag", "probe_resnet50_lunit_bt", "probe_resnet50_lunit_swav",
          "probe_resnet50_lunit_mocov2", "probe_vit_s16_lunit_dino"]
TTA = {"C": {"erm": ["bnadapt", "tent"], "simclr": ["bnadapt"], "he_jitter": ["bnadapt"]},
       "S": {"erm": ["bnadapt"]}}
SEEDS = {"C": [0, 1, 2, 3, 4], "S": [0, 1, 2]}

# PLAN §3 tuning grid (method -> list of hp dicts)
def _grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, v)) for v in itertools.product(*axes.values())]


GRID = {
    "groupdro": _grid(eta=[1e-3, 1e-2, 1e-1]),
    "irm": _grid(irm_lambda=[1.0, 10.0, 100.0]),           # anneal at 1/4 of the budget (default)
    "coral": _grid(coral_lambda=[0.1, 1.0, 10.0]),
    "mixstyle": _grid(ms_p=[0.5, 1.0], ms_alpha=[0.1, 0.3]),
    "fish": _grid(meta_lr=[0.01, 0.05, 0.1, 0.5]),
    "lisa": _grid(lisa_alpha=[0.5, 2.0], p_sel=[0.5, 1.0]),
    "sam": _grid(rho=[0.02, 0.05, 0.1]),
    "he_jitter": _grid(jitter=[0.2, 0.4]),
}
LR_GRID_S = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]  # extended twice: optimum sat on the grid edge (1e-2 on 19/20 folds, then 3e-2 on 15/20)
LR_GRID_VIT = [1e-5, 3e-5, 1e-4]
# Barlow Twins fine-tunes get their own LR (inheriting the ImageNet LR made training fail; app:tuning-bt)
LR_GRID_BT = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
BT_ARMS = ("erm_lunit_bt", "he_jitter_lunit_bt")
# arms that inherit a tuned hyperparameter from another arm (same tier, cohort)
INHERIT = {"simclr_sam": ("sam", ["rho"]), "he_jitter_lunit_bt": ("he_jitter", ["jitter"]),
           "he_jitter_lunit_dino": ("he_jitter", ["jitter"])}
VIT_ARMS = ("erm_lunit_dino", "he_jitter_lunit_dino")


def shard_path(phase_dir, tier, cohort, fold):
    return os.path.join(phase_dir, f"{tier}_{cohort}_fold{fold}.json")


def load_selected(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def _pick(entry, fold, selection):
    """The fold's (or pooled) part of a selected_hparams entry, or None."""
    if entry is None:
        return None
    if selection == "pooled":
        return entry.get("pooled")
    return (entry.get("folds") or {}).get(str(int(fold)))


def base_lr(sel, tier, cohort, arm, allow_default, fold=None, selection="per_fold"):
    """Tier-S shared LR (ERM-tuned) or ViT LR for this fold; {} for tier C."""
    if tier != "S" or arm.startswith("probe_"):
        return {}
    if arm in BT_ARMS:                     # own grid, selected on erm_lunit_bt (stage lr_bt)
        key = "erm_lunit_bt"
        got = _pick(sel.get(tier, {}).get(cohort, {}).get(key), fold, selection)
        got = {"lr": got["hp"]["lr"]} if got else None
    else:
        key = "_vit" if arm in VIT_ARMS else "_base"
        got = _pick(sel.get(tier, {}).get(cohort, {}).get(key), fold, selection)
    if got is None:
        if allow_default:
            return {}
        raise SystemExit(f"no selected {key} LR for {tier}/{cohort}/fold{fold} ({selection}); run the lr stage + "
                         f"select_hparams first (or pass --allow_default)")
    return {"lr": got["lr"]}


def tune_jobs(tier, cohort, stage, sel, allow_default, pcam, fold=None, selection="per_fold"):
    jobs = []
    if tier == "S" and stage == "lr":
        jobs += [dict(arm="erm", seed=0, hp={"lr": lr}) for lr in LR_GRID_S]
        jobs += [dict(arm="erm_lunit_dino", seed=0, hp={"lr": lr}) for lr in LR_GRID_VIT]
    elif tier == "S" and stage == "lr_bt":
        jobs += [dict(arm="erm_lunit_bt", seed=0, hp={"lr": lr}) for lr in LR_GRID_BT]
    else:
        arms = TIER_C_ARMS if tier == "C" else TIER_S_ARMS
        for arm in arms:
            if arm not in GRID:
                continue
            lr = base_lr(sel, tier, cohort, arm, allow_default, fold, selection)
            jobs += [dict(arm=arm, seed=0, hp={**lr, **g}) for g in GRID[arm]]
    for j in jobs:
        j["phase"] = "tune"
        j["pcam"] = False           # PCam is never used for selection
    return jobs


def final_hp(sel, tier, cohort, arm, allow_default, fold=None, selection="per_fold"):
    hp = dict(base_lr(sel, tier, cohort, arm, allow_default, fold, selection))
    src, keys = INHERIT.get(arm, (arm, None))
    if src in GRID:
        got = _pick(sel.get(tier, {}).get(cohort, {}).get(src), fold, selection)
        if got is None:
            if not allow_default:
                raise SystemExit(f"no selected hparams for {tier}/{cohort}/{src}/fold{fold} ({selection}); "
                                 f"run select_hparams.py")
        else:
            h = {k: v for k, v in got["hp"].items() if k != "lr"}
            if keys:
                h = {k: h[k] for k in keys if k in h}
            hp.update(h)
    return hp


def final_jobs(tier, cohort, sel, allow_default, arms=None, phase="final", fold=None, selection="per_fold"):
    jobs = []
    arms = arms or (TIER_C_ARMS if tier == "C" else TIER_S_ARMS)
    for arm in arms:
        hp = final_hp(sel, tier, cohort, arm, allow_default, fold, selection) if phase == "final" else {}
        for s in SEEDS[tier]:
            j = dict(arm=arm, seed=s, hp=hp, tta=TTA[tier].get(arm, []), pcam=cohort == "c17", phase="final")
            if phase == "final" and hp:
                # fold-independent tag (per-fold hp values differ); values are in the record
                j["hpkey"] = "selected" if selection == "per_fold" else None
                j["hp_source"] = f"select_hparams:{selection}"
                if j["hpkey"] is None:
                    del j["hpkey"]
            jobs.append(j)
    # group probes of the same backbone together (features are extracted once per backbone)
    jobs.sort(key=lambda j: (j["arm"].startswith("probe_"), j["arm"] if j["arm"].startswith("probe_") else ""))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["tune", "final", "ladder"])
    ap.add_argument("--stage", default="arms", choices=["lr", "lr_bt", "arms"])
    ap.add_argument("--arms", default=None, help="final phase: comma list of arms (default: all arms of the tier)")
    ap.add_argument("--name", default=None, help="plan directory name (default: derived from phase and stage)")
    ap.add_argument("--tier", default="C,S")
    ap.add_argument("--cohorts", default=",".join(C.COHORTS))
    ap.add_argument("--folds", default=None, help="comma list (default: all folds of each cohort)")
    ap.add_argument("--selected", default=SELECTED)
    ap.add_argument("--plan_root", default=PLAN_ROOT)
    ap.add_argument("--only_existing", action="store_true", help="skip shards whose fold cache is missing")
    ap.add_argument("--cache_root", default=C.V2)
    ap.add_argument("--allow_default", action="store_true", help="use default hp where no selection exists")
    ap.add_argument("--selection", default="per_fold", choices=["per_fold", "pooled"],
                    help="per_fold (default; no cross-fold test-domain leakage) or pooled cross-fold mean")
    ap.add_argument("--ladder_all", action="store_true",
                    help="include ladder rungs that are also final-phase arms (same output files)")
    a = ap.parse_args()
    sel = load_selected(a.selected)
    name = a.name or (a.phase + (f"_{a.stage}" if a.phase == "tune" and "S" in a.tier and a.stage != "arms" else ""))
    phase_dir = os.path.join(a.plan_root, name)
    os.makedirs(phase_dir, exist_ok=True)
    shards, n_jobs = [], 0
    for tier in a.tier.split(","):
        if a.phase == "tune" and a.stage != "arms" and tier == "C":
            continue
        for cohort in a.cohorts.split(","):
            folds = [int(f) for f in a.folds.split(",")] if a.folds else range(C.N_FOLDS[cohort])
            for t in folds:
                if a.only_existing and not os.path.exists(C.fold_dir(a.cache_root, tier, cohort, t)):
                    continue
                if a.phase == "tune":
                    jobs = tune_jobs(tier, cohort, a.stage, sel, a.allow_default, False, t, a.selection)
                elif a.phase == "final":
                    jobs = final_jobs(tier, cohort, sel, a.allow_default, fold=t, selection=a.selection,
                                      arms=a.arms.split(",") if a.arms else None)
                else:
                    if tier != "S":
                        continue
                    rungs = LADDER if a.ladder_all else [x for x in LADDER if x not in TIER_S_ARMS]
                    jobs = final_jobs(tier, cohort, sel, True, arms=rungs, phase="ladder")
                if not jobs:
                    continue
                p = shard_path(phase_dir, tier, cohort, t)
                json.dump(dict(tier=tier, cohort=cohort, fold=t, phase="tune" if a.phase == "tune" else "final",
                               jobs=jobs), open(p, "w"), indent=1)
                shards.append(p)
                n_jobs += len(jobs)
    sh = os.path.join(phase_dir, "shards_%s.txt" % a.tier.replace(",", ""))
    with open(sh, "w") as f:
        f.write("\n".join(shards) + ("\n" if shards else ""))
    print(f"{len(shards)} shards, {n_jobs} jobs -> {sh}")


if __name__ == "__main__":
    main()
