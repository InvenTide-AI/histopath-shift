#!/usr/bin/env python
"""analyze_v2 -- CLI for the v2 statistics (PLAN.md §4) on v2 run records (CONTRACTS.md B).

Sub-commands
------------
runs       analyse $V2/runs/{tier}/{cohort}/fold*/*.json  ->  {out}/{tier}/{cohort}/*  and  {out}/{tier}/cross/*
v1         same analysis on the v1 record layout (runs/fold*/{arm}_seed{s}.json); for regression / demo only
domainbed  parse the per-environment tables of Gulrajani & Lopez-Paz (arXiv 2007.01434) from the e-print
           LaTeX and compute per-algorithm per-dataset sigma_env, sigma_run, ICC (external ICC example)

Examples
--------
module load PyTorch/2.7.0-CONDA
python src/v2/analyze_v2.py runs --tier C --select id                        # all cohorts present
python src/v2/analyze_v2.py runs --tier S --select ood --cohorts c17 canine midogpp
python src/v2/analyze_v2.py v1 --runs $HISTOPATH_DATA/runs --cohort c17 --out /tmp/v1demo
python src/v2/analyze_v2.py domainbed

Outputs per (tier, cohort) in {out}/{tier}/{cohort}/ (``{sel}`` = selection rule, default 'id'):
  per_run_{sel}.csv          one row per run (arm, fold, domain, seed, metric, ece)
  summary_{sel}.{csv,json}   §4a per-arm summaries with bootstrap CIs
  variance_{sel}.{csv,json}  §4b ANOVA / REML / exact ICC CI / MLS & chi2 CIs / Bayesian posteriors
  contrasts_{sel}.csv        §4c paired contrasts vs --ref, Holm, fold-dependence band
  power_{sel}.csv            §4d difference-variance components and K_min table
  ranks_{sel}.csv            §4e rank + bootstrap rank intervals (worst / mean; seed / domain)
  response_{sel}.csv         §4f Oldham / Pitman-Morgan vs --ref
  tables/*.tex               booktabs snippets (tabular only)
Cross-cohort ({out}/{tier}/cross/): kendall_{sel}.csv, interaction_{sel}.json, clustering_{sel}.json, kendall_{sel}.tex

Notes
  * --lower-better (ECE, NLL, Brier): 'worst' = max over domains everywhere (summaries, bootstrap CIs,
    ranks, Kendall tau on 'worst'), and 'improved' = d < 0 in the contrasts.
  * Tier S: the supplementary representation-ladder probes (make_plan.LADDER) are written as phase
    'final' into the same tree; arms not in make_plan.TIER_S_ARMS (+ their TTA pseudo-arms) are dropped
    unless --ladder include (keep them) or --ladder only (analyse only the ladder) is given.
  * Power rows use the seed-paired difference variance (difference_variance(paired=True)) whenever arm
    and reference have identical seed sets on every fold (TTA pseudo-arms, same-seed training runs).
  * v1 sub-command outputs are labelled sel='v1rule' (the v1 selection rule), never 'id'.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import warnings
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats_v2 as S  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
V2 = os.path.join(os.environ.get("HISTOPATH_DATA", "data"), "v2")
COHORTS = ["c17", "canine", "midog21sn", "midogpp"]
try:  # single source of truth for the arm lists (training track's planner)
    import make_plan as _MP  # noqa: E402
    TIER_S_ARMS, LADDER = list(_MP.TIER_S_ARMS), list(_MP.LADDER)
except Exception:  # noqa: BLE001  (fallback copy of make_plan.py as of 2026-10-05)
    TIER_S_ARMS = ["erm", "sam", "groupdro", "irm", "coral", "mixstyle", "fish", "lisa", "macenko", "he_jitter",
                   "tia_style", "simclr", "erm_lunit_bt", "he_jitter_lunit_bt", "erm_lunit_dino", "he_jitter_lunit_dino",
                   "probe_resnet50_imagenet", "probe_resnet50_lunit_bt", "probe_vit_s16_lunit_dino"]
    LADDER = ["probe_resnet18_imagenet", "probe_resnet50_imagenet", "probe_convnext_tiny_imagenet",
              "probe_vit_b16_imagenet", "probe_vit_b16_swag", "probe_resnet50_lunit_bt", "probe_resnet50_lunit_swav",
              "probe_resnet50_lunit_mocov2", "probe_vit_s16_lunit_dino"]
RHOS = (0.0, 0.25, 0.5, 1.0)

# ----------------------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------------------


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, set):
        return sorted(_jsonable(v) for v in o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not math.isfinite(v) else v
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_jsonable(obj), f, indent=1)


def write_csv(path, rows, keys=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        open(path, "w").close()
        return
    keys = keys or list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt_cell(r.get(k)) for k in keys})


def _fmt_cell(v):
    if isinstance(v, (float, np.floating)):
        return "" if not math.isfinite(float(v)) else repr(float(v))
    if isinstance(v, (list, tuple)):
        return json.dumps(_jsonable(v))
    return v


# ----------------------------------------------------------------------------------------
# loading v2 / v1 records into  {arm: {fold: {seed: value}}}
# ----------------------------------------------------------------------------------------


class Cohort:
    """Container: per-arm per-fold per-seed metric values plus fold design."""

    def __init__(self, name):
        self.name = name
        self.vals = defaultdict(lambda: defaultdict(dict))   # arm -> fold -> seed -> metric
        self.ece = defaultdict(lambda: defaultdict(dict))
        self.domain = {}       # fold -> test domain name
        self.val = {}          # fold -> val domain name or None
        self.train = {}        # fold -> set of training-domain names (from manifest if present)
        self.rows = []
        self.notes = []

    def folds(self):
        return sorted(self.domain)

    def matrix(self, arm, which="vals"):
        """List of per-fold arrays (fold order); None if the arm misses a fold."""
        src = self.vals if which == "vals" else self.ece
        out = []
        for f in self.folds():
            d = src[arm].get(f, {})
            v = [d[s] for s in sorted(d) if d[s] is not None and math.isfinite(d[s])]
            if not v:
                return None
            out.append(np.array(v, float))
        return out

    def seeds(self, arm):
        """Per-fold sorted tuple of seeds with a finite value (fold order)."""
        return [tuple(s for s in sorted(self.vals[arm].get(f, {})) if self.vals[arm][f][s] is not None
                      and math.isfinite(self.vals[arm][f][s])) for f in self.folds()]

    def design(self):
        """Training-domain sets per fold (indices into folds()); manifest > all-minus-{test,val}."""
        names = [self.domain[f] for f in self.folds()]
        idx = {n: i for i, n in enumerate(names)}
        sets = []
        for f in self.folds():
            if f in self.train and self.train[f]:
                tr = {idx[n] for n in self.train[f] if n in idx}
                # training domains that are not test domains of any fold are still overlap:
                extra = [n for n in self.train[f] if n not in idx]
                if extra:
                    tr |= {f"x:{n}" for n in extra}
            else:
                excl = {self.domain[f]} | ({self.val[f]} if self.val.get(f) else set())
                tr = {idx[n] for n in names if n not in excl}
            sets.append(tr)
        return sets


def _hpkey_from_tag(tag):
    parts = tag.split("__")
    return parts[1] if len(parts) >= 3 else "default"


def _base_arm(label):
    """'bnadapt@erm[hp]' -> 'erm'; 'sam[rho0.05]' -> 'sam'."""
    return label.split("@", 1)[-1].split("[", 1)[0]


def load_v2(root, tier, cohort, select, metric, phase="final", include_tta=True, ladder="exclude"):
    """ladder (tier S only): 'exclude' drops arms outside TIER_S_ARMS (the ladder-only probes),
    'include' keeps everything, 'only' keeps just the LADDER arms."""
    C = Cohort(cohort)
    files = sorted(glob.glob(os.path.join(root, "runs", tier, cohort, "fold*", "*.json")))
    seen_hp = defaultdict(set)
    recs = []
    for fp in files:
        try:
            r = json.load(open(fp))
        except Exception as e:  # noqa: BLE001
            C.notes.append(f"unreadable {fp}: {e}")
            continue
        if r.get("schema") != "v2":
            continue
        if phase and r.get("phase", "final") != phase:
            continue
        tag = os.path.basename(fp)[:-5]
        if tier == "S" and ladder != "include":
            keep = r.get("arm") in (TIER_S_ARMS if ladder == "exclude" else LADDER)
            if not keep:
                C.notes.append(f"ladder={ladder}: skipped {r.get('arm')} ({tag})")
                continue
        recs.append((fp, tag, r))
        seen_hp[r["arm"]].add(_hpkey_from_tag(tag))
    missing = defaultdict(int)
    for fp, tag, r in recs:
        arm = r["arm"]
        hp = _hpkey_from_tag(tag)
        label = arm if len(seen_hp[arm]) == 1 else f"{arm}[{hp}]"
        f = int(r["fold"])
        C.domain[f] = str(r.get("test_domain", f))
        C.val[f] = r.get("val_domain")
        man = os.path.join(root, f"cache{r.get('input_res', 64)}", cohort, f"fold{f}", "manifest.json")
        if f not in C.train and os.path.exists(man):
            try:
                C.train[f] = [str(x) for x in json.load(open(man)).get("train_domains", [])]
            except Exception:  # noqa: BLE001
                pass
        seed = int(r["seed"])
        ta = (r.get("test_at") or {}).get(select)
        if not ta or metric not in ta:
            missing[f"no test_at.{select}.{metric}"] += 1
        else:
            C.vals[label][f][seed] = float(ta[metric])
            C.ece[label][f][seed] = float(ta.get("ece", float("nan")))
            C.rows.append(dict(arm=label, fold=f, domain=C.domain[f], seed=seed, metric=float(ta[metric]),
                               ece=ta.get("ece"), file=os.path.relpath(fp, root)))
        if include_tta and select == "id":
            for tname, tm in (r.get("tta") or {}).items():
                if tm and metric in tm:
                    tl = f"{tname}@{label}"
                    C.vals[tl][f][seed] = float(tm[metric])
                    C.ece[tl][f][seed] = float(tm.get("ece", float("nan")))
                    C.rows.append(dict(arm=tl, fold=f, domain=C.domain[f], seed=seed, metric=float(tm[metric]),
                                       ece=tm.get("ece"), file=os.path.relpath(fp, root)))
    for k, v in missing.items():
        C.notes.append(f"{v} records with {k}")
    for arm, hps in seen_hp.items():
        if len(hps) > 1:
            C.notes.append(f"arm {arm} has several final hpkeys {sorted(hps)}: analysed separately")
    return C


V1_ARM_MAP = {"baseline": "erm"}


def load_v1(runs, cohort, metric="auroc", val_offset=-1):
    """v1 layout: runs/fold{t}/{arm}_seed{s}.json with ood_test.{metric}.  The v1 records were
    selected on OOD val (C17/canine) or ID val (MIDOG), so this is a demo/regression only."""
    C = Cohort(cohort)
    pat = re.compile(r"^(?P<arm>.+)_seed(?P<s>\d+)\.json$")
    for fp in sorted(glob.glob(os.path.join(runs, "fold*", "*.json"))):
        m = pat.match(os.path.basename(fp))
        if not m:
            continue
        f = int(re.search(r"fold(\d+)", fp).group(1))
        r = json.load(open(fp))
        ot = r.get("ood_test") or {}
        if metric not in ot:
            continue
        arm = V1_ARM_MAP.get(m["arm"], m["arm"])
        C.domain[f] = f"d{f}"
        C.vals[arm][f][int(m["s"])] = float(ot[metric])
        C.ece[arm][f][int(m["s"])] = float(ot.get("ece", float("nan")))
        C.rows.append(dict(arm=arm, fold=f, domain=C.domain[f], seed=int(m["s"]), metric=float(ot[metric]),
                           ece=ot.get("ece"), file=os.path.relpath(fp, runs)))
    K = len(C.domain)
    for f in C.folds():
        C.val[f] = f"d{(f + val_offset) % K}" if val_offset is not None else None
    C.notes.append("v1 records (selection: v1 rule), domains labelled d{fold}")
    return C


# ----------------------------------------------------------------------------------------
# per-cohort analysis
# ----------------------------------------------------------------------------------------


def complete_arms(C, min_seeds=1, keep=None, drop=None):
    out = {}
    for arm in sorted(C.vals):
        if (keep and arm not in keep) or (drop and arm in drop):
            continue
        g = C.matrix(arm)
        if g is None:
            C.notes.append(f"arm {arm} missing a fold -> excluded")
            continue
        if min(r.size for r in g) < min_seeds:
            C.notes.append(f"arm {arm} has < {min_seeds} seeds on some fold -> excluded")
            continue
        out[arm] = g
    return out


def analyse_cohort(C, outdir, sel, ref="erm", B=10000, Bperm=10000, prior_scale=0.1, min_seeds=1,
                   higher_better=True, keep=None, drop=None, seeds_table=(1, 3, 5, 10), deltas=(0.02, 0.05, 0.10, 0.20),
                   snapshot=None):
    os.makedirs(os.path.join(outdir, "tables"), exist_ok=True)
    arms = complete_arms(C, min_seeds, keep, drop)
    names = [C.domain[f] for f in C.folds()]
    design = C.design()
    has_val = any(C.val.get(f) for f in C.folds())
    write_csv(os.path.join(outdir, f"per_run_{sel}.csv"), C.rows)
    res = dict(cohort=C.name, selection=sel, K=len(names), domains=names, design=[sorted(map(str, s)) for s in design],
               notes=C.notes, arms=list(arms))
    # ---- a) summaries
    summ_rows, summ = [], {}
    hb = higher_better
    for arm, g in arms.items():
        s = S.arm_summary(g, names, higher_better=hb)
        b = S.bootstrap_summary(g, B=B, seed=0, higher_better=hb)
        e = C.matrix(arm, "ece")
        em = np.array([np.nanmean(r) for r in e]) if e is not None else np.full(len(names), np.nan)
        s.update(boot=b, ece_mean=float(np.nanmean(em)), ece_worst=float(np.nanmax(em)) if np.isfinite(em).any() else None,
                 ece_domain_mean=em.tolist())
        summ[arm] = s
        summ_rows.append(dict(arm=arm, K=s["K"], seeds=min(s["n_seeds"]), mean=s["mean"], mean_ci_t=s["mean_ci_t"],
                              worst=s["worst"],
                              worst_domain=s["worst_domain"], sd_across=s["sd_across"], range=s["range"],
                              mean_within_sd=s["mean_within_sd"],
                              worst_ci_seed=b["seed"]["worst_ci"], mean_ci_seed=b["seed"]["mean_ci"],
                              worst_ci_domain=b["domain"]["worst_ci"], mean_ci_domain=b["domain"]["mean_ci"],
                              worst_ci_two_stage=b["two_stage"]["worst_ci"], mean_ci_two_stage=b["two_stage"]["mean_ci"],
                              ece_mean=s["ece_mean"], ece_worst=s["ece_worst"],
                              **{f"dom_{n}": v for n, v in zip(names, s["domain_mean"])}))
    write_csv(os.path.join(outdir, f"summary_{sel}.csv"), summ_rows)
    write_json(os.path.join(outdir, f"summary_{sel}.json"), summ)
    res["summary"] = summ
    # ---- b) variance components
    var_rows, var = [], {}
    priors = (("halfnormal", prior_scale), ("halfcauchy", prior_scale))
    for arm, g in arms.items():
        if sum(r.size for r in g) - len(g) < 1:
            continue
        v = S.variance_components(g, priors=priors)
        var[arm] = v
        a, r, c = v["anova"], v["reml"], v["icc_ci"]
        row = dict(arm=arm, K=a["k"], N=a["N"], sigma_domain_anova=a["sigma_domain"], sigma_run_anova=a["sigma_run"],
                   s2_domain_raw=a["s2_domain_raw"], icc_anova=a["icc"], icc_raw=a["icc_raw"], F=a["F"], p_F=a["p"],
                   sigma_domain_reml=r["sigma_domain"], sigma_run_reml=r["sigma_run"], icc_reml=r["icc"],
                   reml_boundary=r["boundary"], anova_truncated=a["truncated"], icc_ci_lo=c["lo"], icc_ci_hi=c["hi"], icc_ci_exact=c["exact"],
                   sigma_run_ci=v["sigma_run_ci"], sigma_domain_ci_mls=v["sigma_domain_ci"])
        for pk, pb in v["bayes"].items():
            tag = pk.split("(")[0][:5]
            for q in ("sigma_domain", "sigma_run", "icc"):
                row[f"{tag}_{q}_med"] = pb[q]["median"]
                row[f"{tag}_{q}_ci"] = [pb[q]["lo"], pb[q]["hi"]]
                row[f"{tag}_{q}_u95"] = pb[q]["upper_one_sided"]
        var_rows.append(row)
    write_csv(os.path.join(outdir, f"variance_{sel}.csv"), var_rows)
    write_json(os.path.join(outdir, f"variance_{sel}.json"), var)
    res["variance"] = var
    # ---- c) contrasts vs ref, d) power, f) response shape
    con_rows, pow_rows, resp_rows = [], [], []
    if ref in arms:
        others = [a for a in arms if a != ref]
        cons = {a: S.paired_contrast(arms[a], arms[ref], higher_better=hb) for a in others}
        padj = S.holm([cons[a]["p"] for a in others]) if others else []
        for a, ph in zip(others, padj):
            c = cons[a]
            row = dict(arm=a, ref=ref, K=c["K"], mean_diff=c["mean"], sd_diff=c["sd"], ci=c["ci"], t=c["t"], p=c["p"],
                       p_holm=ph, n_improved=c["n_improved"], n_worse=c["n_worse"])
            for sr in S.dependence_sensitivity(c["d"], design, RHOS):
                k = f"rho{sr['rho']:g}"
                row[f"{k}_Keff"], row[f"{k}_ci"], row[f"{k}_p"] = sr["K_eff"], sr["ci"], sr["p"]
            con_rows.append(row)
            # seed-paired run noise when arm and ref share seeds on every fold (TTA pseudo-arms reuse the
            # reference's models; same-seed training runs share init and batch order)
            sa, sr_ = C.seeds(a), C.seeds(ref)
            paired = sa == sr_ and min(len(x) for x in sa) >= 2
            if paired:
                comp = S.difference_variance(arms[a], arms[ref], paired=True)
            else:
                comp = S.difference_variance(arms[a], arms[ref])
                if _base_arm(a) == ref and "@" in a:
                    C.notes.append(f"{a}: seeds not aligned with {ref}; power uses the independent-noise formula")
            for pr in S.power_table(comp, deltas=deltas, seeds=seeds_table, rhos=(0.0, 0.25, 0.5),
                                    val_offset=-1 if has_val else None, train_sets=design):
                pow_rows.append(dict(arm=a, ref=ref, paired=comp["paired"], sd_d_observed=comp["sd_d_observed"],
                                     sigma_dxa=math.sqrt(comp["s2_dxa"]), sigma_dxa_raw2=comp["s2_dxa_raw"],
                                     sigma_run_arm=math.sqrt(comp["s2_run_arm"]),
                                     sigma_run_ref=math.sqrt(comp["s2_run_ref"]),
                                     sigma_run_diff=math.sqrt(comp["s2_run_diff"]), run_corr=comp["s2_run_corr"], **pr))
            o = S.oldham_pitman_morgan(S.domain_means(arms[a]), S.domain_means(arms[ref]))
            resp_rows.append(dict(arm=a, ref=ref, **o))
    else:
        C.notes.append(f"reference arm {ref!r} not available -> no contrasts / power / response shape")
    write_csv(os.path.join(outdir, f"contrasts_{sel}.csv"), con_rows)
    write_csv(os.path.join(outdir, f"power_{sel}.csv"), pow_rows)
    write_csv(os.path.join(outdir, f"response_{sel}.csv"), resp_rows)
    res.update(contrasts=con_rows, power=pow_rows, response=resp_rows)
    # ---- e) ranks within cohort
    rank_rows = []
    if len(arms) >= 2:  # 'worst' = min (higher_better) or max (lower_better) over domains
        rk = {}
        for key in ("worst", "mean"):
            for mode in ("seed", "domain"):
                rk[(key, mode)] = S.bootstrap_ranks(arms, key, mode, B=min(B, 5000), higher_better=higher_better)
        for a in arms:
            row = dict(arm=a, worst=summ[a]["worst"], mean=summ[a]["mean"])
            for (key, mode), d in rk.items():
                row[f"rank_{key}"] = d[a]["rank"]
                row[f"rank_{key}_{mode}_ci"] = [d[a]["lo"], d[a]["hi"]]
                row[f"pbest_{key}_{mode}"] = d[a]["p_best"]
            rank_rows.append(row)
        rank_rows.sort(key=lambda r: r["rank_worst"])
    write_csv(os.path.join(outdir, f"ranks_{sel}.csv"), rank_rows)
    res["ranks"] = rank_rows
    write_tables(outdir, sel, C.name, summ_rows, var_rows, con_rows, pow_rows, rank_rows, higher_better=hb)
    write_json(os.path.join(outdir, f"meta_{sel}.json"), dict(cohort=C.name, selection=sel, K=len(names), domains=names,
                                                              design=res["design"], arms=list(arms), notes=C.notes,
                                                              ref=ref, B=B, prior_scale=prior_scale,
                                                              higher_better=hb, snapshot=snapshot))
    return res, arms


# ----------------------------------------------------------------------------------------
# cross-cohort analysis
# ----------------------------------------------------------------------------------------


def analyse_cross(cohort_arms, outdir, sel, ref="erm", B=2000, Bperm=10000, higher_better=True):
    names = [c for c in COHORTS if c in cohort_arms] + [c for c in cohort_arms if c not in COHORTS]
    rows, pairs = [], {}
    for key in ("worst", "mean"):
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                common = [x for x in cohort_arms[a] if x in cohort_arms[b]]
                if len(common) < 3:
                    continue
                t = S.tau_between_cohorts({x: cohort_arms[a][x] for x in common}, {x: cohort_arms[b][x] for x in common},
                                          key, B=B, return_draws=True, higher_better=higher_better)
                if key == "worst":
                    pairs[(a, b)] = t
                rows.append(dict(key=key, cohort_a=a, cohort_b=b, n_arms=t["n_arms"], tau_b=t["tau"], ci=t["ci"],
                                 boot_median=t["boot_median"], p=t["p"],
                                 same_task=S.COHORT_META.get(a, {}).get("task") == S.COHORT_META.get(b, {}).get("task"),
                                 same_shift=S.COHORT_META.get(a, {}).get("shift") == S.COHORT_META.get(b, {}).get("shift")))
    write_csv(os.path.join(outdir, f"kendall_{sel}.csv"), rows)
    clus = S.cluster_summary(pairs) if pairs and all(k in S.COHORT_META for p in pairs for k in p) else {}
    write_json(os.path.join(outdir, f"clustering_{sel}.json"), clus)
    # interaction test on per-domain gains over ref (arms common to all cohorts)
    inter = {}
    common = None
    for c in names:
        if ref not in cohort_arms[c]:
            continue
        s = set(cohort_arms[c]) - {ref}
        common = s if common is None else common & s
    if common and len(common) >= 2:
        common = sorted(common)
        R, lab = [], []
        for c in names:
            if ref not in cohort_arms[c]:
                continue
            g = np.stack([S.domain_means(cohort_arms[c][a]) - S.domain_means(cohort_arms[c][ref]) for a in common], 1)
            R.append(g)
            lab += [c] * g.shape[0]
        R = np.concatenate(R)
        inter["arms"] = common
        # cohort: domains are the replicates (studentised, wild bootstrap; robust to unequal noise)
        inter["cohort"] = S.interaction_permutation_test(R, lab, B=Bperm, method="wild")
        # task / shift are constant within cohort -> cohort is the unit (row-level shuffles are invalid)
        labs = list(dict.fromkeys(lab))
        if len(labs) == 4 and all(c in S.COHORT_META for c in labs):
            inter["task_shift_2x2"] = S.interaction_2x2_cohort(R, lab)
        else:
            inter["task_shift_2x2"] = dict(note=f"needs the 4 cohorts of the 2x2 design; have {labs}")
    write_json(os.path.join(outdir, f"interaction_{sel}.json"), inter)
    with open(os.path.join(outdir, f"kendall_{sel}.tex"), "w") as f:
        f.write(tex_kendall(rows))
    return dict(kendall=rows, clustering=clus, interaction=inter)


# ----------------------------------------------------------------------------------------
# LaTeX (booktabs) snippets
# ----------------------------------------------------------------------------------------


def _e(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def _f(v, d=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "--"
    return f"{v:.{d}f}"


def _ci(ci, d=3):
    if ci is None:
        return "--"
    return f"[{_f(ci[0], d)}, {_f(ci[1], d)}]"


def _p(p):
    if p is None or not math.isfinite(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def _tab(cols, head, body):
    return ("\\begin{tabular}{" + cols + "}\n\\toprule\n" + head + " \\\\\n\\midrule\n"
            + "\n".join(r + " \\\\" for r in body) + "\n\\bottomrule\n\\end{tabular}\n")


def write_tables(outdir, sel, cohort, summ_rows, var_rows, con_rows, pow_rows, rank_rows, higher_better=True):
    T = os.path.join(outdir, "tables")
    rk = {r["arm"]: r for r in rank_rows}
    sgn = -1 if higher_better else 1
    body = [f"{_e(r['arm'])} & {_f(r['mean'])} & {_ci(r['mean_ci_t'])} & {_f(r['worst'])} & {_ci(r['worst_ci_seed'])} & "
            f"{_f(r['sd_across'])} & {_f(r['mean_within_sd'])} & {_f(rk.get(r['arm'], {}).get('rank_worst'), 1)}"
            for r in sorted(summ_rows, key=lambda r: sgn * r["worst"])]
    with open(os.path.join(T, f"summary_{sel}.tex"), "w") as f:
        f.write(f"% {cohort}, selection={sel}; mean CI: t-interval over domain means (df K-1); worst "
                f"(= {'min' if higher_better else 'max'} over domains) CI: seed-within-domain bootstrap "
                "(observed domains only); domain-bootstrap CIs are supplementary (summary csv)\n")
        f.write(_tab("lrrrrrrr", "Arm & Mean & 95\\% CI & Worst & 95\\% CI & SD$_\\text{dom}$ & SD$_\\text{seed}$ & Rank$_\\text{worst}$", body))
    def _bayes_cell(r):
        # at the zero boundary an equal-tailed interval cannot contain 0 -> one-sided [0, q95] bound
        if r.get("reml_boundary") or r.get("anova_truncated"):
            u = r.get("halfc_icc_u95")
            return f"[0, {_f(u, 2)}]$^\\dagger$" if u is not None else "--"
        return _ci(r.get("halfc_icc_ci"), 2)

    body = [f"{_e(r['arm'])} & {_f(r['sigma_domain_reml'])} & {_ci(r['sigma_domain_ci_mls'])} & {_f(r['sigma_run_reml'])} & "
            f"{_ci(r['sigma_run_ci'])} & {_f(r['icc_reml'], 2)} & {_ci([r['icc_ci_lo'], r['icc_ci_hi']], 2)} & "
            f"{_f(r.get('halfc_icc_med'), 2)} & {_bayes_cell(r)}" for r in var_rows]
    with open(os.path.join(T, f"variance_{sel}.tex"), "w") as f:
        f.write(f"% {cohort}, selection={sel}; REML; sigma_domain CI: MLS (Burdick-Graybill); sigma_run CI: chi2; "
                "ICC CI: exact F; Bayes: half-Cauchy posterior median [95% equal-tailed CrI]; dagger: REML/ANOVA "
                "estimate at the zero boundary, one-sided 95% upper credible bound [0, q95] shown instead\n")
        f.write(_tab("lrrrrrrrr", "Arm & $\\hat\\sigma_\\text{dom}$ & 95\\% CI & $\\hat\\sigma_\\text{run}$ & 95\\% CI & ICC(1) & "
                     "exact 95\\% CI & ICC$_\\text{Bayes}$ & 95\\% CrI", body))
    body = []
    for r in sorted(con_rows, key=lambda r: -r["mean_diff"]):
        body.append(f"{_e(r['arm'])} & {r['mean_diff']:+.3f} & {_ci(r['ci'])} & {_p(r['p'])} & {_p(r['p_holm'])} & "
                    f"{r['n_improved']}/{r['K']} & {_ci(r.get('rho0.5_ci'))} & {_p(r.get('rho0.5_p'))}")
    with open(os.path.join(T, f"contrasts_{sel}.tex"), "w") as f:
        f.write(f"% {cohort}, selection={sel}; paired over domains vs ref; last two cols: fold-dependence rho=0.5\n")
        f.write(_tab("lrrrrrrr", "Arm & $\\bar\\Delta$ & 95\\% CI & $p$ & $p_\\text{Holm}$ & improved & CI$_{\\rho=.5}$ & $p_{\\rho=.5}$", body))
    # power: K_min at s=3 for each delta, rho in {0, .5}
    by = defaultdict(dict)
    deltas = sorted({r["delta"] for r in pow_rows})
    for r in pow_rows:
        if r["seeds"] == 3 and r["rho"] in (0.0, 0.5) and r["keff_mode"] in ("none", "ratio"):
            by[r["arm"]][(r["rho"], r["delta"])] = r["K_min"]
            if r["rho"] == 0.0:
                by[r["arm"]]["sd"] = r["sd_delta"]

    def _cell(d, key):
        if d.get("sd") is None or not math.isfinite(d["sd"]):
            return "--"  # sigma_run not estimable (single seed): no seed extrapolation
        return str(d[key]) if d.get(key) is not None else "$>$1000"

    body = []
    for a, d in by.items():
        cells = [_cell(d, (rho, dl)) for rho in (0.0, 0.5) for dl in deltas]
        body.append(f"{_e(a)} & {_f(d['sd'])} & " + " & ".join(cells))
    with open(os.path.join(T, f"power_{sel}.tex"), "w") as f:
        f.write(f"% {cohort}, selection={sel}; K for 80% power (two-sided paired t, alpha .05), s=3 seeds; "
                f"left block independent folds, right block rho=0.5 x training overlap (observed-design VIF held fixed, "
                f"sigma_Delta divided by sqrt(1 - rbar_obs) as in the contrasts' fold-dependence CI); sigma_Delta column = "
                f"independent-fold value; seed-paired run noise where arm and ref share seeds; deltas={deltas}\n")
        hd = " & ".join([f"$\\Delta$={dl:g}" for dl in deltas] * 2)
        f.write(_tab("lr" + "r" * 2 * len(deltas), "Arm & $\\sigma_\\Delta$ & " + hd, body))


def tex_kendall(rows):
    body = [f"{_e(r['cohort_a'])}--{_e(r['cohort_b'])} & {r['key']} & {r['n_arms']} & {_f(r['tau_b'], 2)} & {_ci(r['ci'], 2)} & {_p(r['p'])}"
            for r in rows]
    return _tab("llrrrr", "Cohorts & Score & $n_\\text{arms}$ & $\\tau_b$ & 95\\% CI (seed bootstrap) & $p$", body)


# ----------------------------------------------------------------------------------------
# DomainBed external example
# ----------------------------------------------------------------------------------------

DB_TEX = os.path.join(REPO, "results", "v2", "external", "src_2007.01434", "extracted", "paper.tex")
SEL_MAP = {"training domain validation set": "training_domain", "leave-one-domain-out cross-validation": "loo",
           "test-domain validation set": "oracle"}
CELL = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*\$\\pm\$\s*(\d+(?:\.\d+)?)\s*$")


def parse_domainbed(tex_path):
    """Parse every per-environment table in the appendix 'Domain generalization accuracies per
    algorithm, dataset, and domain'.  Returns one row per (dataset, selection, algorithm, env)
    with the verbatim cell text, the source file and 1-based line number."""
    lines = open(tex_path, encoding="utf-8", errors="replace").read().split("\n")
    rows = []
    in_app = False
    dataset = selection = None
    cols = None
    in_body = False
    for ln, raw in enumerate(lines, 1):
        s = raw.strip()
        if s.startswith("\\section"):
            in_app = "Domain generalization accuracies per algorithm" in s
            continue
        if not in_app:
            continue
        m = re.match(r"\\subsection\{(.+?)\}", s)
        if m:
            dataset, selection, cols, in_body = m.group(1), None, None, False
            continue
        m = re.search(r"Model selection method:\s*(.+?)(?:\}|\s*\\textit)", s)
        if m:
            selection = m.group(1).strip()
            cols, in_body = None, False
            continue
        if s.startswith("\\textbf{Algorithm}"):
            cells = [c.strip() for c in s.rstrip("\\").split("&")]
            cols = [re.sub(r"\\textbf\{(.*)\}", r"\1", c).strip() for c in cells[1:]]
            continue
        if s.startswith("\\midrule") and cols is not None:
            in_body = True
            continue
        if s.startswith("\\bottomrule"):
            in_body = False
            cols = None
            continue
        if in_body and "&" in s:
            cells = [c.strip() for c in re.sub(r"\\\\\s*$", "", s).split("&")]
            alg = cells[0]
            if len(cells) - 1 != len(cols):
                raise ValueError(f"{tex_path}:{ln}: {len(cells) - 1} cells vs {len(cols)} columns")
            for env, cell in zip(cols, cells[1:]):
                mm = CELL.match(cell)
                if not mm:
                    raise ValueError(f"{tex_path}:{ln}: cannot parse cell {cell!r}")
                rows.append(dict(dataset=dataset, selection=SEL_MAP.get(selection, selection), selection_text=selection,
                                 algorithm=alg, env=env, mean=float(mm.group(1)), se=float(mm.group(2)), cell=cell,
                                 source_file=os.path.relpath(tex_path, REPO), line=ln))
    return rows


def analyse_domainbed(rows, n_trials=3, priors=(("halfcauchy", 10.0), ("halfnormal", 25.0))):
    """Per (dataset, selection, algorithm): one-way random effects on env x trial accuracies (pp),
    reconstructed from mean +- SE.  DomainBed's collect_results.format_mean computes
    SE = np.std(x / sqrt(n)) (population SD, ddof=0, over sqrt(n)), hence
    sample SD (ddof=1) = SE * n / sqrt(n - 1)   [primary, 'code' convention]
    and, as asked in the spec, sd = SE * sqrt(n) (equals the ddof=0 SD) [alt convention]."""
    grp = defaultdict(list)
    for r in rows:
        grp[(r["dataset"], r["selection"], r["algorithm"])].append(r)
    out = []
    n = n_trials
    for (ds, sel, alg), rr in grp.items():
        envs = [r for r in rr if r["env"].lower() not in ("avg", "average")]
        means = [r["mean"] for r in envs]
        se = np.array([r["se"] for r in envs])
        row = dict(dataset=ds, selection=sel, algorithm=alg, K=len(envs), n_trials=n,
                   envs=[r["env"] for r in envs], mean_over_envs=float(np.mean(means)),
                   worst_env=float(np.min(means)), sd_env_means=float(np.std(means, ddof=1)),
                   n_se_zero=int((se == 0).sum()), lines=f"{min(r['line'] for r in rr)}-{max(r['line'] for r in rr)}")
        for conv, sd in (("code", se * n / math.sqrt(n - 1)), ("alt", se * math.sqrt(n))):
            g = S.groups_from_summary(means, sd, n)
            a = S.oneway_anova(g)
            c = S.icc1_exact_ci(g)
            row[f"{conv}_sigma_run"] = a["sigma_run"]
            row[f"{conv}_sigma_env"] = a["sigma_domain"]
            row[f"{conv}_s2_env_raw"] = a["s2_domain_raw"]
            row[f"{conv}_icc"] = a["icc"]
            row[f"{conv}_icc_lo"] = c["lo"]
            row[f"{conv}_icc_hi"] = c["hi"]
            row[f"{conv}_F"] = a["F"]
            row[f"{conv}_p"] = a["p"]
            if conv == "code":
                r_ = S.oneway_reml(g)
                row["code_sigma_env_reml"], row["code_sigma_run_reml"] = r_["sigma_domain"], r_["sigma_run"]
                row["code_sigma_env_ci_mls"] = S.sigma_domain_mls_ci(g)
                row["code_sigma_run_ci"] = S.sigma_run_ci(g)
                for kind, sc in priors:
                    b = S.bayes_oneway(g, kind, sc)
                    tag = f"{kind}{sc:g}"
                    row[f"{tag}_icc_med"] = b["icc"]["median"]
                    row[f"{tag}_icc_ci"] = [b["icc"]["lo"], b["icc"]["hi"]]
                    row[f"{tag}_sigma_env_med"] = b["sigma_domain"]["median"]
                    row[f"{tag}_sigma_run_med"] = b["sigma_run"]["median"]
        out.append(row)
    return out


def tex_domainbed(an, sel="training_domain"):
    """booktabs snippet: per dataset, ERM sigma_env / sigma_run / ICC [exact CI] and the ICC range over
    algorithms (code SD convention), for the given selection rule."""
    order = ["Colored MNIST", "Rotated MNIST", "VLCS", "PACS", "Office-Home", "TerraIncognita", "DomainNet"]
    body = []
    for ds in order + sorted({r["dataset"] for r in an} - set(order)):
        rr = [r for r in an if r["dataset"] == ds and r["selection"] == sel]
        if not rr:
            continue
        e = [r for r in rr if r["algorithm"] == "ERM"]
        e = e[0] if e else rr[0]
        ic = [r["code_icc"] for r in rr]
        lo = min(rr, key=lambda r: r["code_icc"])
        body.append(f"{_e(ds)} & {e['K']} & {_f(e['code_sigma_env'], 1)} & {_f(e['code_sigma_run'], 1)} & {_f(e['code_icc'], 2)} & "
                    f"{_ci([e['code_icc_lo'], e['code_icc_hi']], 2)} & {_f(min(ic), 2)}--{_f(max(ic), 2)} ({_e(lo['algorithm'])})")
    return ("% DomainBed (Gulrajani & Lopez-Paz, arXiv 2007.01434v1, App. 'Domain generalization accuracies per algorithm, "
            "dataset, and domain'), selection=" + sel + "; accuracy in pp; 3 trials; SD from SE via DomainBed's "
            "np.std(x/sqrt(n)) (ddof=0): sd = SE*n/sqrt(n-1); one-way model env x trial\n"
            + _tab("lrrrrrl", "Dataset & $K$ & $\\hat\\sigma_\\text{env}$ (ERM) & $\\hat\\sigma_\\text{run}$ (ERM) & ICC (ERM) & "
                   "exact 95\\% CI & ICC range over algorithms (min alg.)", body))


# ----------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------


def read_json_or_none(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def record_snapshot(root):
    """{id, created, source, snapshot} of a frozen record tree (make_figures_v2 --freeze), else None."""
    m = read_json_or_none(os.path.join(root, "SNAPSHOT.json"))
    if not m or not m.get("id"):
        return None
    return dict(id=m["id"], created=m.get("created"), source=m.get("source"), snapshot=os.path.abspath(root))


def run_cohorts(cohorts_obj, args, tier_out):
    cohort_arms = {}
    for C in cohorts_obj:
        if not C.domain or not C.vals:
            print(f"[{C.name}] no usable records for selection {args.select!r}: {'; '.join(C.notes[:3])}", file=sys.stderr)
            continue
        od = os.path.join(tier_out, C.name)
        sel = getattr(args, "sel_label", None) or args.select
        res, arms = analyse_cohort(C, od, sel, ref=args.ref, B=args.B, prior_scale=args.prior_scale,
                                   min_seeds=args.min_seeds, higher_better=not args.lower_better,
                                   keep=args.arms, drop=args.exclude, snapshot=getattr(args, "snapshot", None))
        cohort_arms[C.name] = arms
        print(f"[{C.name}] K={res['K']} arms={len(arms)} -> {od}")
        for n in C.notes[:20]:
            print("   note:", n)
    if len(cohort_arms) >= 2:
        cr = analyse_cross(cohort_arms, os.path.join(tier_out, "cross"), getattr(args, "sel_label", None) or args.select,
                           ref=args.ref, B=args.B_tau, Bperm=args.B_perm, higher_better=not args.lower_better)
        print(f"[cross] {len(cr['kendall'])} tau rows -> {os.path.join(tier_out, 'cross')}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--select", default="id", choices=["id", "ood", "last", "oracle"],
                       help="model-selection rule (test_at key); default id = training-domain validation")
        p.add_argument("--metric", default="auroc")
        p.add_argument("--ref", default="erm", help="reference arm for paired contrasts")
        p.add_argument("--B", type=int, default=10000, help="bootstrap draws for summaries / ranks")
        p.add_argument("--B-tau", type=int, default=2000, help="bootstrap draws for Kendall tau CIs")
        p.add_argument("--B-perm", type=int, default=10000, help="permutations for the interaction test")
        p.add_argument("--prior-scale", type=float, default=0.1, help="half-normal / half-Cauchy scale (metric units)")
        p.add_argument("--min-seeds", type=int, default=1)
        p.add_argument("--lower-better", action="store_true",
                       help="metric is lower-is-better (ECE/NLL/Brier): worst = max over domains, improved = d < 0, ranks")
        p.add_argument("--arms", nargs="+", default=None, help="analyse only these arm labels (must include --ref)")
        p.add_argument("--exclude", nargs="+", default=None, help="drop these arm labels")

    p = sub.add_parser("runs", help="v2 run records")
    p.add_argument("--v2-root", default=V2)
    p.add_argument("--tier", default="C", choices=["C", "S"])
    p.add_argument("--cohorts", nargs="+", default=COHORTS)
    p.add_argument("--phase", default="final")
    p.add_argument("--ladder", default="exclude", choices=["exclude", "include", "only"],
                   help="tier S: drop (default), keep, or analyse only the supplementary ladder probes")
    p.add_argument("--out", default=os.path.join(REPO, "results", "v2", "analysis"))
    common(p)
    p = sub.add_parser("v1", help="v1 record layout (demo / regression)")
    p.add_argument("--runs", required=True, nargs="+", help="one or more v1 run dirs (one per cohort)")
    p.add_argument("--cohort", required=True, nargs="+")
    p.add_argument("--val-offset", nargs="+", default=["-1"],
                   help="per cohort: '-1' (val=(t-1) mod K) or 'none' (no val domain); one value = all cohorts")
    p.add_argument("--tier", default="v1")
    p.add_argument("--out", required=True)
    common(p)
    p = sub.add_parser("domainbed", help="DomainBed external ICC example")
    p.add_argument("--tex", default=DB_TEX)
    p.add_argument("--n-trials", type=int, default=3)
    p.add_argument("--out", default=os.path.join(REPO, "results", "v2", "external"))
    args = ap.parse_args(argv)

    if args.cmd == "runs":
        Cs = [load_v2(args.v2_root, args.tier, c, args.select, args.metric, args.phase, ladder=args.ladder)
              for c in args.cohorts]
        tier_dir = args.tier if not (args.tier == "S" and args.ladder != "exclude") else f"S_ladder_{args.ladder}"
        args.snapshot = record_snapshot(args.v2_root)
        run_cohorts(Cs, args, os.path.join(args.out, tier_dir))
        if args.snapshot:   # provenance stamp of the tier directory (make_figures_v2 checks it against --runs)
            stamp = os.path.join(args.out, tier_dir, "SNAPSHOT.json")
            man = read_json_or_none(os.path.join(args.v2_root, "SNAPSHOT.json")) or {}
            write_json(stamp, dict(args.snapshot, counts=(man.get("counts") or {}).get(args.tier, {})))
    elif args.cmd == "v1":
        vos = [None if v.lower() == "none" else int(v) for v in args.val_offset]
        if len(vos) == 1:
            vos = vos * len(args.cohort)
        if not (len(args.runs) == len(args.cohort) == len(vos)):
            ap.error("--runs, --cohort (and --val-offset if several) must have the same length")
        Cs = [load_v1(r, c, args.metric, vo) for r, c, vo in zip(args.runs, args.cohort, vos)]
        if args.select != "id":
            warnings.warn("v1 records have a single selection; --select is ignored")
        args.sel_label = "v1rule"  # v1 records: OOD-val selection (C17/canine), ID-val (MIDOG) -- never label 'id'
        run_cohorts(Cs, args, os.path.join(args.out, args.tier))
    elif args.cmd == "domainbed":
        rows = parse_domainbed(args.tex)
        write_csv(os.path.join(args.out, "domainbed_tables.csv"), rows,
                  ["dataset", "selection", "selection_text", "algorithm", "env", "mean", "se", "cell", "source_file", "line"])
        an = analyse_domainbed(rows, args.n_trials)
        write_csv(os.path.join(args.out, "domainbed_icc.csv"), an)
        with open(os.path.join(args.out, "domainbed_icc_training_domain.tex"), "w") as f:
            f.write(tex_domainbed(an))
        print(f"parsed {len(rows)} cells from {args.tex}; {len(an)} (dataset, selection, algorithm) groups")


if __name__ == "__main__":
    main()
