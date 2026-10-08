"""Empirical-Bayes site acceptance test (SAT) for deploying a trained model at a new site.

Question a practitioner asks before using a model at a new hospital or scanner: "is this model's AUROC at the new
site at least T?". Two sources of evidence are combined.

1. Development prior. The model's method was evaluated leave-one-site-out on K' development sites with n runs
   (seeds) each. On the logit-AUROC scale, the one-way random-effects model of the paper,
       z_sr = mu + b_s + e_sr,   b_s ~ N(0, sd_D^2),   e_sr ~ N(0, sd_R^2),
   with a flat prior on mu and independent half-Cauchy priors on sd_D and sd_R, gives the posterior predictive
   distribution of a single run's logit AUROC at a new site,
       theta_new | sd ~ N(zbar, sd_D^2 + sd_R^2 + (sd_D^2 + sd_R^2 / n) / K'),
   integrated over the grid posterior of (sd_D, sd_R) (restricted likelihood, balanced data). The result is a
   scale mixture of normals centred at the grand mean zbar.
2. Local labels. m patches of the new site are labelled at random. Their AUROC a_m (continuity-corrected) gives
   z_m = logit(a_m) with the Hanley-McNeil variance mapped by the delta method, v_m = Var(a_m) / (a_m (1 - a_m))^2.

The posterior of the site's logit AUROC is again a normal mixture (each component updated conjugately). The site
is accepted if P(AUROC >= T | data) >= 1 - alpha (default alpha = 0.05). `labels_needed` plans m by pre-posterior
simulation from the development prior.

Validation (`evaluate`): every held-out site of every cohort plays the new site in turn. Its prior uses only the
other folds' results of the same method; local samples are drawn from the run's saved test predictions; the truth
is the run's AUROC on the full test split. Baselines: local labels alone (one-sided Wald bound on the logit scale)
and the development prior alone (m = 0).

    python src/v2/site_acceptance.py evaluate --tier S --out results/v2/site_acceptance
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
from scipy import stats
from scipy.special import expit, logit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

EPS = 1e-12


# --------------------------------------------------------------------------------------------- development prior
def _log_halfcauchy(s, scale):
    return -np.log1p((s / scale) ** 2)


def fit_prior(Z, scale=0.5, n_grid=(160, 160), n_bins=120):
    """Posterior predictive of one run's logit AUROC at a new site.

    Z: array (K', n) of logit AUROCs (K' development sites, n runs each; balanced).
    scale: half-Cauchy scale for sd_D and sd_R on the logit scale (0.5 logit ~ 0.06 AUROC at 0.85).
    Returns dict(center, V, w): mixture of N(center, V_k) with weights w_k (sum 1).
    """
    Z = np.asarray(Z, float)
    if Z.ndim != 2 or Z.shape[0] < 2 or Z.shape[1] < 2:
        raise ValueError("need >= 2 sites and >= 2 runs per site")
    K, n = Z.shape
    zs = Z.mean(1)
    zbar = zs.mean()
    ssw = float(((Z - zs[:, None]) ** 2).sum())
    ssb = float(n * ((zs - zbar) ** 2).sum())
    sd = np.concatenate([[0.0], np.geomspace(1e-3, 8.0, n_grid[0] - 1)])
    sr = np.geomspace(1e-3, 8.0, n_grid[1])
    SD, SR = np.meshgrid(sd, sr, indexing="ij")
    s2r, lam = SR ** 2, SR ** 2 + n * SD ** 2
    # restricted log-likelihood of the balanced one-way model (mu integrated out under a flat prior)
    ll = -0.5 * (K * (n - 1) * np.log(s2r) + ssw / s2r + (K - 1) * np.log(lam) + ssb / lam)
    lp = ll + _log_halfcauchy(SD, scale) + _log_halfcauchy(SR, scale)
    wd = np.gradient(sd)
    wr = np.gradient(sr)
    lp = lp + np.log(np.outer(wd, wr))
    lp -= lp.max()
    w = np.exp(lp)
    w /= w.sum()
    V = SD ** 2 + SR ** 2 + (SD ** 2 + SR ** 2 / n) / K
    # collapse the 2-D grid into a 1-D scale mixture (the predictive depends on (sd_D, sd_R) only through V)
    lv = np.log(V.ravel())
    edges = np.linspace(lv.min(), lv.max() + 1e-9, n_bins + 1)
    idx = np.clip(np.digitize(lv, edges) - 1, 0, n_bins - 1)
    wb = np.bincount(idx, weights=w.ravel(), minlength=n_bins)
    vb = np.bincount(idx, weights=(w * V).ravel(), minlength=n_bins)
    keep = wb > 1e-14
    Vk, wk = vb[keep] / wb[keep], wb[keep] / wb[keep].sum()
    return dict(center=float(zbar), centers=np.full(Vk.size, float(zbar)), V=Vk, w=wk, K=K, n=n)


def robustify(prior, eps=0.1, center=0.0, sd=2.0):
    """epsilon-contamination: (1 - eps) x development prior + eps x vague N(center, sd^2) on the logit scale
    (default: centred at AUROC 0.5, 95 % of its mass between AUROC 0.02 and 0.98)."""
    return dict(prior, centers=np.append(prior["centers"], center), V=np.append(prior["V"], sd ** 2),
                w=np.append((1 - eps) * prior["w"], eps), eps=eps)


# --------------------------------------------------------------------------------------------- local estimate
def hanley_mcneil_var(a, n1, n0):
    a = np.asarray(a, float)
    q1, q2 = a / (2 - a), 2 * a ** 2 / (1 + a)
    return (a * (1 - a) + (n1 - 1) * (q1 - a ** 2) + (n0 - 1) * (q2 - a ** 2)) / (n1 * n0)


def local_estimate(y, p):
    """Continuity-corrected AUROC of labelled samples (rows of y, p: shape (draws, m)) on the logit scale.

    Returns (z, v, n1, n0); rows without both classes get z = nan."""
    y = np.atleast_2d(y).astype(bool)
    p = np.atleast_2d(p)
    r = stats.rankdata(p, axis=1)
    n1 = y.sum(1)
    n0 = y.shape[1] - n1
    u = (r * y).sum(1) - n1 * (n1 + 1) / 2.0
    ok = (n1 > 0) & (n0 > 0)
    nn = np.where(ok, n1 * n0, 1)
    a = (u + 0.5) / (nn + 1.0)                     # stays inside (0, 1) even under perfect separation
    v = hanley_mcneil_var(a, np.maximum(n1, 1), np.maximum(n0, 1)) / (a * (1 - a)) ** 2
    z = np.where(ok, logit(a), np.nan)
    return z, np.where(ok, v, np.nan), n1, n0


# --------------------------------------------------------------------------------------------- posterior
def posterior(prior, z=None, v=None):
    """Mixture posterior for the site's logit AUROC. z, v: arrays (draws,) or None (prior only).

    Returns dict(w, mean, sd) with arrays of shape (draws, comps)."""
    c, V, w = prior["centers"][None, :], prior["V"][None, :], prior["w"][None, :]
    if z is None:
        return dict(w=w, mean=c + 0 * V, sd=np.sqrt(V))
    z = np.asarray(z, float)[:, None]
    v = np.asarray(v, float)[:, None]
    lw = np.log(w + EPS) - 0.5 * np.log(V + v) - 0.5 * (z - c) ** 2 / (V + v)
    lw -= lw.max(1, keepdims=True)
    ww = np.exp(lw)
    ww /= ww.sum(1, keepdims=True)
    mean = (c * v + z * V) / (V + v)
    sd = np.sqrt(V * v / (V + v))
    return dict(w=ww, mean=mean, sd=sd)


def prob_above(post, t_logit):
    return (post["w"] * stats.norm.sf((t_logit - post["mean"]) / post["sd"])).sum(1)


def quantile(post, q, iters=40):
    """Mixture quantile by vectorised bisection (logit scale)."""
    lo = (post["mean"] - 10 * post["sd"]).min(1)
    hi = (post["mean"] + 10 * post["sd"]).max(1)
    for _ in range(iters):
        mid = (lo + hi) / 2
        cdf = (post["w"] * stats.norm.cdf((mid[:, None] - post["mean"]) / post["sd"])).sum(1)
        lo = np.where(cdf < q, mid, lo)
        hi = np.where(cdf < q, hi, mid)
    return (lo + hi) / 2


def accept(post, target, alpha=0.05):
    return prob_above(post, logit(target)) >= 1 - alpha


def labels_needed(prior, target, margin=0.02, power=0.8, alpha=0.05, prevalence=0.5,
                  ms=(25, 50, 100, 200, 400, 800), sims=4000, seed=0):
    """Smallest m such that, for sites drawn from the development prior whose AUROC is >= target + margin,
    the test accepts with probability >= power (pre-posterior simulation; binormal local scores)."""
    rng = np.random.default_rng(seed)
    k = rng.choice(len(prior["w"]), size=sims, p=prior["w"])
    th = expit(prior["centers"][k] + np.sqrt(prior["V"][k]) * rng.standard_normal(sims))
    good = th >= target + margin
    if not good.any():
        return None, {}
    th = th[good]
    out = {}
    for m in ms:
        n1 = max(1, int(round(m * prevalence)))
        n0 = max(1, m - n1)
        d = math.sqrt(2) * stats.norm.ppf(np.clip(th, 1e-6, 1 - 1e-6))     # binormal separation for AUROC th
        s1 = rng.standard_normal((th.size, n1)) + d[:, None]
        s0 = rng.standard_normal((th.size, n0))
        yy = np.concatenate([np.ones((th.size, n1)), np.zeros((th.size, n0))], 1)
        z, v, _, _ = local_estimate(yy, np.concatenate([s1, s0], 1))
        acc = accept(posterior(prior, z, v), target, alpha)
        out[m] = float(acc.mean())
        if acc.mean() >= power:
            return m, out
    return None, out


# --------------------------------------------------------------------------------------------- validation
def _records(snap, tier, cohort, arms=None):
    """{(arm, fold, seed): (status, preds path)} for the final runs of one tier and cohort."""
    out = {}
    for f in glob.glob(os.path.join(snap, "runs", tier, cohort, "fold*", "*__s*.json")):
        r = json.load(open(f))
        if r.get("phase") != "final":
            continue
        if arms and r["arm"] not in arms:
            continue
        out[(r["arm"], int(r["fold"]), int(r["seed"]))] = (r.get("status"), f[:-5] + "_preds.npz")
    return out


def _run_rng(seed, *key):
    """RNG seeded from a stable key, so the draws do not depend on file-listing order."""
    import zlib
    return np.random.default_rng([seed, zlib.crc32("/".join(map(str, key)).encode())])


def evaluate(snap, tier, cohorts, out_dir, ms=(25, 50, 100, 200, 400), draws=200, targets=(0.80, 0.85, 0.90),
             alpha=0.05, scale=0.5, arms=None, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for cohort in cohorts:
        rec = _records(snap, tier, cohort, arms)
        arm_list = sorted({a for a, _, _ in rec})
        folds = sorted({f for _, f, _ in rec})
        for arm in arm_list:
            if arm.startswith("probe_") and arm not in ("probe_resnet50_imagenet", "probe_resnet50_lunit_bt",
                                                         "probe_vit_s16_lunit_dino"):
                continue        # ladder rungs: supplementary only
            truth = {}
            for (a, f, s), (auc, pp) in sorted(rec.items()):
                if a != arm:
                    continue
                d = np.load(pp)
                y, p = d["y_ood_test"].astype(bool), d["p_ood_test_id"].astype(float)
                p = np.nan_to_num(p, nan=0.5)
                full = float(stats.mannwhitneyu(p[y], p[~y]).statistic / (y.sum() * (~y).sum()))
                truth[(f, s)] = (full, y, p)
            seeds = sorted({s for _, s in truth})
            for t in folds:
                dev = [f for f in folds if f != t]
                try:
                    Z = np.array([[logit(np.clip(truth[(f, s)][0], 1e-4, 1 - 1e-4)) for s in seeds] for f in dev])
                except KeyError:
                    continue
                prior = fit_prior(Z, scale=scale)
                rprior = robustify(prior)
                p0 = posterior(prior)
                q0 = expit(quantile(p0, np.array([0.05]))[0]), expit(quantile(p0, np.array([0.95]))[0])
                for s in seeds:
                    full, y, p = truth[(t, s)]
                    rng = _run_rng(seed, tier, cohort, arm, t, s)
                    base = dict(tier=tier, cohort=cohort, arm=arm, fold=t, seed=s, truth=full,
                                dev_center=float(expit(prior["center"])))
                    row = dict(base, m=0, method="dev", est=float(expit(prior["center"])),
                               lo90=float(q0[0]), hi90=float(q0[1]))
                    for T in targets:
                        row[f"acc_{T}"] = float(prob_above(p0, logit(T))[0] >= 1 - alpha)
                    rows.append(row)
                    n = y.size
                    for m in ms:
                        idx = np.stack([rng.choice(n, size=m, replace=False) for _ in range(draws)])
                        z, v, n1, n0 = local_estimate(y[idx], p[idx])
                        ok = np.isfinite(z)
                        z, v = z[ok], v[ok]
                        post = posterior(prior, z, v)
                        med = expit(quantile(post, 0.5))
                        lo, hi = expit(quantile(post, 0.05)), expit(quantile(post, 0.95))
                        pa = {T: prob_above(post, logit(T)) >= 1 - alpha for T in targets}
                        rpost = posterior(rprior, z, v)
                        rmed = expit(quantile(rpost, 0.5))
                        rlo, rhi = expit(quantile(rpost, 0.05)), expit(quantile(rpost, 0.95))
                        rpa = {T: prob_above(rpost, logit(T)) >= 1 - alpha for T in targets}
                        loc_lo = expit(z - 1.645 * np.sqrt(v))
                        loc_hi = expit(z + 1.645 * np.sqrt(v))
                        for meth, est, l, h, acc in (
                                ("eb", med, lo, hi, pa),
                                ("eb_robust", rmed, rlo, rhi, rpa),
                                ("local", expit(z), loc_lo, loc_hi, {T: loc_lo >= T for T in targets})):
                            r = dict(base, m=m, method=meth, n_draws=int(ok.sum()),
                                     est=float(np.mean(est)), abs_err=float(np.mean(np.abs(est - full))),
                                     cover90=float(np.mean((l <= full) & (full <= h))),
                                     width90=float(np.mean(h - l)))
                            for T in targets:
                                r[f"acc_{T}"] = float(np.mean(acc[T]))
                            rows.append(r)
        print(f"[{tier}/{cohort}] {len(rows)} rows", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"sat_rows_{tier}.csv"), index=False)
    return df


def summarize(df, targets=(0.80, 0.85, 0.90), margin=0.0):
    """Decision error rates per method and m: false acceptance (truth < T) and acceptance of good sites."""
    import pandas as pd
    out = []
    for (tier, method, m), g in df.groupby(["tier", "method", "m"]):
        r = dict(tier=tier, method=method, m=m, n=len(g),
                 cover90=g["cover90"].mean() if "cover90" in g and g["cover90"].notna().any() else np.nan,
                 abs_err=g["abs_err"].mean() if "abs_err" in g and g["abs_err"].notna().any() else np.nan)
        for T in targets:
            bad = g["truth"] < T
            good = g["truth"] >= T + margin
            r[f"far_{T}"] = g.loc[bad, f"acc_{T}"].mean() if bad.any() else np.nan
            r[f"power_{T}"] = g.loc[good, f"acc_{T}"].mean() if good.any() else np.nan
            r[f"n_bad_{T}"] = int(bad.sum())
        out.append(r)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------------- tables / numbers
METHOD_TEX = {"dev": "Development prior only", "eb": "Site acceptance test", "eb_robust": "Robust variant",
              "local": "Local labels only"}


def rates(g, T, margin=0.0):
    """False acceptance among runs with truth < T - margin, power among runs with truth >= T."""
    bad, good = g.truth < T - margin, g.truth >= T
    return (g.loc[bad, f"acc_{T}"].mean() if bad.any() else float("nan"),
            g.loc[good, f"acc_{T}"].mean() if good.any() else float("nan"), int(bad.sum()), int(good.sum()))


def table_tex(df, methods=("dev", "eb", "local"), targets=(0.80, 0.85, 0.90), ms=(25, 50, 100, 200, 400)):
    lines = []
    for meth in methods:
        mm = [0] if meth == "dev" else list(ms)
        for k, m in enumerate(mm):
            g = df[(df.method == meth) & (df.m == m)]
            cells = [METHOD_TEX[meth] if k == 0 else "", str(m)]
            for T in targets:
                far, pw, _, _ = rates(g, T)
                cells += [f"{100 * far:.1f}", f"{100 * pw:.0f}"]
            if meth == "dev":
                cells += ["--", "--"]
            else:
                cells += [f"{g.abs_err.mean():.3f}", f"{100 * g.cover90.mean():.0f}"]
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\addlinespace[2pt]")
    head1 = " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$T={T:.2f}$}}" for T in targets) + " & & \\\\"
    cm = "".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(targets)))
    head2 = "Rule & $m$ & " + " & ".join(["FA (\\%)", "power (\\%)"] * len(targets)) + \
        " & MAE & cover. (\\%) \\\\"
    return ("% generated by src/v2/site_acceptance.py tables (results/v2/site_acceptance/sat_rows_*.csv)\n"
            "\\begin{tabular*}{\\tblwidth}{@{\\extracolsep{\\fill}}lr" + "rr" * len(targets) + "rr@{}}\n\\toprule\n"
            + head1 + "\n" + cm + "\n" + head2 + "\n\\midrule\n" + "\n".join(lines[:-1]) +
            "\n\\bottomrule\n\\end{tabular*}\n")


def cohort_table_tex(dfs, T=0.85, ms=(50, 100), methods=("eb", "eb_robust", "local")):
    lines = []
    for tier, df in dfs:
        for c in ("c17", "canine", "midog21sn", "midogpp"):
            g0 = df[df.cohort == c]
            cells = [tier if c == "c17" else "", {"c17": "Camelyon17", "canine": "Canine cSCC",
                                                  "midog21sn": "MIDOG 2021", "midogpp": "MIDOG++"}[c]]
            nb = int((g0[(g0.method == "dev")].truth < T).sum())
            cells.append(str(nb))
            for meth in methods:
                for m in ms:
                    far, pw, _, _ = rates(g0[(g0.method == meth) & (g0.m == m)], T)
                    cells.append(f"{100 * far:.1f} / {100 * pw:.0f}")
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\addlinespace[3pt]")
    head = "Tier & Cohort & bad & " + " & ".join(f"{METHOD_TEX[mt].split()[0]} {m}" for mt in methods for m in ms)
    return ("% generated by src/v2/site_acceptance.py tables; cells: false acceptance (%) / power (%), T = "
            f"{T}\n\\begin{{tabular*}}{{\\tblwidth}}{{@{{\\extracolsep{{\\fill}}}}llr" + "r" * len(methods) * len(ms)
            + "@{}}\n\\toprule\n" + head + " \\\\\n\\midrule\n" + "\n".join(lines[:-1]) +
            "\n\\bottomrule\n\\end{tabular*}\n")


def numbers(dfs, targets=(0.80, 0.85, 0.90)):
    out = {}
    for tier, df in dfs:
        d = {}
        for meth in ("dev", "eb", "eb_robust", "local"):
            for m in sorted(df[df.method == meth].m.unique()):
                g = df[(df.method == meth) & (df.m == m)]
                r = {}
                for T in targets:
                    far, pw, nb, ng = rates(g, T)
                    far2, _, nb2, _ = rates(g, T, margin=0.02)
                    r[str(T)] = dict(far=far, power=pw, n_bad=nb, n_good=ng, far_clear=far2, n_clear_bad=nb2)
                if meth != "dev":
                    r["abs_err"], r["cover90"], r["width90"] = (float(g.abs_err.mean()), float(g.cover90.mean()),
                                                                float(g.width90.mean()))
                d[f"{meth}/{int(m)}"] = r
        # worst cohort x target false acceptance per rule and m
        worst = {}
        for meth in ("eb", "eb_robust", "local"):
            for m in (25, 50, 100, 200, 400):
                best = (0, None)
                for c in df.cohort.unique():
                    for T in targets:
                        far, _, nb, _ = rates(df[(df.cohort == c) & (df.method == meth) & (df.m == m)], T)
                        if nb >= 5 and far > best[0]:
                            best = (far, f"{c}@{T}")
                worst[f"{meth}/{m}"] = dict(far=best[0], where=best[1])
        d["worst_cohort_far"] = worst
        out[tier] = d
    return out


def plan_check(snap, tier, cohorts, n_priors=150, T=0.85, margin=0.02, seed=0, rows=None):
    """Planner calibration: predicted vs observed acceptance of good sites (truth >= T + margin)."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    df = rows
    keys = df[df.method == "eb"][["cohort", "arm", "fold"]].drop_duplicates().values.tolist()
    rng.shuffle(keys)
    res = []
    for cohort, arm, fold in keys[:n_priors]:
        g = df[(df.cohort == cohort) & (df.arm == arm) & (df.method == "dev")]
        dev = g[g.fold != fold]
        seeds = sorted(dev.seed.unique())
        try:
            Z = np.array([[logit(np.clip(dev[(dev.fold == f) & (dev.seed == s)].truth.iloc[0], 1e-4, 1 - 1e-4))
                           for s in seeds] for f in sorted(dev.fold.unique())])
        except IndexError:
            continue
        prior = fit_prior(Z)
        _, curve = labels_needed(prior, T, margin=margin, power=2.0, ms=(25, 50, 100, 200, 400), sims=2000,
                                 seed=int(rng.integers(1 << 30)))
        obs = df[(df.cohort == cohort) & (df.arm == arm) & (df.fold == fold) & (df.method == "eb")
                 & (df.truth >= T + margin)]
        for m, pred in curve.items():
            o = obs[obs.m == m][f"acc_{T}"]
            if len(o):
                res.append(dict(cohort=cohort, arm=arm, fold=fold, m=m, predicted=pred, observed=float(o.mean())))
    return pd.DataFrame(res)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--snap", default=os.path.join(os.environ.get("HISTOPATH_DATA", "data"), "v2",
                                                  "snap_final2_20261005"))
    e.add_argument("--tier", default="S", choices=["C", "S"])
    e.add_argument("--cohorts", nargs="+", default=["c17", "canine", "midog21sn", "midogpp"])
    e.add_argument("--draws", type=int, default=200)
    e.add_argument("--scale", type=float, default=0.5)
    e.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(HERE)), "results", "v2",
                                                 "site_acceptance"))
    t = sub.add_parser("tables")
    t.add_argument("--rows", default=os.path.join(os.path.dirname(os.path.dirname(HERE)), "results", "v2",
                                                  "site_acceptance"))
    t.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(HERE)), "paper", "media", "tables"))
    t.add_argument("--plan-check", action="store_true", help="also check the label planner on 150 priors per tier")
    a = ap.parse_args()
    if a.cmd == "tables":
        import pandas as pd
        dfs = [(tier, pd.read_csv(os.path.join(a.rows, f"sat_rows_{tier}.csv"))) for tier in ("S", "C")]
        for tier, df in dfs:
            open(os.path.join(a.out, f"tab_sat_{tier}.tex"), "w").write(table_tex(df))
            open(os.path.join(a.out, f"tab_sat_robust_{tier}.tex"), "w").write(
                table_tex(df, methods=("eb_robust",)))
        open(os.path.join(a.out, "tab_sat_cohorts.tex"), "w").write(cohort_table_tex(dfs))
        nums = numbers(dfs)
        if a.plan_check:
            for tier, df in dfs:
                pc = plan_check(None, tier, None, rows=df)
                pc.to_csv(os.path.join(a.rows, f"plan_check_{tier}.csv"), index=False)
                agg = pc.groupby("m")[["predicted", "observed"]].mean()
                nums[tier]["plan_check"] = {str(int(m)): dict(predicted=float(r.predicted), observed=float(r.observed))
                                            for m, r in agg.iterrows()}
                nums[tier]["plan_check_n_priors"] = int(pc[["cohort", "arm", "fold"]].drop_duplicates().shape[0])
        json.dump(nums, open(os.path.join(a.rows, "sat_numbers.json"), "w"), indent=1)
        print(json.dumps({k: {kk: v for kk, v in d.items() if kk in ("worst_cohort_far", "plan_check")}
                          for k, d in nums.items()}, indent=1)[:3000])
        return
    if a.cmd == "evaluate":
        df = evaluate(a.snap, a.tier, a.cohorts, a.out, draws=a.draws, scale=a.scale)
        s = summarize(df)
        s.to_csv(os.path.join(a.out, f"sat_summary_{a.tier}.csv"), index=False)
        print(s.to_string())


if __name__ == "__main__":
    main()
