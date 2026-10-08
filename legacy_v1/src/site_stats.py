"""Statistical framework for site-robustness claims.

Two tools the paper needs to survive editorial scrutiny about single-seed,
single-site evaluation:

  1. `variance_decomposition(df)` -- decomposes off-site AUROC variance into
     three sources under a two-way random-effects model:
        y_{fold,seed} = mu + b_fold + e_seed(fold),
     estimated by matched pairing across seeds.  Returns sigma^2_fold,
     sigma^2_seed and the intraclass correlation ICC = sigma^2_fold /
     (sigma^2_fold + sigma^2_seed).  This makes concrete the paper's core
     claim that fold effects dominate seed effects, without borrowing a
     mixed-effects package the community may not have installed.

  2. `min_folds_for_power(sigma_fold, sigma_seed, effect, alpha, power)` --
     given a target effect size (e.g. the mean SSL gain of 0.092 AUROC)
     returns the smallest K such that a K-fold, s-seed design detects the
     effect with the requested power.  Uses the closed-form expression for a
     paired t-test with a two-way variance structure.  This lets a reviewer
     answer "how many hospitals is enough" from the paper's own numbers,
     replacing "trust me" with a design-time criterion.

Both functions accept a pandas DataFrame whose rows are (fold, arm, seed,
ood_test_auroc) triples and are careful about small-n behavior --- the paper
sits at n=5 folds, so we report degrees-of-freedom-corrected estimates and
Student-t critical values rather than normal ones.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy import stats


def variance_decomposition(df: pd.DataFrame, metric: str = "ood_test_auroc") -> dict:
    """Two-way random-effects variance components.

    df:   long-form, one row per (fold, arm, seed).
    metric: column to decompose.
    """
    if "arm" in df.columns:
        # Decompose within each arm and also for the combined-arm distribution.
        rows = []
        for arm, g in df.groupby("arm"):
            v = _decompose(g, metric)
            v["arm"] = arm
            rows.append(v)
        v = _decompose(df, metric); v["arm"] = "ALL"
        rows.append(v)
        return {"per_arm": rows}
    return _decompose(df, metric)


def _decompose(df: pd.DataFrame, metric: str) -> dict:
    if len(df) < 3:
        return dict(n=len(df), sigma2_fold=float("nan"),
                    sigma2_seed=float("nan"), icc=float("nan"))
    tab = df.pivot_table(index="fold", columns="seed", values=metric)
    tab = tab.dropna(how="any")
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        # Fall back: only per-fold means, no seed replication.
        s2f = float(np.var(df.groupby("fold")[metric].mean(), ddof=1))
        return dict(n=len(df), n_folds=int(tab.shape[0]),
                    n_seeds=int(tab.shape[1]),
                    sigma2_fold=s2f, sigma2_seed=float("nan"),
                    icc=float("nan"))
    # Two-way ANOVA-style estimators (Searle 1971, eq. 3.10).
    k, s = tab.shape  # folds, seeds
    y = tab.values
    grand = y.mean()
    ms_fold = s * np.sum((y.mean(axis=1) - grand) ** 2) / (k - 1)
    ms_seed = k * np.sum((y.mean(axis=0) - grand) ** 2) / (s - 1)
    ms_err = np.sum((y - y.mean(axis=1, keepdims=True)
                     - y.mean(axis=0, keepdims=True) + grand) ** 2) / ((k - 1) * (s - 1))
    sigma2_err = ms_err
    sigma2_fold = max((ms_fold - ms_err) / s, 0.0)
    sigma2_seed = max((ms_seed - ms_err) / k, 0.0)
    total = sigma2_fold + sigma2_seed + sigma2_err
    return dict(
        n=int(y.size), n_folds=int(k), n_seeds=int(s),
        sigma2_fold=float(sigma2_fold),
        sigma2_seed=float(sigma2_seed),
        sigma2_error=float(sigma2_err),
        icc_fold=float(sigma2_fold / total) if total > 0 else float("nan"),
    )


def min_folds_for_power(sigma_fold: float, sigma_seed: float,
                        effect: float, alpha: float = 0.05,
                        power: float = 0.8, seeds_per_fold: int = 1,
                        two_sided: bool = True) -> dict:
    """How many held-out sites you need to detect a mean-across-sites effect.

    Model: mean effect estimator variance = sigma_fold^2 / K + sigma_seed^2 /
    (K * seeds_per_fold), where K is the number of held-out sites.  Solves
    for the smallest K under a paired t-test with df=K-1.
    """
    def se(K):
        return math.sqrt(sigma_fold ** 2 / K
                         + sigma_seed ** 2 / (K * seeds_per_fold))

    for K in range(2, 200):
        df = K - 1
        tcrit = stats.t.ppf(1 - alpha / (2 if two_sided else 1), df)
        # non-central t power: P(|T| > tcrit | ncp) with ncp = effect/se(K).
        ncp = effect / se(K)
        pow_est = 1.0 - stats.nct.cdf(tcrit, df, ncp) + stats.nct.cdf(-tcrit, df, ncp)
        if pow_est >= power:
            return dict(K=K, power=float(pow_est), se=float(se(K)), ncp=float(ncp))
    return dict(K=None, power=0.0, se=float("nan"), ncp=float("nan"))


def worst_site_bootstrap(df: pd.DataFrame, arm: str,
                         metric: str = "ood_test_auroc",
                         n_boot: int = 10000, rng: int = 0) -> dict:
    """Site-cluster bootstrap for the worst-site value of `arm`.

    Under the paper's design, the natural summary of a site-robustness
    intervention is not its mean but the value at its *worst* held-out site;
    bootstrap-resampling folds gives a variance estimate for that summary.
    """
    r = np.random.default_rng(rng)
    per_fold = df[df["arm"] == arm].groupby("fold")[metric].mean().values
    K = len(per_fold)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = r.integers(0, K, K)
        boots[i] = per_fold[idx].min()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(worst=float(per_fold.min()),
                boot_mean=float(boots.mean()),
                ci_lo=float(lo), ci_hi=float(hi))


if __name__ == "__main__":
    import argparse, json, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/camelyon17_fivehospital/results_folds.csv")
    ap.add_argument("--effect", type=float, default=0.092,
                    help="target mean effect size in AUROC (default: SSL gain)")
    ap.add_argument("--seeds_per_fold", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    # If the file is single-seed (like the shipped one), we still can compute
    # sigma_fold from the per-arm distribution across folds; sigma_seed uses
    # the between-seed value reported in the paper's own limitations (0.035).
    dec = variance_decomposition(df)
    if isinstance(dec, dict) and "per_arm" in dec:
        for row in dec["per_arm"]:
            print(row)
    print()
    # Power scan using the reported sigma_seed of 0.035 AUROC:
    sigma_seed = 0.035
    # Use ERM's across-fold SD as sigma_fold (0.127).
    baseline_row = next(r for r in dec["per_arm"] if r["arm"] == "baseline")
    sigma_fold = math.sqrt(baseline_row["sigma2_fold"] if not math.isnan(baseline_row["sigma2_fold"]) else 0.127**2)
    for eff in (0.05, 0.10, 0.20, a.effect):
        r = min_folds_for_power(sigma_fold, sigma_seed, eff, seeds_per_fold=a.seeds_per_fold)
        print(f"effect={eff:.3f}: minimum K = {r['K']}  (power {r['power']:.2f}, "
              f"SE {r['se']:.3f}, seeds/fold {a.seeds_per_fold}, sigma_fold {sigma_fold:.3f})")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(dec, open(a.out, "w"), indent=2)
