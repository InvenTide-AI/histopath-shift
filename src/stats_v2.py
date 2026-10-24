"""stats_v2 -- statistics for the v2 leave-one-domain-out (LODO) evaluation.

numpy / scipy only (no statsmodels, no pymc).  Every function takes plain arrays;
reading run records lives in ``analyze_v2.py``.

Data convention
---------------
An *arm matrix* ``Y`` holds one metric (AUROC, ECE, ...) for one arm on one cohort:
domain ``t`` (= held-out test domain of fold ``t``) by seed ``s``.  It may be

* a 2-D array of shape (K, S), with ``np.nan`` for missing runs (unbalanced), or
* a list of K 1-D arrays (one per domain, any lengths >= 1).

Seeds are *nested* in domains: seed ``s`` of fold ``t`` and seed ``s`` of fold ``t'``
are different training runs on different data, so the model is the one-way random
effects model ``y_ts = mu + b_t + e_ts`` with ``b_t ~ N(0, sigma_domain^2)`` and
``e_ts ~ N(0, sigma_run^2)`` (Searle, Casella & McCulloch 1992, ch. 3).  We never
fit a crossed seed effect.

Sections (PLAN.md §4)
---------------------
a) per-arm summaries + bootstrap CIs .................. ``arm_summary``, ``bootstrap_summary``
b) one-way random effects: ANOVA, REML, exact-F ICC CI,
   chi-square / MLS CIs, Bayesian grid posterior ...... ``oneway_anova``, ``oneway_reml``,
                                                         ``icc1_exact_ci``, ``bayes_oneway``,
                                                         ``variance_components``
c) paired contrasts, Holm, fold-dependence sensitivity . ``paired_contrast``, ``holm``,
                                                         ``fold_overlap``, ``lodo_design``,
                                                         ``dependence_sensitivity``
d) power for the paired design ........................ ``difference_variance``, ``power_paired_t``,
                                                         ``min_domains``, ``power_table``
e) rank stability ..................................... ``rank_scores``, ``bootstrap_ranks``,
                                                         ``kendall_tau_b``, ``tau_between_cohorts``,
                                                         ``interaction_permutation_test``,
                                                         ``interaction_2x2_cohort``,
                                                         ``cluster_summary``
f) response shape ..................................... ``oldham_pitman_morgan``

References
----------
Bates, S., Hastie, T., Tibshirani, R. (2023). Cross-validation: what does it estimate and
    how well does it do it? JASA 119:1434-1445.
Burdick, R.K., Graybill, F.A. (1992). Confidence Intervals on Variance Components. Dekker.
Gelman, A. (2006). Prior distributions for variance parameters in hierarchical models.
    Bayesian Analysis 1:515-534.
Harville, D.A. (1974). Bayesian inference for variance components using only error
    contrasts. Biometrika 61:383-385.
Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scand J Stat 6:65-70.
James, G.S. (1951). The comparison of several groups of observations when the ratios of the
    population variances are unknown. Biometrika 38:324-329.
Kendall, M.G. (1945). The treatment of ties in ranking problems. Biometrika 33:239-251.
Liu, R.Y. (1988). Bootstrap procedures under some non-i.i.d. models. Ann Stat 16:1696-1708.
Mammen, E. (1993). Bootstrap and wild bootstrap for high dimensional linear models. Ann Stat 21:255-285.
McGraw, K.O., Wong, S.P. (1996). Forming inferences about some intraclass correlation
    coefficients. Psychological Methods 1:30-46.
Nadeau, C., Bengio, Y. (2003). Inference for the generalization error. Machine Learning 52:239-281.
Oldham, P.D. (1962). A note on the analysis of repeated measurements of the same subjects.
    J Chronic Dis 15:969-977.
Pitman, E.J.G. (1939). A note on normal correlation. Biometrika 31:9-12;
Morgan, W.A. (1939). A test for the significance of the difference between the two variances
    in a sample from a normal bivariate population. Biometrika 31:13-19.
Searle, S.R. (1971). Linear Models. Wiley.  Searle, Casella, McCulloch (1992). Variance Components.
Ting, N., Burdick, R.K., Graybill, F.A., Jeyaratnam, S., Lu, T.-F.C. (1990). Confidence intervals
    on linear combinations of variance components that are unrestricted in sign.
    J Stat Comput Simul 35:135-143.
Welch, B.L. (1951). On the comparison of several mean values: an alternative approach.
    Biometrika 38:330-336.
Wu, C.F.J. (1986). Jackknife, bootstrap and other resampling methods in regression analysis.
    Ann Stat 14:1261-1295.
"""
from __future__ import annotations

import functools
import itertools
import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy import optimize, stats

ALPHA = 0.05

# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------


def as_groups(Y) -> List[np.ndarray]:
    """Return a list of 1-D float arrays (one per domain), NaNs dropped.

    Accepts a (K, S) array with NaN for missing cells or a list of 1-D arrays.
    Raises if a domain has no observation.
    """
    if isinstance(Y, np.ndarray) and Y.ndim == 2:
        g = [np.asarray(r, float)[~np.isnan(np.asarray(r, float))] for r in Y]
    else:
        g = [np.asarray(r, float).ravel() for r in Y]
        g = [r[~np.isnan(r)] for r in g]
    for i, r in enumerate(g):
        if r.size == 0:
            raise ValueError(f"domain {i} has no observations")
    return g


def groups_from_summary(means: Sequence[float], sds: Sequence[float], n) -> List[np.ndarray]:
    """Pseudo-data reproducing per-domain (mean, sample SD with ddof=1, n) exactly.

    The one-way ANOVA, REML and the restricted likelihood depend on the data only through
    (n_i, ybar_i, SSW_i), so analysing these pseudo-groups is exact for every §4(b) estimator.
    Used for published tables that report mean +- SE (e.g. DomainBed).  n: int or per-domain list.
    """
    ns = [int(n)] * len(means) if np.isscalar(n) else [int(v) for v in n]
    out = []
    for m, s, k in zip(means, sds, ns):
        if k < 2:
            out.append(np.array([float(m)]))
            continue
        z = np.linspace(-1.0, 1.0, k)
        z = (z - z.mean()) / z.std(ddof=1)
        out.append(float(m) + float(s) * z)
    return out


def as_matrix(Y) -> np.ndarray:
    """(K, S_max) float matrix with NaN padding."""
    g = as_groups(Y)
    S = max(r.size for r in g)
    M = np.full((len(g), S), np.nan)
    for i, r in enumerate(g):
        M[i, : r.size] = r
    return M


def domain_means(Y) -> np.ndarray:
    """Seed-mean per domain (length K)."""
    return np.array([r.mean() for r in as_groups(Y)])


def _is_balanced(groups) -> bool:
    return len({r.size for r in groups}) == 1


def _rng(seed):
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


# ----------------------------------------------------------------------------------------
# a) per-arm summaries
# ----------------------------------------------------------------------------------------


def _worst(D, higher_better: bool = True, axis=-1):
    """Worst domain value along ``axis``: min for higher-is-better metrics, max otherwise."""
    return D.min(axis) if higher_better else D.max(axis)


def mean_t_ci(m: Sequence[float], level: float = 0.95) -> List[float]:
    """t-interval (df = K-1) for the population mean of domain means (one-way model: the
    seed-mean of domain t has variance s2_domain + s2_run/n_t, so for balanced data this is
    exact).  Preferred over the percentile domain bootstrap, which under-covers badly for
    K <= 7 (Davison & Hinkley 1997 §2.4; checked in stats_tests/test_contrasts_power.py)."""
    m = np.asarray(m, float)
    K = m.size
    if K < 2:
        return [float("nan"), float("nan")]
    h = stats.t.ppf(1 - (1 - level) / 2, K - 1) * m.std(ddof=1) / math.sqrt(K)
    return [float(m.mean() - h), float(m.mean() + h)]


def arm_summary(Y, domains: Optional[Sequence[str]] = None, higher_better: bool = True,
                level: float = 0.95) -> dict:
    """Descriptive LODO summary of one arm matrix.

    Returns per-domain seed mean / seed SD / n, the worst domain (minimum seed-mean, or
    maximum when ``higher_better=False``, e.g. ECE), the best domain, the mean of domain
    means with its t-interval over domains (``mean_ci_t``, df = K-1), the across-domain SD
    of domain means (ddof=1), the range, and the mean within-domain seed SD (ddof=1,
    domains with a single seed are skipped).
    """
    g = as_groups(Y)
    K = len(g)
    m = np.array([r.mean() for r in g])
    sd = np.array([r.std(ddof=1) if r.size > 1 else np.nan for r in g])
    iw = int(np.argmin(m)) if higher_better else int(np.argmax(m))
    out = dict(
        K=K,
        n_seeds=[int(r.size) for r in g],
        domain_mean=m.tolist(),
        domain_seed_sd=sd.tolist(),
        worst=float(m[iw]),
        worst_index=iw,
        worst_domain=(domains[iw] if domains is not None else iw),
        best=float(m.max() if higher_better else m.min()),
        higher_better=bool(higher_better),
        mean=float(m.mean()),
        mean_ci_t=mean_t_ci(m, level),
        sd_across=float(m.std(ddof=1)) if K > 1 else float("nan"),
        range=float(m.max() - m.min()),
        mean_within_sd=float(np.nanmean(sd)) if np.isfinite(sd).any() else float("nan"),
    )
    return out


def _resampled_domain_means(g, B, rng, mode):
    """B x K matrix of resampled domain means under a bootstrap ``mode``.

    mode='seed'      : domains fixed, seeds resampled with replacement within each domain
                       (run-to-run uncertainty only; Davison & Hinkley 1997 §3.8 stratified).
    mode='domain'    : domains resampled with replacement, seed means kept
                       (domain-population uncertainty; cluster bootstrap).
    mode='two_stage' : domains resampled, then seeds resampled independently within each
                       chosen domain slot, so a domain drawn twice gets two independent seed
                       resamples (Davison & Hinkley 1997 §3.8, two-stage strategy 1).
    """
    K = len(g)
    if mode == "seed":
        out = np.empty((B, K))
        for i, r in enumerate(g):
            idx = rng.integers(0, r.size, size=(B, r.size))
            out[:, i] = r[idx].mean(1)
        return out
    dom = rng.integers(0, K, size=(B, K))
    if mode == "domain":
        m = np.array([r.mean() for r in g])
        return m[dom]
    if mode == "two_stage":
        out = np.empty((B, K))
        for i, r in enumerate(g):
            sel = dom == i  # every (replicate, slot) that drew domain i
            n_i = int(sel.sum())
            if n_i:
                out[sel] = r[rng.integers(0, r.size, size=(n_i, r.size))].mean(1)
        return out
    raise ValueError(mode)


def bootstrap_summary(Y, B: int = 10000, modes=("seed", "domain", "two_stage"),
                      level: float = 0.95, seed=0, higher_better: bool = True) -> dict:
    """Percentile bootstrap CIs for worst-domain and mean-over-domains.

    Hierarchical data: the 'seed' mode resamples seeds within domain (the uncertainty of the
    *observed* domains' values), 'domain' resamples domains (generalisation to new domains of
    the same population), 'two_stage' does both.  The worst-domain statistic under domain
    resampling is reported for completeness but is not a well-behaved target with K<=7, and
    the percentile 'domain' / 'two_stage' CIs for the mean under-cover for K<=7 (use
    ``mean_t_ci`` as the headline interval).  ``higher_better=False``: worst = max.
    """
    g = as_groups(Y)
    rng = _rng(seed)
    a = (1 - level) / 2
    res = {}
    for mode in modes:
        D = _resampled_domain_means(g, B, rng, mode)
        w, mu = _worst(D, higher_better, 1), D.mean(1)
        res[mode] = dict(
            worst_ci=np.quantile(w, [a, 1 - a]).tolist(),
            mean_ci=np.quantile(mu, [a, 1 - a]).tolist(),
            worst_se=float(w.std(ddof=1)),
            mean_se=float(mu.std(ddof=1)),
        )
    return res


# ----------------------------------------------------------------------------------------
# b) one-way random effects model
# ----------------------------------------------------------------------------------------


def oneway_anova(Y) -> dict:
    """Method-of-moments (ANOVA) fit of y_ts = mu + b_t + e_ts (balanced or unbalanced).

    For unbalanced data the ANOVA estimator uses n0 = (N - sum n_i^2 / N) / (k - 1)
    (Searle 1971, §10.6; Searle, Casella & McCulloch 1992, §3.5).  Returns raw (possibly
    negative) and truncated estimates, F = MSB/MSW with p-value, and ICC(1) =
    (MSB - MSW) / (MSB + (n0 - 1) MSW) (McGraw & Wong 1996, case 1).  With a single seed
    per domain (N - k = 0) MSW, F, p and the ICC are NaN (not identifiable).
    """
    g = as_groups(Y)
    k = len(g)
    n = np.array([r.size for r in g], float)
    N = n.sum()
    ybar = np.array([r.mean() for r in g])
    grand = sum(r.sum() for r in g) / N
    ssb = float((n * (ybar - grand) ** 2).sum())
    ssw = float(sum(((r - r.mean()) ** 2).sum() for r in g))
    dfb, dfw = k - 1, N - k
    msb = ssb / dfb if dfb > 0 else float("nan")
    msw = ssw / dfw if dfw > 0 else float("nan")
    n0 = (N - (n ** 2).sum() / N) / (k - 1) if k > 1 else float("nan")
    s2b_raw = (msb - msw) / n0
    if dfw <= 0 or dfb <= 0:  # one seed per domain (or one domain): F undefined, not infinite
        F, p = float("nan"), float("nan")
    elif msw > 0:
        F = msb / msw
        p = float(stats.f.sf(F, dfb, dfw))
    else:  # replicates agree exactly within every domain
        F, p = (float("inf"), 0.0) if msb > 0 else (float("nan"), float("nan"))
    icc_raw = (msb - msw) / (msb + (n0 - 1) * msw) if (msb + (n0 - 1) * msw) > 0 else float("nan")
    return dict(
        k=k, N=int(N), n0=float(n0), balanced=_is_balanced(g), grand_mean=float(grand),
        ssb=ssb, ssw=ssw, dfb=int(dfb), dfw=int(dfw), msb=msb, msw=msw, F=F, p=p,
        s2_domain_raw=float(s2b_raw), s2_domain=max(float(s2b_raw), 0.0), s2_run=msw,
        sigma_domain=math.sqrt(max(s2b_raw, 0.0)), sigma_run=math.sqrt(msw),
        icc_raw=float(icc_raw), icc=max(float(icc_raw), 0.0), truncated=bool(s2b_raw < 0),
    )


def _reml_profile_terms(g, gamma):
    """Pieces of the one-way REML criterion at variance ratio gamma = s2b / s2w.

    -2 l_R(s2b, s2w) = (N-1) log s2w + sum log(1 + n_i gamma) + log sum w_i + Q(gamma)/s2w,
    w_i = n_i / (1 + n_i gamma), Q = SSW + sum w_i (ybar_i - mu_hat)^2 (Harville 1974;
    Searle, Casella & McCulloch 1992 §6.6).  Vectorised over gamma (any shape).
    """
    n = np.array([r.size for r in g], float)
    ybar = np.array([r.mean() for r in g])
    ssw = sum(((r - r.mean()) ** 2).sum() for r in g)
    gam = np.asarray(gamma, float)[..., None]
    w = n / (1.0 + n * gam)
    mu = (w * ybar).sum(-1, keepdims=True) / w.sum(-1, keepdims=True)
    Q = ssw + (w * (ybar - mu) ** 2).sum(-1)
    logdet = np.log1p(n * gam).sum(-1)
    logW = np.log(w.sum(-1))
    return Q, logdet, logW, n.sum()


def oneway_reml(Y) -> dict:
    """REML estimates of sigma_domain^2, sigma_run^2 for the one-way model.

    The REML criterion is profiled over s2w (closed form s2w = Q(gamma)/(N-1)) and minimised
    over gamma >= 0 by a log-grid scan + bounded Brent refinement, with the boundary gamma=0
    compared explicitly.  For balanced data with positive ANOVA estimate REML equals ANOVA;
    at the boundary REML gives s2b = 0 and s2w = SST/(N-1) (Searle et al. 1992, §6.7).
    The upper end of the log-gamma grid adapts to the data (log(MSB/MSW) + 8, at least 12) and
    is extended further if the minimum sits on the last grid point; a warning is issued if it
    still does.  Requires N - k >= 1 (at least one domain with two seeds).
    """
    g = as_groups(Y)
    a = oneway_anova(g)
    nan = float("nan")
    if a["dfw"] <= 0 or a["k"] < 2:  # no within-domain replication: not identifiable
        return dict(s2_domain=nan, s2_run=nan, sigma_domain=nan, sigma_run=nan, icc=nan, boundary=False,
                    neg2loglik=nan)
    if a["ssw"] <= 0:  # replicates identical within every domain: s2w = 0, s2b = var of domain means
        s2b = float(np.var([r.mean() for r in g], ddof=1))
        return dict(s2_domain=s2b, s2_run=0.0, sigma_domain=math.sqrt(s2b), sigma_run=0.0,
                    icc=1.0 if s2b > 0 else nan, boundary=False, neg2loglik=float("-inf"))

    def crit(gam):
        Q, ld, lW, N = _reml_profile_terms(g, gam)
        return (N - 1) * np.log(Q / (N - 1)) + ld + lW

    top = 12.0
    if np.isfinite(a["msb"]) and a["msb"] > a["msw"] > 0:
        top = max(top, math.log(a["msb"] / a["msw"]) + 8.0)
    for _ in range(6):
        lg = np.linspace(-25, top, 400 + int(max(0.0, top - 12.0) * 10))
        vals = crit(np.exp(lg))
        i = int(np.argmin(vals))
        if i < len(lg) - 1:
            break
        top += 15.0
    else:
        import warnings
        warnings.warn("oneway_reml: REML optimum at the upper end of the gamma grid; estimates may be capped")
    lo, hi = lg[max(i - 1, 0)], lg[min(i + 1, len(lg) - 1)]
    r = optimize.minimize_scalar(lambda t: float(crit(np.exp(t))), bounds=(lo, hi), method="bounded",
                                 options=dict(xatol=1e-10))
    best_gam, best = math.exp(r.x), float(r.fun)
    c0 = float(crit(0.0))
    if c0 <= best + 1e-12 or i == 0:
        best_gam, best = 0.0, c0
    Q, _, _, N = _reml_profile_terms(g, best_gam)
    s2w = float(Q / (N - 1))
    s2b = best_gam * s2w
    return dict(s2_domain=s2b, s2_run=s2w, sigma_domain=math.sqrt(s2b), sigma_run=math.sqrt(s2w),
                icc=s2b / (s2b + s2w) if s2b + s2w > 0 else float("nan"),
                boundary=bool(best_gam == 0.0), neg2loglik=best)


def icc1_exact_ci(Y, level: float = 0.95) -> dict:
    """Exact F-based CI for ICC(1) (Searle 1971 §9.7; McGraw & Wong 1996, Table 7 case 1).

    F_L = F / F_{1-a/2}(k-1, N-k), F_U = F / F_{a/2}(k-1, N-k);
    lower = (F_L - 1)/(F_L + n - 1), upper = (F_U - 1)/(F_U + n - 1).
    Exact for balanced data; for unbalanced data n is replaced by n0 (approximate;
    Donner 1986).  Bounds are reported raw and clipped to [0, 1].
    """
    a = oneway_anova(Y)
    al = (1 - level) / 2
    n = a["n0"]
    F = a["F"]
    if np.isnan(F):  # not identifiable (single seed per domain, or a single domain)
        nan = float("nan")
        return dict(icc=nan, icc_raw=nan, lo=nan, hi=nan, lo_raw=nan, hi_raw=nan, exact=a["balanced"])
    if np.isinf(F):  # MSW = 0 with dfw > 0: all variance between domains
        return dict(icc=1.0, icc_raw=1.0, lo=1.0, hi=1.0, lo_raw=1.0, hi_raw=1.0, exact=a["balanced"])
    FL = F / stats.f.ppf(1 - al, a["dfb"], a["dfw"])
    FU = F / stats.f.ppf(al, a["dfb"], a["dfw"])
    lo = (FL - 1) / (FL + n - 1)
    hi = (FU - 1) / (FU + n - 1)
    return dict(icc=a["icc"], icc_raw=a["icc_raw"], lo=float(np.clip(lo, 0, 1)), hi=float(np.clip(hi, 0, 1)),
                lo_raw=float(lo), hi_raw=float(hi), exact=a["balanced"])


def sigma_run_ci(Y, level: float = 0.95) -> List[float]:
    """Exact chi-square CI for sigma_run: SSW/sigma^2 ~ chi2(N-k)."""
    a = oneway_anova(Y)
    al = (1 - level) / 2
    if a["dfw"] <= 0:
        return [float("nan"), float("nan")]
    lo = math.sqrt(a["ssw"] / stats.chi2.ppf(1 - al, a["dfw"]))
    hi = math.sqrt(a["ssw"] / stats.chi2.ppf(al, a["dfw"]))
    return [lo, hi]


def sigma_domain_mls_ci(Y, level: float = 0.95) -> List[float]:
    """Modified-large-sample (MLS) CI for sigma_domain (Ting et al. 1990; Burdick & Graybill
    1992 §3.3) for theta = (MSB - MSW)/n.  Uses n0 for unbalanced data (approximate).
    Bounds truncated at 0 and returned on the SD scale.
    """
    a = oneway_anova(Y)
    al = (1 - level) / 2
    d1, d2, n = a["dfb"], a["dfw"], a["n0"]
    if d1 <= 0 or d2 <= 0:
        return [float("nan"), float("nan")]
    S1, S2 = a["msb"], a["msw"]
    F1 = stats.chi2.ppf(1 - al, d1) / d1  # F_{1-al; d1, inf}
    F2 = stats.chi2.ppf(al, d1) / d1
    F3 = stats.chi2.ppf(1 - al, d2) / d2
    F4 = stats.chi2.ppf(al, d2) / d2
    F5 = stats.f.ppf(1 - al, d1, d2)
    F6 = stats.f.ppf(al, d1, d2)
    G1, H1 = 1 - 1 / F1, 1 / F2 - 1
    G2, H2 = 1 - 1 / F3, 1 / F4 - 1
    G12 = ((F5 - 1) ** 2 - G1 ** 2 * F5 ** 2 - H2 ** 2) / F5
    H12 = ((1 - F6) ** 2 - H1 ** 2 * F6 ** 2 - G2 ** 2) / F6
    VL = G1 ** 2 * S1 ** 2 + H2 ** 2 * S2 ** 2 + G12 * S1 * S2
    VU = H1 ** 2 * S1 ** 2 + G2 ** 2 * S2 ** 2 + H12 * S1 * S2
    L = (S1 - S2 - math.sqrt(max(VL, 0))) / n
    U = (S1 - S2 + math.sqrt(max(VU, 0))) / n
    return [math.sqrt(max(L, 0.0)), math.sqrt(max(U, 0.0))]


def _log_prior(sig, kind, scale):
    """Log density (up to a constant) of a half-normal or half-Cauchy prior on an SD
    (Gelman 2006).  kind in {'halfnormal', 'halfcauchy', 'flat'}."""
    if kind == "halfnormal":
        return -0.5 * (sig / scale) ** 2
    if kind == "halfcauchy":
        return -np.log1p((sig / scale) ** 2)
    if kind == "flat":
        return np.zeros_like(sig)
    raise ValueError(kind)


def _trapz_w(x):
    """Trapezoid quadrature weights for a 1-D grid."""
    w = np.zeros_like(x)
    d = np.diff(x)
    w[:-1] += d / 2
    w[1:] += d / 2
    return w


def _quantiles_from_mass(values, mass, qs):
    o = np.argsort(values, axis=None)
    v = values.ravel()[o]
    c = np.cumsum(mass.ravel()[o])
    c /= c[-1]
    return [float(np.interp(q, c, v)) for q in qs]


def bayes_oneway(Y, prior: str = "halfnormal", scale_domain: float = 0.1, scale_run: Optional[float] = None,
                 n_domain: int = 400, n_run: int = 300, level: float = 0.95) -> dict:
    """Grid posterior for (sigma_domain, sigma_run) in the one-way model.

    Likelihood: the REML / restricted likelihood, i.e. the marginal likelihood after
    integrating mu under a flat prior (Harville 1974); valid for unbalanced data.
    Priors: independent half-normal or half-Cauchy on each SD with the given scales
    (Gelman 2006); ``scale_run`` defaults to ``scale_domain``.  Integration on a 2-D
    tensor grid (sigma_domain: 0 plus a geometric grid spanning prior and data ranges;
    sigma_run: geometric grid around the chi-square range of SSW), trapezoid weights.

    Returns posterior median, mean and equal-tailed ``level`` credible intervals for
    sigma_domain, sigma_run and ICC = s2b/(s2b+s2w), plus one-sided upper bounds
    [0, q_level] which are the appropriate summary when the posterior piles up near 0
    (equal-tailed intervals never contain exactly 0).
    """
    g = as_groups(Y)
    a = oneway_anova(Y)
    scale_run = scale_domain if scale_run is None else scale_run
    N, k = a["N"], a["k"]
    dfw = a["dfw"]
    ssw = a["ssw"]
    eps = 1e-9
    # sigma_run grid
    if dfw > 0 and ssw > 0:
        wlo = math.sqrt(ssw / stats.chi2.ppf(1 - eps, dfw)) * 0.5
        whi = math.sqrt(ssw / stats.chi2.ppf(eps, dfw)) * 1.5
        whi = max(whi, math.sqrt(a["msb"]) * 1.5)
    else:
        wlo, whi = 1e-6 * scale_run, 20 * scale_run
    sw = np.geomspace(wlo, whi, n_run)
    # sigma_domain grid
    msb = max(a["msb"], 1e-300)
    nmin = min(r.size for r in g)
    dhi = math.sqrt(msb * a["dfb"] / stats.chi2.ppf(1e-7, a["dfb"]) / nmin)
    if prior != "flat":
        pq = {"halfnormal": 6.0, "halfcauchy": 1e4}[prior] * scale_domain  # prior tail mass < 1e-8 / 6e-5
        dhi = min(dhi, max(pq, 10 * math.sqrt(msb / nmin)))
    dlo = min(wlo, math.sqrt(msb / nmin)) * 1e-4
    sb = np.concatenate([[0.0], np.geomspace(dlo, dhi, n_domain - 1)])
    SB, SW = np.meshgrid(sb, sw, indexing="ij")
    gam = (SB / SW) ** 2
    Q, ld, lW, _ = _reml_profile_terms(g, gam)
    s2w = SW ** 2
    ll = -0.5 * ((N - 1) * np.log(s2w) + ld + lW + Q / s2w)
    lp = ll + _log_prior(SB, prior, scale_domain) + _log_prior(SW, prior, scale_run)
    lp -= lp.max()
    dens = np.exp(lp)
    mass = dens * np.outer(_trapz_w(sb), _trapz_w(sw))
    mass /= mass.sum()
    al = (1 - level) / 2
    icc = SB ** 2 / (SB ** 2 + SW ** 2)

    qs = [al, 0.5, 1 - al, level]

    def summ_marg(x, marg):
        # marginal density on a 1-D grid -> cumulative trapezoid CDF -> interpolated quantiles
        c = np.concatenate([[0.0], np.cumsum(np.diff(x) * (marg[1:] + marg[:-1]) / 2)])
        mean = float((x * marg * _trapz_w(x)).sum() / c[-1])
        q = [float(np.interp(p, c / c[-1], x)) for p in qs]
        return dict(median=q[1], mean=mean, lo=q[0], hi=q[2], upper_one_sided=q[3])

    def summ_mass(vals):
        q = _quantiles_from_mass(vals, mass, qs)
        return dict(median=q[1], mean=float((vals * mass).sum()), lo=q[0], hi=q[2], upper_one_sided=q[3])

    out = dict(prior=prior, scale_domain=scale_domain, scale_run=scale_run,
               sigma_domain=summ_marg(sb, dens @ _trapz_w(sw)), sigma_run=summ_marg(sw, _trapz_w(sb) @ dens),
               icc=summ_mass(icc),
               grid=dict(domain=[float(sb[1]), float(sb[-1])], run=[float(sw[0]), float(sw[-1])]))
    # edge mass diagnostic: posterior mass in outermost 1% of grid cells
    out["edge_mass"] = float(mass[-max(1, n_domain // 100):, :].sum() + mass[:, -max(1, n_run // 100):].sum()
                             + mass[:, : max(1, n_run // 100)].sum())
    return out


def variance_components(Y, level: float = 0.95, priors=(("halfnormal", 0.1), ("halfcauchy", 0.1)),
                        bayes_grid=(400, 300)) -> dict:
    """All §4(b) estimates for one arm matrix: ANOVA, REML, exact ICC CI, chi-square CI for
    sigma_run, MLS CI for sigma_domain, and Bayesian posteriors for each prior in ``priors``
    (list of (kind, scale) pairs; scale applies to both SDs)."""
    out = dict(anova=oneway_anova(Y), reml=oneway_reml(Y), icc_ci=icc1_exact_ci(Y, level),
               sigma_run_ci=sigma_run_ci(Y, level), sigma_domain_ci=sigma_domain_mls_ci(Y, level))
    out["bayes"] = {f"{kind}({scale:g})": bayes_oneway(Y, kind, scale, n_domain=bayes_grid[0], n_run=bayes_grid[1],
                                                        level=level) for kind, scale in priors}
    return out


def simulate_oneway(K: int, S: int, icc: float, sigma_total: float = 1.0, mu: float = 0.0, rng=None) -> np.ndarray:
    """Draw a K x S matrix from the one-way model with the given ICC and total SD."""
    rng = _rng(rng)
    sb, sw = sigma_total * math.sqrt(icc), sigma_total * math.sqrt(1 - icc)
    return mu + rng.normal(0, sb, (K, 1)) + rng.normal(0, sw, (K, S))


# ----------------------------------------------------------------------------------------
# c) paired contrasts and fold dependence
# ----------------------------------------------------------------------------------------


def paired_contrast(Ya, Yr, level: float = 0.95, higher_better: bool = True) -> dict:
    """Paired comparison of arm a vs reference r across domains.

    d_t = (seed-mean of a at domain t) - (seed-mean of r at domain t); reports mean d, SD,
    t-interval with K-1 df, two-sided p, and number of domains improved (d_t > 0, or d_t < 0
    when ``higher_better=False``) / worse.
    ``Ya``/``Yr`` may be arm matrices or per-domain mean vectors.
    """
    d = domain_means(Ya) - domain_means(Yr)
    K = d.size
    m, s = float(d.mean()), float(d.std(ddof=1))
    se = s / math.sqrt(K)
    tc = stats.t.ppf(1 - (1 - level) / 2, K - 1)
    t = m / se if se > 0 else (float("inf") if m != 0 else 0.0)
    p = float(2 * stats.t.sf(abs(t), K - 1)) if np.isfinite(t) else 0.0
    return dict(K=K, d=d.tolist(), mean=m, sd=s, se=se, t=t, df=K - 1, p=p,
                ci=[m - tc * se, m + tc * se], n_improved=int(((d > 0) if higher_better else (d < 0)).sum()),
                n_worse=int(((d < 0) if higher_better else (d > 0)).sum()))


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values (monotone, capped at 1); NaNs passed through."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    m = q.size
    o = np.argsort(q)
    adj = np.maximum.accumulate((m - np.arange(m)) * q[o])
    res = np.empty(m)
    res[o] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def lodo_design(K: int, val_offset: Optional[int] = -1) -> List[set]:
    """Training-domain sets of a K-fold leave-one-domain-out design.

    Fold t tests on domain t; if ``val_offset`` is not None the validation domain is
    (t + val_offset) mod K and is excluded from training (PLAN.md §2: val = (t-1) mod K).
    """
    out = []
    for t in range(K):
        excl = {t} if val_offset is None else {t, (t + val_offset) % K}
        out.append(set(range(K)) - excl)
    return out


def fold_overlap(train_sets: Sequence[Iterable]) -> np.ndarray:
    """overlap_ij = |T_i ∩ T_j| / sqrt(|T_i| |T_j|) (= |T_i ∩ T_j|/|T| for equal-size sets)."""
    T = [set(x) for x in train_sets]
    K = len(T)
    O = np.empty((K, K))
    for i in range(K):
        for j in range(K):
            O[i, j] = len(T[i] & T[j]) / math.sqrt(len(T[i]) * len(T[j]))
    return O


def dependence_matrix(train_sets, rho: float) -> np.ndarray:
    """Working correlation R_ij = rho * overlap_ij (i != j), R_ii = 1."""
    O = fold_overlap(train_sets)
    R = rho * O
    np.fill_diagonal(R, 1.0)
    return R


def k_eff(train_sets, rho: float) -> float:
    """Effective number of independent folds K_eff = K^2 / sum_ij R_ij (variance of the mean of
    K equicorrelated-like terms: Var(mean) = sigma^2 sum(R) / K^2)."""
    R = dependence_matrix(train_sets, rho)
    return R.shape[0] ** 2 / R.sum()


def dependence_sensitivity(d: Sequence[float], train_sets, rhos=(0.0, 0.25, 0.5, 1.0), level: float = 0.95) -> List[dict]:
    """Fold-dependence sensitivity analysis for a mean over LODO folds.

    Assumes corr(d_i, d_j) = rho * overlap_ij (training-set overlap of folds i, j).  Then
    Var(d_bar) = sigma^2 sum(R)/K^2 and E[s^2] = sigma^2 (1 - mean offdiag R), so the corrected
    variance is s^2/(1 - rbar) * sum(R)/K^2 (cf. the corrected resampled t of Nadeau & Bengio
    2003, and Bates, Hastie & Tibshirani 2023 on CV's correlated folds).  t on K-1 df.
    """
    d = np.asarray(d, float)
    K = d.size
    m, s2 = d.mean(), d.var(ddof=1)
    tc = stats.t.ppf(1 - (1 - level) / 2, K - 1)
    out = []
    for rho in rhos:
        R = dependence_matrix(train_sets, rho)
        rbar = (R.sum() - K) / (K * (K - 1))
        sig2 = s2 / (1 - rbar) if rbar < 1 else float("inf")
        se = math.sqrt(sig2 * R.sum() / K ** 2)
        t = m / se if se > 0 else float("inf")
        out.append(dict(rho=rho, K_eff=K ** 2 / R.sum(), mean_offdiag_R=rbar, se=se, t=t,
                        p=float(2 * stats.t.sf(abs(t), K - 1)), ci=[m - tc * se, m + tc * se]))
    return out


# ----------------------------------------------------------------------------------------
# d) power
# ----------------------------------------------------------------------------------------


def difference_variance(Ya, Yr, paired: bool = False) -> dict:
    """Components of Var(seed-averaged per-domain difference) for arm a vs reference r.

    Model: y^a_ts = mu_a + b_t + (ab)_t + e^a_ts, y^r_ts = mu_r + b_t + e^r_ts, so
    Var(d_t) = sigma2_dxa + sigma2_run,diff/s with sigma2_run,diff = Var(e^a_ts - e^r_ts).
    The domain main effect b_t cancels (do NOT use the baseline's sigma_domain for paired power).

    paired=False : the run noises of the two arms are assumed INDEPENDENT (different seeds, or
                   seeds that do not induce shared randomness), so sigma2_run,diff = s2_run,a +
                   s2_run,r with s2_run,* = pooled within-domain MSW of each arm, and
                   sigma2_dxa = max(var(d_t) - MSW_a*h_a - MSW_r*h_r, 0), h = mean_t(1/n_t).
    paired=True  : Ya[t][j] and Yr[t][j] come from the same seed / the same trained model (e.g. a
                   test-time-adaptation pseudo-arm and its base arm, or arms trained with identical
                   seeds -> same init and batch order).  Their run noises are correlated, so
                   sigma2_run,diff is estimated directly as the pooled within-domain variance of
                   the per-seed differences a_tj - r_tj, and sigma2_dxa = max(var(d_t) -
                   s2_run,diff * h, 0).  Requires equal, aligned seed sets per domain.  This is
                   unbiased whether or not the noises are correlated.
    """
    ga, gr = as_groups(Ya), as_groups(Yr)
    d = domain_means(ga) - domain_means(gr)
    va = oneway_anova(ga)["msw"]
    vr = oneway_anova(gr)["msw"]
    ha = float(np.mean([1 / r.size for r in ga]))
    hr = float(np.mean([1 / r.size for r in gr]))
    vd = float(d.var(ddof=1))
    out = dict(K=int(d.size), var_d_observed=vd, sd_d_observed=math.sqrt(vd), s2_run_arm=va, s2_run_ref=vr,
               seeds_harmonic_arm=1 / ha, seeds_harmonic_ref=1 / hr, paired=bool(paired))
    if paired:
        if any(x.size != y.size for x, y in zip(ga, gr)):
            raise ValueError("paired=True needs the same number of (aligned) seeds per domain in both arms")
        gd = [x - y for x, y in zip(ga, gr)]
        vdiff = oneway_anova(gd)["msw"]
        raw = vd - vdiff * ha
        out.update(s2_run_diff=vdiff, s2_run_corr=(va + vr - vdiff) / (2 * math.sqrt(va * vr)) if va * vr > 0 else
                   float("nan"))
    else:
        raw = vd - va * ha - vr * hr
        out.update(s2_run_diff=va + vr, s2_run_corr=0.0)
    out.update(s2_dxa_raw=raw, s2_dxa=max(raw, 0.0))
    return out


def sd_delta(comp: dict, s: int) -> float:
    """sigma_Delta at s seeds per arm: sqrt(s2_dxa + s2_run,diff/s) (s2_run,diff = s2_run,a + s2_run,r
    for independent run noise; the paired estimate for seed-paired arms, see difference_variance)."""
    v = comp.get("s2_run_diff", comp["s2_run_arm"] + comp["s2_run_ref"])
    return math.sqrt(comp["s2_dxa"] + v / s)


def power_paired_t(K: int, delta: float, sd: float, alpha: float = ALPHA, keff: Optional[float] = None) -> float:
    """Power of the two-sided paired t test over K domains (noncentral t; df = K-1).

    ncp = delta / (sd / sqrt(K_eff)), K_eff = K unless supplied (fold-dependence inflation of
    the variance of the mean; Nadeau & Bengio 2003)."""
    if K < 2:
        return float("nan")
    ke = K if keff is None else keff
    df = K - 1
    ncp = delta * math.sqrt(ke) / sd
    tc = stats.t.ppf(1 - alpha / 2, df)
    with np.errstate(divide="ignore", invalid="ignore"):  # scipy boost nct warns harmlessly in the tails
        return float(stats.nct.sf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp))


def _power_vec(Ks: np.ndarray, delta: float, sd: float, alpha: float, ke: np.ndarray) -> np.ndarray:
    """Vectorised ``power_paired_t`` over arrays of K and K_eff."""
    df = Ks - 1.0
    ncp = delta * np.sqrt(ke) / sd
    tc = stats.t.ppf(1 - alpha / 2, df)
    with np.errstate(divide="ignore", invalid="ignore"):
        return stats.nct.sf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp)


def min_domains(delta: float, sd: float, power: float = 0.8, alpha: float = ALPHA, kmax: int = 1000,
                keff_fn: Optional[Callable[[int], float]] = None) -> Optional[int]:
    """Smallest K in [2, kmax] with power >= ``power``; None if never reached.

    ``keff_fn(K)`` optionally returns the effective number of folds for a K-domain design
    (e.g. ``lambda K: k_eff(lodo_design(K), rho)``); under fold dependence the power can
    plateau below the target, in which case None is returned.  (Vectorised over K; identical
    to scanning ``power_paired_t``.)"""
    Ks = np.arange(2, kmax + 1)
    ke = Ks.astype(float) if keff_fn is None else np.array([keff_fn(int(K)) for K in Ks], float)
    ok = np.isfinite(ke)
    if not ok.any():
        return None
    pw = np.full(Ks.size, -1.0)
    pw[ok] = _power_vec(Ks[ok].astype(float), delta, sd, alpha, ke[ok])
    hit = np.nonzero(pw >= power)[0]
    return int(Ks[hit[0]]) if hit.size else None


def keff_lodo_fn(rho: float, val_offset: Optional[int] = -1) -> Callable[[int], float]:
    """K -> K_eff for a K-domain LODO design (training = all but test [and val]).

    Closed form (all folds the same size m = K - 1 - [val]): for the val design, a pair of
    folds shares m - 2 domains unless one fold's test is the other's val (shares m - 1).
    Computed exactly via fold_overlap for small K, by the closed form for large K.
    """
    @functools.lru_cache(maxsize=None)
    def f(K):
        if K < (2 if val_offset is None else 3):
            return float("nan")  # design infeasible (no training domain left)
        if K <= 60:
            return k_eff(lodo_design(K, val_offset), rho)
        if val_offset is None:
            m = K - 1
            off = K * (K - 1) * (m - 1) / m
        else:
            m = K - 2
            adj = 2 * K  # ordered pairs (t, t+-1)
            off = (adj * (m - 1) + (K * (K - 1) - adj) * (m - 2)) / m
        return K ** 2 / (K + rho * off)
    return f


def power_table(comp: dict, deltas=(0.02, 0.05, 0.10, 0.20), seeds=(1, 3, 5, 10), rhos=(0.0, 0.25, 0.5),
                val_offset: Optional[int] = -1, power: float = 0.8, alpha: float = ALPHA, kmax: int = 1000,
                train_sets=None, keff_modes=("ratio", "design")) -> List[dict]:
    """Table of min K over delta x s (x rho, x K_eff mode) from difference components.

    K_eff inflation for a hypothetical K-domain study under corr = rho * overlap:
      'ratio'  : the variance-inflation factor K/K_eff of the OBSERVED design (``train_sets``) is
                 held fixed, K_eff(K) = K / VIF  (default; mirrors Nadeau & Bengio's constant
                 correction factor);
      'design' : K_eff of a K-domain LODO design with all-but-test[-and-val] training sets.  Since
                 overlap -> 1 as K grows, K_eff(K) -> 1/rho: under a fixed rho no number of domains
                 can beat ~1/rho independent folds (min K is then often None = unreachable).
    rho = 0 rows (independent folds) are emitted once with keff_mode 'none'.

    Variance correction (consistent with ``dependence_sensitivity``): under corr(d_i, d_j) =
    rho * overlap_ij the observed SD of d is biased low, E[s^2] = sigma^2 (1 - rbar_obs), where
    rbar_obs is the mean off-diagonal working correlation of the OBSERVED design (``train_sets``;
    if None, a LODO design with ``comp['K']`` domains and ``val_offset``).  For rho > 0 the
    seed-extrapolated sigma_Delta is therefore divided by sqrt(1 - rbar_obs) in BOTH modes (the
    hypothetical design enters only through K_eff(K)).  The correlation is applied to the whole
    seed-averaged difference, as in the CI band; at the observed number of seeds the implied SE
    at the observed K equals the ``dependence_sensitivity`` SE exactly.
    """
    rows = []
    if train_sets is None and "K" in comp:
        train_sets = lodo_design(int(comp["K"]), val_offset)
    for s in seeds:
        sd0 = sd_delta(comp, s)
        for rho in rhos:
            modes = ("none",) if rho == 0 else keff_modes
            rbar = 0.0
            if rho > 0 and train_sets is not None:
                Ko = len(train_sets)
                R = dependence_matrix(train_sets, rho)
                rbar = float((R.sum() - Ko) / (Ko * (Ko - 1)))
            sd = sd0 / math.sqrt(1 - rbar) if rbar < 1 else float("inf")
            for mode in modes:
                if mode == "none":
                    fn = None
                elif mode == "design":
                    fn = keff_lodo_fn(rho, val_offset)
                elif mode == "ratio":
                    if train_sets is None:
                        continue
                    vif = len(train_sets) / k_eff(train_sets, rho)
                    fn = (lambda v: (lambda K: K / v))(vif)
                else:
                    raise ValueError(mode)
                for dl in deltas:
                    rows.append(dict(seeds=s, rho=rho, keff_mode=mode, delta=dl, sd_delta=sd, sd_delta_uncorrected=sd0,
                                     rbar_obs=rbar,
                                     K_min=min_domains(dl, sd, power, alpha, kmax, fn) if np.isfinite(sd) else None))
    return rows


# ----------------------------------------------------------------------------------------
# e) rank stability
# ----------------------------------------------------------------------------------------


def rank_scores(scores: Dict[str, float], higher_better: bool = True) -> Dict[str, float]:
    """Rank arms (1 = best), average ranks for ties."""
    arms = list(scores)
    v = np.array([scores[a] for a in arms], float)
    r = stats.rankdata(-v if higher_better else v, method="average")
    return dict(zip(arms, r.tolist()))


def _stat(D, key, higher_better: bool = True):
    return _worst(D, higher_better, -1) if key == "worst" else D.mean(-1)


def bootstrap_ranks(arms: Dict[str, object], key: str = "worst", mode: str = "seed", B: int = 5000,
                    higher_better: bool = True, level: float = 0.95, seed=0) -> Dict[str, dict]:
    """Bootstrap rank intervals for arms on one cohort.

    mode='seed'  : seeds resampled within each (arm, domain) independently;
    mode='domain': domains resampled with replacement, the SAME domain draw for every arm
                   (paired; requires all arms on the same K domains).
    key: 'worst' (min over domains of seed means; max if ``higher_better=False``) or 'mean'.
    Returns observed rank, median rank, percentile interval and P(rank = 1) per arm.
    """
    rng = _rng(seed)
    names = list(arms)
    G = {a: as_groups(arms[a]) for a in names}
    K = len(G[names[0]])
    S = np.empty((B, len(names)))
    if mode == "seed":
        for j, a in enumerate(names):
            S[:, j] = _stat(_resampled_domain_means(G[a], B, rng, "seed"), key, higher_better)
    elif mode == "domain":
        dom = rng.integers(0, K, size=(B, K))
        for j, a in enumerate(names):
            m = domain_means(G[a])
            S[:, j] = _stat(m[dom], key, higher_better)
    else:
        raise ValueError(mode)
    R = stats.rankdata(-S if higher_better else S, axis=1, method="average")
    obs = rank_scores({a: float(_stat(domain_means(G[a])[None], key, higher_better)[0]) for a in names}, higher_better)
    al = (1 - level) / 2
    return {a: dict(rank=obs[a], median=float(np.median(R[:, j])),
                    lo=float(np.quantile(R[:, j], al)), hi=float(np.quantile(R[:, j], 1 - al)),
                    p_best=float((R[:, j] == 1).mean())) for j, a in enumerate(names)}


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall's tau-b (Kendall 1945), tie-corrected; scipy implementation."""
    return float(stats.kendalltau(x, y, variant="b").statistic)


def tau_between_cohorts(arms1: Dict[str, object], arms2: Dict[str, object], key: str = "worst", B: int = 2000,
                        level: float = 0.95, seed=0, return_draws: bool = False, higher_better: bool = True) -> dict:
    """Kendall tau-b between the arm orderings of two cohorts (common arms) with a bootstrap CI
    over seeds (seeds resampled within (arm, domain) in both cohorts independently).  The
    p-value is the exact permutation p of scipy (null: independent orderings).  ``higher_better``
    only matters for key='worst' (min vs max over domains); tau itself is direction-free."""
    rng = _rng(seed)
    common = [a for a in arms1 if a in arms2]
    g1 = {a: as_groups(arms1[a]) for a in common}
    g2 = {a: as_groups(arms2[a]) for a in common}
    hb = higher_better
    s1 = np.array([_stat(domain_means(g1[a])[None], key, hb)[0] for a in common])
    s2 = np.array([_stat(domain_means(g2[a])[None], key, hb)[0] for a in common])
    kt = stats.kendalltau(s1, s2, variant="b")
    b1 = np.stack([_stat(_resampled_domain_means(g1[a], B, rng, "seed"), key, hb) for a in common], 1)
    b2 = np.stack([_stat(_resampled_domain_means(g2[a], B, rng, "seed"), key, hb) for a in common], 1)
    draws = np.array([kendall_tau_b(b1[i], b2[i]) for i in range(B)])
    al = (1 - level) / 2
    out = dict(n_arms=len(common), arms=common, tau=float(kt.statistic), p=float(kt.pvalue),
               ci=np.nanquantile(draws, [al, 1 - al]).tolist(), boot_median=float(np.nanmedian(draws)))
    if return_draws:
        out["draws"] = draws
    return out


def _interaction_stat(E, codes, G, studentize):
    """Arm x group interaction statistic on row-centred gains E (R x A).

    studentize=False : sum_g n_g ||m_g - m_bar||^2 (homoscedastic between-group SS).
    studentize=True  : Welch/James-type sum_g w_g ||m_g - m_w||^2 with w_g = n_g / s2_g, s2_g =
                       within-group residual variance pooled over arms (df (n_g - 1)(A - 1)) and
                       m_w = sum_g w_g m_g / sum_g w_g (Welch 1951; James 1951).
    """
    A = E.shape[1]
    ms, ws = [], []
    for gi in range(G):
        X = E[codes == gi]
        n = X.shape[0]
        m = X.mean(0)
        ms.append(m)
        if studentize:
            s2 = ((X - m) ** 2).sum() / max((n - 1) * (A - 1), 1)
            ws.append(n / s2 if s2 > 0 else 1e300)
        else:
            ws.append(float(n))
    ms, ws = np.array(ms), np.array(ws)
    mt = (ws[:, None] * ms).sum(0) / ws.sum()
    return float((ws[:, None] * (ms - mt) ** 2).sum())


def interaction_permutation_test(rows: np.ndarray, labels: Sequence, B: int = 10000, seed=0,
                                 method: str = "wild") -> dict:
    """Test of arm x group interaction on per-domain gains, groups = the units the rows are
    nested in (use it for COHORT; for task / shift use ``interaction_2x2_cohort``).

    rows: (R, A) matrix, one row per (cohort, domain) unit, columns = arms, entries = gain of
    the arm over the reference on that domain (seed means).  labels: group of each row.  Each
    row is centred on its own mean across arms (removes domain and group main effects).
    H0: the expected centred gain vector is the same in every group.  Rows are only
    exchangeable WITHIN groups: a row-level shuffle of labels that are constant within higher-
    level clusters (e.g. task or shift, constant within cohort) is invalid, and so is an
    unstudentised permutation when groups differ in noise level.

    method='wild' (default): studentised Welch/James-type statistic (``_interaction_stat``),
        reference distribution from a wild bootstrap under H0: whole residual rows
        (E_i - m_g(i)) * sqrt(n_g/(n_g-1)) are multiplied by Rademacher signs (Wu 1986; Liu 1988;
        Mammen 1993), which keeps each group's own variance and the within-row correlation
        between arms.  Type-I error 0.02-0.065 at alpha .05 in stats_tests (K = 5/5/3/7,
        homo- and heteroscedastic, iid and correlated arms).  Needs >= 2 rows per group.
    method='perm': original unstudentised statistic with label permutation over rows; valid
        only for exchangeable (homoscedastic) rows.
    p = (1 + #{T* >= T})/(1 + B).
    """
    rng = _rng(seed)
    X = np.asarray(rows, float)
    E = X - X.mean(1, keepdims=True)
    lab = np.asarray(labels)
    uniq = list(dict.fromkeys(lab.tolist()))
    codes = np.array([uniq.index(v) for v in lab])
    G = len(uniq)
    n_g = np.array([(codes == g).sum() for g in range(G)])
    if method == "wild":
        if n_g.min() < 2:
            raise ValueError("method='wild' needs at least 2 rows per group")
        t0 = _interaction_stat(E, codes, G, True)
        M = np.array([E[codes == g].mean(0) for g in range(G)])
        Rz = (E - M[codes]) * np.sqrt(n_g[codes] / (n_g[codes] - 1.0))[:, None]
        cnt = 0
        for _ in range(B):
            v = rng.choice([-1.0, 1.0], size=(E.shape[0], 1))
            if _interaction_stat(v * Rz, codes, G, True) >= t0 * (1 - 1e-12):
                cnt += 1
    elif method == "perm":
        t0 = _interaction_stat(E, codes, G, False)
        cnt = 0
        for _ in range(B):
            if _interaction_stat(E, rng.permutation(codes), G, False) >= t0 - 1e-15:
                cnt += 1
    else:
        raise ValueError(method)
    return dict(T=float(t0), p=(1 + cnt) / (1 + B), B=B, method=method, groups=uniq, n_per_group=n_g.tolist(),
                n_rows=int(X.shape[0]), n_arms=int(X.shape[1]))


def interaction_2x2_cohort(rows: np.ndarray, cohorts: Sequence, meta: Optional[Dict[str, dict]] = None) -> dict:
    """Task vs shift structure of the arm x cohort interaction, with COHORT as the unit.

    rows / cohorts as in ``interaction_permutation_test`` (one row per (cohort, domain), cohort
    label per row).  Each cohort contributes one vector m_c = mean over its domains of the
    row-centred gains (equal cohort weights).  With the 2 x 2 design (task x shift, one cohort
    per cell) the between-cohort interaction SS sum_c ||m_c - m_bar||^2 splits exactly into
    three orthogonal 1-df-per-arm contrasts: task, shift and task x shift, SS_f =
    ||sum_c s_fc m_c||^2 / 4 (s = +-1 codes).  Reported:
      * SS and fraction of the between-cohort interaction SS for each contrast (descriptive);
      * split ranks: among the 3 balanced 2-vs-2 splits of 4 cohorts (task, shift, task x shift)
        the rank of the task / shift split; a cohort-level permutation p is rank/3 (minimum
        1/3), so no cohort-level permutation test at alpha .05 is possible;
      * an approximate F test of each main contrast against the task x shift contrast as error
        (cohort-level arm effects as the unit): F = SS_f / SS_txs on (A-1, A-1) df.  Valid if the
        cohort-level deviations are iid across cohorts and spherical across arms; the +-1
        contrasts have equal variance even when cohorts differ in noise level, but they are
        then correlated, so treat p_F as approximate.  It has low power by construction (the
        error has only A-1 df and absorbs any genuine task x shift pattern).
    """
    meta = COHORT_META if meta is None else meta
    X = np.asarray(rows, float)
    E = X - X.mean(1, keepdims=True)
    lab = list(cohorts)
    names = list(dict.fromkeys(lab))
    if len(names) != 4:
        raise ValueError("interaction_2x2_cohort needs exactly 4 cohorts")
    tasks = sorted({meta[c]["task"] for c in names})
    shifts = sorted({meta[c]["shift"] for c in names})
    if len(tasks) != 2 or len(shifts) != 2:
        raise ValueError("cohorts do not form a 2 x 2 task x shift design")
    m = np.array([E[np.array(lab) == c].mean(0) for c in names])
    st = np.array([1.0 if meta[c]["task"] == tasks[0] else -1.0 for c in names])
    ss_ = np.array([1.0 if meta[c]["shift"] == shifts[0] else -1.0 for c in names])
    codes = dict(task=st, shift=ss_, task_x_shift=st * ss_)
    A = X.shape[1]
    tot = float(((m - m.mean(0)) ** 2).sum())
    out = dict(cohorts=names, n_arms=A, between_cohort_ss=tot, unit="cohort",
               note="descriptive: 4 cohorts give only 3 balanced 2-vs-2 splits (min permutation p = 1/3)")
    ss = {}
    for f, c in codes.items():
        L = (c[:, None] * m).sum(0) / 2.0
        ss[f] = float((L ** 2).sum())
        out[f] = dict(ss=ss[f], frac=ss[f] / tot if tot > 0 else float("nan"),
                      contrast=L.tolist(), codes=dict(zip(names, c.tolist())))
    order = sorted(ss, key=lambda f: -ss[f])
    df = A - 1
    for f in ("task", "shift"):
        rank = order.index(f) + 1
        F = ss[f] / ss["task_x_shift"] if ss["task_x_shift"] > 0 else float("inf")
        out[f].update(split_rank=rank, p_split=rank / 3.0, F=F, df=[df, df],
                      p_F=float(stats.f.sf(F, df, df)) if np.isfinite(F) else 0.0)
    return out


COHORT_META = {
    "c17": dict(task="tumour", shift="institutional"),
    "canine": dict(task="tumour", shift="acquisition"),
    "midog21": dict(task="mitosis", shift="acquisition"),
    "midog21sn": dict(task="mitosis", shift="acquisition"),
    "midogpp": dict(task="mitosis", shift="institutional"),
}


def cluster_summary(tau_pairs: Dict[tuple, dict], meta: Dict[str, dict] = COHORT_META, level: float = 0.95) -> dict:
    """Does ranking agreement cluster by task or by shift type (2 x 2 design)?

    tau_pairs: {(c1, c2): output of tau_between_cohorts(..., return_draws=True)}.  Pairs are
    classified as same-task, same-shift or neither; reports mean tau per class and bootstrap
    CIs for (same-task - same-shift) and each minus 'neither' (bootstrap draws are combined
    draw-wise; cohort-pair bootstraps share no resampling, so this is descriptive)."""
    cls, names = {}, {}
    for (a, b), r in tau_pairs.items():
        if meta[a]["task"] == meta[b]["task"]:
            c = "same_task"
        elif meta[a]["shift"] == meta[b]["shift"]:
            c = "same_shift"
        else:
            c = "neither"
        cls.setdefault(c, []).append(r)
        names.setdefault(c, []).append(f"{a}-{b}")
    al = (1 - level) / 2
    out = {c: dict(pairs=names[c], mean_tau=float(np.mean([v["tau"] for v in cls[c]]))) for c in cls}
    draws = {c: np.nanmean(np.stack([v["draws"] for v in L]), 0) for c, L in cls.items() if all("draws" in v for v in L)}
    for a, b in (("same_task", "same_shift"), ("same_task", "neither"), ("same_shift", "neither")):
        if a in draws and b in draws:
            dd = draws[a] - draws[b]
            out[f"{a}_minus_{b}"] = dict(point=out[a]["mean_tau"] - out[b]["mean_tau"],
                                         ci=np.nanquantile(dd, [al, 1 - al]).tolist())
    return out


# ----------------------------------------------------------------------------------------
# f) response shape
# ----------------------------------------------------------------------------------------


def oldham_pitman_morgan(M: Sequence[float], E: Sequence[float], n_perm: int = 100000, seed=0) -> dict:
    """Oldham's correlation and the Pitman-Morgan test of equal variances.

    M, E: per-domain (seed-mean) scores of method and reference.  Oldham (1962):
    r = corr(M - E, (M + E)/2); r < 0 means gains are larger where the reference is weaker
    beyond what regression to the mean produces.  Pitman (1939)/Morgan (1939): var(M) = var(E)
    iff corr(M - E, M + E) = 0 (same r), t = r sqrt(K-2)/sqrt(1-r^2) on K-2 df.
    Permutation p: within-domain swap of (M_t, E_t) (sign flips of M-E), exhaustive when
    2^K <= n_perm.  Also returns the OLS slope of M on E (the r - r0 'compression' view).
    """
    M, E = np.asarray(M, float), np.asarray(E, float)
    K = M.size
    D, A = M - E, (M + E) / 2

    def r_of(D_, A_):
        if D_.std() == 0 or A_.std() == 0:
            return 0.0
        return float(np.corrcoef(D_, A_)[0, 1])

    r = r_of(D, A)
    t = r * math.sqrt(K - 2) / math.sqrt(max(1 - r * r, 1e-300)) if K > 2 else float("nan")
    p = float(2 * stats.t.sf(abs(t), K - 2)) if K > 2 else float("nan")
    # permutation: swapping M_t and E_t flips D_t, keeps A_t
    if 2 ** K <= n_perm:
        signs = np.array(list(itertools.product([1, -1], repeat=K)))
    else:
        signs = _rng(seed).choice([1, -1], size=(n_perm, K))
    Ds = signs * D
    Dc = Ds - Ds.mean(1, keepdims=True)
    Ac = A - A.mean()
    den = np.sqrt((Dc ** 2).sum(1) * (Ac ** 2).sum())
    rs = np.where(den > 0, (Dc * Ac).sum(1) / np.where(den > 0, den, 1), 0.0)
    pperm = float((np.abs(rs) >= abs(r) - 1e-12).mean())
    slope = float(np.polyfit(E, M, 1)[0]) if E.std() > 0 else float("nan")
    return dict(K=K, oldham_r=r, pm_t=t, pm_df=K - 2, pm_p=p, perm_p=pperm,
                n_perm=int(signs.shape[0]), exhaustive=bool(2 ** K <= n_perm),
                var_ratio=float(M.var(ddof=1) / E.var(ddof=1)) if E.var() > 0 else float("nan"),
                slope_M_on_E=slope)
