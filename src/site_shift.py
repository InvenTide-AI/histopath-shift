"""Quantify how far apart the hospitals are, independently of any model.

Why this is a prerequisite and not a side analysis
--------------------------------------------------
The central result of the earlier work is that combining a representation
mechanism with a flatness mechanism is super-additive under shift and absent
in-distribution, but that a net gain requires the interaction to exceed the
summed cost of the mechanisms alone (I > c). Whether that condition holds varied
by axis, and with one held-out hospital there was no way to tell whether the
variation tracked the SIZE of the shift or something idiosyncratic to that site.

Measuring site distance before training makes the fold difficulty an independent
variable rather than a post-hoc story. If benefit tracks shift magnitude, that is
a prediction the design can state in advance and check afterwards.

Two complementary measures
--------------------------
1. Stain statistics in optical-density space. H&E absorbance is close to linear
   in OD, so a scanner/protocol difference appears mostly as a per-channel
   affine change. Mean/covariance in OD space and the Macenko stain-vector
   angles therefore capture the physically meaningful part of the difference,
   unlike raw RGB distances which are dominated by tissue content.
2. Domain separability. A logistic regression on colour histograms trying to
   tell held-out-site patches from training-site patches; its AUROC is a direct,
   model-free measure of how distinguishable the sites are. 0.5 means
   indistinguishable, 1.0 means trivially separable.

The two disagree in an informative way: stain distance sees only colour, while
the classifier can also exploit texture and tissue-composition differences.

Output: results/site_shift.csv, results/site_shift_pairwise.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def od(x, eps=1e-6):
    """RGB in [0,1] -> optical density."""
    return -np.log10(np.clip(x, eps, 1.0))


def stain_stats(imgs, max_n=4000, rng=None):
    """Mean/cov of OD over tissue pixels, plus Macenko stain vectors.

    Background (bright, low-OD) pixels carry no stain information and would
    dominate the statistics, so they are excluded by an OD threshold -- the same
    convention Macenko et al. use.
    """
    rng = rng or np.random.default_rng(0)
    if len(imgs) > max_n:
        imgs = imgs[rng.choice(len(imgs), max_n, replace=False)]
    x = np.asarray(imgs, dtype=np.float32) / 255.0
    p = od(x.reshape(-1, 3))
    tissue = p.sum(1) > 0.15          # drop background
    p = p[tissue] if tissue.sum() > 1000 else p
    mu = p.mean(0)
    cov = np.cov(p.T)
    # Macenko: the two leading eigenvectors of OD span the stain plane; the
    # extreme angles within that plane are the H and E vectors.
    w, v = np.linalg.eigh(cov)
    V = v[:, [2, 1]]                  # top-2 eigenvectors
    proj = p @ V
    ang = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(ang, 1), np.percentile(ang, 99)
    s1 = V @ np.array([np.cos(lo), np.sin(lo)])
    s2 = V @ np.array([np.cos(hi), np.sin(hi)])
    s1, s2 = np.abs(s1) / (np.linalg.norm(s1) + 1e-9), np.abs(s2) / (np.linalg.norm(s2) + 1e-9)
    return dict(od_mean=mu, od_cov=cov, stain1=s1, stain2=s2,
                od_mean_norm=float(np.linalg.norm(mu)), n_pixels=int(len(p)))


def frechet(mu1, cov1, mu2, cov2):
    """2-Wasserstein distance between Gaussians -- a scale-aware distance that,
    unlike a mean difference, also registers a change in stain VARIABILITY."""
    from scipy.linalg import sqrtm
    d = mu1 - mu2
    c = sqrtm(cov1 @ cov2)
    if np.iscomplexobj(c):
        c = c.real
    return float(np.sqrt(max(d @ d + np.trace(cov1 + cov2 - 2 * c), 0.0)))


def colour_hist(imgs, bins=12):
    """Per-channel histogram features; cheap and model-free."""
    x = np.asarray(imgs, dtype=np.float32) / 255.0
    feats = []
    for i in range(len(x)):
        h = [np.histogram(x[i, :, :, c], bins=bins, range=(0, 1), density=True)[0]
             for c in range(3)]
        feats.append(np.concatenate(h))
    return np.asarray(feats, dtype=np.float32)


def domain_auroc(f_a, f_b, seed=0):
    """How separable are two sites? Logistic regression, held-out AUROC.

    Patch-level features from the same patient are correlated, but here we only
    need a summary of site distinguishability, and both classes are sampled
    across many patients, so a plain split suffices.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X = np.concatenate([f_a, f_b])
    y = np.concatenate([np.zeros(len(f_a)), np.ones(len(f_b))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed,
                                          stratify=y)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    return float(roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="cache")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--out", default="results")
    p.add_argument("--per-site", type=int, default=3000,
                   help="cap on patches per site (the true n is the smallest "
                        "site's label-matched pool)")
    p.add_argument("--label-match", type=int, default=0,
                   help="restrict to this tissue class (0=normal) so site is "
                        "not confounded with tumour content")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    meta = pd.read_parquet(os.path.join(a.cache, "camelyon17_metadata.parquet"))
    meta["center"] = meta["center"].astype(str)
    z = np.load(os.path.join(a.cache, f"site_sample_{a.size}.npz"))
    imgs_all = z["images"]
    smeta = pd.DataFrame(dict(center=z["center"].astype(str), label=z["label"],
                              patient=z["patient"]))

    # Tumour and normal tissue differ in colour, and the per-site sample is not
    # class-balanced (the shards each site lives in are not), so comparing sites
    # on mixed tissue would confound SITE with TUMOUR CONTENT. Restricting to
    # normal tissue and equalising n makes the remaining difference attributable
    # to scanner/protocol rather than to what tissue happened to be sampled.
    rng = np.random.default_rng(a.seed)
    sites = sorted(smeta.center.unique())
    keep = smeta.label == a.label_match
    n_per = int(min(keep[smeta.center == s].sum() for s in sites))
    n_per = min(n_per, a.per_site)
    print(f"label-matched comparison on label={a.label_match} tissue, "
          f"n={n_per} per site (limited by the smallest site)")

    per_site_imgs, stats = {}, {}
    for s in sites:
        pool = smeta.index[(smeta.center == s) & keep].to_numpy()
        take = rng.choice(pool, n_per, replace=False)
        per_site_imgs[s] = imgs_all[np.sort(take)]
        stats[s] = stain_stats(per_site_imgs[s], rng=rng)
        print(f"  site {s}: n={n_per} from {smeta.loc[take,'patient'].nunique()} "
              f"patients, |OD mean|={stats[s]['od_mean_norm']:.4f}", flush=True)

    feats = {s: colour_hist(per_site_imgs[s]) for s in sites}

    # pairwise stain distance
    pw = []
    for i, s1 in enumerate(sites):
        for s2 in sites[i + 1:]:
            d = frechet(stats[s1]["od_mean"], stats[s1]["od_cov"],
                        stats[s2]["od_mean"], stats[s2]["od_cov"])
            auc = domain_auroc(feats[s1], feats[s2], seed=a.seed)
            pw.append(dict(site_a=s1, site_b=s2, stain_frechet=round(d, 5),
                           pair_domain_auroc=round(auc, 5)))
    pw = pd.DataFrame(pw)

    # per-fold: held-out site vs the pooled remainder (what the model faces)
    rows = []
    for held in sites:
        rest = [s for s in sites if s != held]
        f_rest = np.concatenate([feats[s] for s in rest])
        rest_imgs = np.concatenate([per_site_imgs[s] for s in rest])
        st_rest = stain_stats(rest_imgs, rng=rng)
        d = frechet(stats[held]["od_mean"], stats[held]["od_cov"],
                    st_rest["od_mean"], st_rest["od_cov"])
        auc = domain_auroc(f_rest, feats[held], seed=a.seed)
        rows.append(dict(holdout=held,
                         stain_frechet_vs_rest=round(d, 5),
                         domain_auroc_vs_rest=round(auc, 5),
                         od_mean_norm=round(stats[held]["od_mean_norm"], 5),
                         n_compared=int(n_per),
                         tumour_prevalence_full=round(
                             float(meta.loc[meta.center == held, "label"].mean()), 5),
                         n_patients_full=int(
                             meta.loc[meta.center == held, "patient"].nunique()),
                         n_patches_full=int((meta.center == held).sum())))
    df = pd.DataFrame(rows).sort_values("domain_auroc_vs_rest", ascending=False)

    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "site_shift.csv"), index=False)
    pw.to_csv(os.path.join(a.out, "site_shift_pairwise.csv"), index=False)
    print("\nper-fold shift (sorted hardest-first by domain separability):")
    print(df.to_string(index=False))
    print("\npairwise:")
    print(pw.to_string(index=False))


if __name__ == "__main__":
    main()
