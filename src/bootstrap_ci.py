"""Bootstrap CIs on the held-out hospital, and paired bootstrap tests between arms.

Seed-level replication is limited by the CPU budget, so uncertainty is quantified
by resampling the 25,000-patch held-out test set (2,000 bootstrap replicates).
Contrasts are PAIRED: the same resampled patch indices are scored for both arms,
which is the correct test for "does mechanism X improve OOD performance".
"""
import glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

SRC = "runs_final" if glob.glob("runs_final/*.json") else "runs"
ARMS = ["baseline", "ssl", "sam", "ssl_sam"]
B = 2000
rng = np.random.RandomState(12345)

preds = {}
for f in sorted(glob.glob(f"{SRC}/*_preds.npz")):
    tag = os.path.basename(f).replace("_preds.npz", "")
    arm, seed = tag.rsplit("_seed", 1)
    z = np.load(f)
    preds.setdefault(arm, {})[int(seed)] = (z["y_ood_test"], z["p_ood_test"])

# one representative run per arm (lowest seed available), plus seed-averaged scores
rep = {a: preds[a][min(preds[a])] for a in ARMS if a in preds}
y0 = list(rep.values())[0][0]
n = len(y0)
assert all(np.array_equal(v[0], y0) for v in rep.values()), "test labels differ across arms"

idx = rng.randint(0, n, size=(B, n))


def metrics_at(y, p):
    return dict(auroc=roc_auc_score(y, p),
                avg_precision=average_precision_score(y, p),
                accuracy=((p > .5).astype(int) == y).mean(),
                f1=f1_score(y, (p > .5).astype(int)))


boot = {a: {m: np.empty(B) for m in ["auroc", "avg_precision", "accuracy", "f1"]}
        for a in rep}
for b in range(B):
    ii = idx[b]
    yb = y0[ii]
    if yb.sum() == 0 or yb.sum() == len(yb):
        continue
    for a, (y, p) in rep.items():
        mm = metrics_at(yb, p[ii])
        for k, v in mm.items():
            boot[a][k][b] = v

rows = []
for a in rep:
    for m in boot[a]:
        v = boot[a][m]
        rows.append(dict(arm=a, metric=m, point=metrics_at(*rep[a])[m],
                         ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5),
                         boot_mean=v.mean(), boot_sd=v.std(ddof=1)))
ci = pd.DataFrame(rows)
ci.to_csv("bootstrap_ci.csv", index=False)

# paired bootstrap contrasts
cons = []
for a, b_ in [("baseline", "ssl"), ("baseline", "sam"), ("baseline", "ssl_sam"),
              ("ssl", "ssl_sam"), ("sam", "ssl_sam")]:
    if a not in rep or b_ not in rep:
        continue
    for m in ["auroc", "avg_precision", "accuracy", "f1"]:
        d = boot[b_][m] - boot[a][m]
        p_two = 2 * min((d <= 0).mean(), (d >= 0).mean())
        cons.append(dict(contrast=f"{a}->{b_}", metric=m,
                         delta=metrics_at(*rep[b_])[m] - metrics_at(*rep[a])[m],
                         ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5),
                         p_boot=max(p_two, 1.0 / B),
                         prob_improvement=(d > 0).mean()))
cons = pd.DataFrame(cons)
cons.to_csv("bootstrap_contrasts.csv", index=False)
np.savez_compressed("bootstrap_draws.npz",
                    **{f"{a}_{m}": boot[a][m] for a in boot for m in boot[a]})
print(ci.to_string(index=False))
print()
print(cons.to_string(index=False))
