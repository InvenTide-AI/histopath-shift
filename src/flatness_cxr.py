"""Loss-landscape sharpness for every trained chest-radiograph model.

Identical measures to flatness.py (the Camelyon17 version) and flatness_crc.py;
only the data cache, run directory and the grayscale loader differ, so the two datasets' sharpness numbers are
computed the same way and are directly comparable.

Implements the two measures the position paper's Proposal (ii) asks the field
to report alongside AUC:
  (a) adversarial (SAM-style) sharpness  max_{||eps||<=rho} L(w+eps) - L(w)
  (b) random filter-normalized perturbation loss curves
Both are computed on the TRAINING cohort data (sharpness is a property of
the found minimum, not of the test set).
"""
import glob, json, os, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_lib as T

torch.set_num_threads(int(os.environ.get("FLAT_THREADS", "8")))
d = np.load("cache_cxr/cxr_splits.npz")
Xtr, ytr = d["Xtr"], d["ytr"]

rs = np.random.RandomState(0)
sub = rs.choice(len(Xtr), int(os.environ.get('FLAT_N','2048')), replace=False)
Xs, ys = Xtr[sub], ytr[sub]
loss_fn = nn.CrossEntropyLoss()
RHOS = [0.02, 0.05, 0.1, 0.2]
N_DIR = 3


def batches():
    for xb, yb in T.iterate(Xs, ys, 512, shuffle=False):
        yield T.normalize(T.to_float_gray(xb)), yb


@torch.no_grad()
def total_loss(model):
    model.eval()
    tot, n = 0.0, 0
    for x, y in batches():
        tot += loss_fn(model(x), y).item() * len(y); n += len(y)
    return tot / n


def grad_vec(model):
    model.eval(); model.zero_grad()
    n = 0
    for x, y in batches():
        (loss_fn(model(x), y) * len(y)).backward(); n += len(y)
    g = [(p.grad / n).clone() if p.grad is not None else torch.zeros_like(p)
         for p in model.parameters()]
    model.zero_grad()
    return g


def perturb(model, vecs, scale):
    with torch.no_grad():
        for p, v in zip(model.parameters(), vecs):
            p.add_(v * scale)


rows, curves = [], {}
SRC = "runs_cxr"
for ck in sorted(glob.glob(f"{SRC}/*.pt")):
    tag = os.path.basename(ck)[:-3]
    arm, seed = tag.rsplit("_seed", 1)
    model = T.Classifier()
    model.load_state_dict(torch.load(ck, map_location="cpu"))
    L0 = total_loss(model)

    # (a) adversarial sharpness: single ascent step along the gradient direction
    g = grad_vec(model)
    gn = torch.sqrt(sum((gi ** 2).sum() for gi in g)) + 1e-12
    adv = {}
    for rho in RHOS:
        perturb(model, g, rho / gn)
        adv[rho] = total_loss(model) - L0
        perturb(model, g, -rho / gn)

    # (b) random filter-normalized directions
    rand = {rho: [] for rho in RHOS}
    for k in range(N_DIR):
        torch.manual_seed(1234 + k)
        dirs = []
        for p in model.parameters():
            v = torch.randn_like(p)
            v = v * (p.norm() / (v.norm() + 1e-12))   # filter-normalized
            dirs.append(v)
        dn = torch.sqrt(sum((v ** 2).sum() for v in dirs)) + 1e-12
        for rho in RHOS:
            perturb(model, dirs, rho / dn)
            rand[rho].append(total_loss(model) - L0)
            perturb(model, dirs, -rho / dn)

    row = dict(arm=arm, seed=int(seed), base_loss=L0,
               **{f"adv_sharp_rho{r}": adv[r] for r in RHOS},
               **{f"rand_sharp_rho{r}": float(np.mean(rand[r])) for r in RHOS},
               **{f"rand_sharp_std_rho{r}": float(np.std(rand[r])) for r in RHOS})
    rows.append(row)
    curves[tag] = dict(rhos=RHOS, adv=[adv[r] for r in RHOS],
                       rand_mean=[float(np.mean(rand[r])) for r in RHOS],
                       rand_std=[float(np.std(rand[r])) for r in RHOS],
                       base_loss=L0)
    print(f"{tag}: L0={L0:.4f} adv@0.05={adv[0.05]:.4f} adv@0.2={adv[0.2]:.4f} "
          f"rand@0.2={np.mean(rand[0.2]):.4f}", flush=True)

import pandas as pd
df = pd.DataFrame(rows)
df["sharpness_rho0.05"] = df["adv_sharp_rho0.05"]
df["sharpness_rho0.2"] = df["adv_sharp_rho0.2"]
df.to_csv("cxr_flatness.csv", index=False)
json.dump(curves, open("cxr_flatness_curves.json", "w"), indent=1)
print("wrote cxr_flatness.csv")
