"""Test-time adaptation arms, applied post hoc to an already-trained checkpoint.

Every other arm in this study intervenes during training: in the input space,
in the representation, or in the objective. Test-time adaptation intervenes at
a fourth place, after deployment, using the unlabelled test batch itself. The
taxonomy in the discussion needs that level filled in, and because it reuses
saved checkpoints it costs almost nothing to add.

Two variants, in increasing order of effort:

  bnadapt   Recompute batch-norm running statistics on the test split and
            change nothing else. No gradients, no labels. This is the cheapest
            possible correction for a covariate shift that is mostly a change
            of first and second moments, which is exactly what a scanner
            change looks like.

  tent      TENT (Wang et al., ICLR 2021): put batch-norm in train mode so it
            uses test-batch statistics, freeze everything except the batch-norm
            affine parameters, and take one gradient step per batch on the mean
            prediction entropy. Still no labels.

Both are evaluated on the same held-out domain as the source checkpoint, so the
numbers drop straight into the same tables.

Usage:
    python src/run_arm_tta.py --data <fold> --out <runs> --arm tent --source baseline --seed 0
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_lib as T


def batch(X, k, bs, dev):
    """uint8 (B,H,W,3) numpy slice -> normalized (B,3,H,W) tensor on `dev`."""
    xb = torch.from_numpy(X[k:k + bs])
    x = T.to_float(xb).to(dev)
    return (x - T.IMAGENET_MEAN.to(dev)) / T.IMAGENET_STD.to(dev)


@torch.no_grad()
def predict(model, X, dev, bs=512):
    model.eval()
    out = []
    for k in range(0, len(X), bs):
        out.append(torch.softmax(model(batch(X, k, bs, dev)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def bn_modules(model):
    return [m for m in model.modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def configure_tent(model):
    """Freeze everything but batch-norm affine parameters; BN uses batch stats."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for m in bn_modules(model):
        m.train()                 # use the current batch's statistics
        m.track_running_stats = False
        m.running_mean = None
        m.running_var = None
        for p in (m.weight, m.bias):
            if p is not None:
                p.requires_grad_(True)
                params.append(p)
    return params


@torch.no_grad()
def refresh_bn_stats(model, X, dev, bs, passes=1):
    """Re-estimate BN running statistics on unlabelled target data."""
    for m in bn_modules(model):
        m.reset_running_stats()
        m.momentum = None          # cumulative average over the pass
    model.train()
    for _ in range(passes):
        for k in range(0, len(X), bs):
            model(batch(X, k, bs, dev))
    model.eval()


def entropy(logits):
    p = logits.softmax(1)
    return -(p * p.clamp_min(1e-8).log()).sum(1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, choices=["bnadapt", "tent"])
    ap.add_argument("--source", default="baseline",
                    help="arm whose checkpoint is adapted")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=1, help="TENT steps per batch")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--arm_name", default=None,
                    help="override the recorded arm name, e.g. bnadapt-on-ssl")
    a = ap.parse_args()

    name = a.arm_name or a.arm
    tag = "%s_seed%d" % (name, a.seed)
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    d = np.load(os.path.join(a.data, "splits.npz"))
    ck = os.path.join(a.out, "%s_seed%d.pt" % (a.source, a.seed))
    if not os.path.exists(ck):
        raise SystemExit("no source checkpoint at %s" % ck)

    model = T.Classifier(T.Encoder()).to(dev)
    model.load_state_dict(torch.load(ck, map_location=dev, weights_only=True))
    t0 = time.time()

    Xot = d["Xot"]
    if a.arm == "bnadapt":
        refresh_bn_stats(model, Xot, dev, a.bs)
    else:
        params = configure_tent(model)
        opt = torch.optim.Adam(params, lr=a.lr)
        for k in range(0, len(Xot), a.bs):
            xb = batch(Xot, k, a.bs, dev)
            for _ in range(a.steps):
                opt.zero_grad()
                loss = entropy(model(xb))
                loss.backward()
                opt.step()

    p = predict(model, Xot, dev)
    res = {"arm": name, "method": a.arm, "seed": a.seed, "source": a.source,
           "lr": a.lr if a.arm == "tent" else None,
           "steps": a.steps if a.arm == "tent" else None,
           "wall_s": time.time() - t0, "device": str(dev),
           "ood_test": T.compute_metrics(d["yot"], p)}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, tag + ".json"), "w"), indent=1)
    print("[%s] DONE from %s: ood_test auroc=%.4f ece=%.4f"
          % (tag, a.source, res["ood_test"]["auroc"], res["ood_test"]["ece"]),
          flush=True)


if __name__ == "__main__":
    main()
