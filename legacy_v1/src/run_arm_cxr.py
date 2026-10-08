"""Train one arm of the cross-institution chest-radiograph experiment.

Arms: baseline | sam | ssl_S | ssl_M | ssl_S_sam | ssl_M_sam
  - ssl_*  loads the SimCLR encoder pre-trained on pool S or pool M
  - *_sam  optimises with SAM (rho carried over unchanged)

Model selection is in-distribution only: the checkpoint with the best NIH
id_val AUROC is the one scored on the CheXpert OOD set. No CheXpert label is
read during training or selection.

Saves per-sample predictions for both test sets so the paired bootstrap in
aggregate_cxr.py resamples the same patients across every arm.
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, _os.environ.get("ARM_THREADS", "3"))

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
from train_lib import SAM, Classifier, Encoder, iterate, sam_step

CACHE = "cache_cxr"
OUT = "runs_cxr"


@torch.no_grad()
def predict_gray(model, X, bs=512):
    """predict() for single-channel uint8 radiographs."""
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs])
        p = torch.softmax(model(T.normalize(T.to_float_gray(xb))), 1)[:, 1]
        out.append(p.numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["baseline", "sam", "ssl_S", "ssl_M",
                             "ssl_S_sam", "ssl_M_sam"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--rho", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    os.makedirs(OUT, exist_ok=True)
    tag = f"{a.arm}_seed{a.seed}"

    d = np.load(f"{CACHE}/cxr_splits.npz")
    Xtr, ytr = d["Xtr"], d["ytr"]
    Xiv, yiv = d["Xiv"], d["yiv"]
    Xit, yit = d["Xit"], d["yit"]
    Xood, yood = d["Xood"], d["yood"]

    use_sam = a.arm.endswith("_sam") or a.arm == "sam"
    pool = "S" if "_S" in a.arm else ("M" if "_M" in a.arm else None)

    enc = Encoder()
    if pool:
        ck = f"{CACHE}/ssl_encoder_cxr_{pool}_seed{a.seed}.pt"
        enc.load_state_dict(torch.load(ck))
        print(f"[{tag}] loaded {ck}", flush=True)
    model = Classifier(enc)

    # train set keeps all positives + subsampled negatives, so it is imbalanced;
    # class weights make the loss prevalence-independent. Identical for all arms.
    cw = torch.tensor([1.0, float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)],
                      dtype=torch.float32)
    lf = nn.CrossEntropyLoss(weight=cw)
    print(f"[{tag}] train n={len(ytr)} prev={ytr.mean():.3f} pos_weight={cw[1]:.2f}",
          flush=True)
    if use_sam:
        opt = SAM(model.parameters(), torch.optim.SGD, rho=a.rho,
                  lr=a.lr, momentum=0.9, weight_decay=5e-4)
        base_opt = opt.base_optimizer
    else:
        opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9,
                              weight_decay=5e-4)
        base_opt = opt
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(base_opt, T_max=a.epochs)

    best = dict(auroc=-1.0)
    for ep in range(a.epochs):
        t0 = time.time()
        model.train()
        tot = nb = 0.0
        for xb, yb in iterate(Xtr, ytr, a.bs, shuffle=True, rng=rng):
            xb = T.normalize(T.sup_augment_cxr(T.to_float_gray(xb)))
            if use_sam:
                loss = sam_step(model, xb, yb, opt, lf)   # already a float
            else:
                opt.zero_grad(); lt = lf(model(xb), yb)
                lt.backward(); opt.step()
                loss = float(lt.detach())                 # detach: no graph retained
            tot += loss; nb += 1
        sched.step()
        model.eval()
        piv = predict_gray(model, Xiv)
        m_iv = T.compute_metrics(yiv, piv)
        print(f"[{tag}] ep{ep} loss={tot / max(nb, 1):.4f} "
              f"id_val_auroc={m_iv['auroc']:.4f} {time.time() - t0:.0f}s", flush=True)
        if m_iv["auroc"] > best["auroc"]:
            best = dict(auroc=m_iv["auroc"], epoch=ep,
                        state={k: v.clone() for k, v in model.state_dict().items()})

    # score the selected checkpoint once on each test set
    model.load_state_dict(best["state"])
    model.eval()
    p_it, p_ood = predict_gray(model, Xit), predict_gray(model, Xood)
    res = dict(arm=a.arm, seed=a.seed, selected_epoch=best["epoch"],
               id_val_auroc=best["auroc"],
               id_test=T.compute_metrics(yit, p_it),
               ood=T.compute_metrics(yood, p_ood),
               epochs=a.epochs, rho=a.rho if use_sam else None, pool=pool)
    torch.save(model.state_dict(), f"{OUT}/{tag}.pt")
    np.savez_compressed(f"{OUT}/{tag}_preds.npz",
                        p_id_test=p_it, y_id_test=yit,
                        p_ood=p_ood, y_ood=yood)
    json.dump(res, open(f"{OUT}/{tag}.json", "w"), indent=1)
    print(f"[{tag}] DONE id_test={res['id_test']['auroc']:.4f} "
          f"ood={res['ood']['auroc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
