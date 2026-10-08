"""Evaluate every fold's checkpoints on PatchCamelyon test split.

External validation for the surviving SSL / SSL+SAM claim: PatchCamelyon [30]
is a re-partition of the Camelyon cohort with a train/test boundary disjoint
from the WILDS one used to select our checkpoints, so a positive result here
is transfer of the SSL floor-lift pattern to data not used to tune any
hyperparameter or select any epoch.

Given the shipped .pt files at runs/foldT/{arm}_seed{s}.pt, decodes the
PatchCamelyon test HDF5, resizes 96x96 -> 64x64, and reports off-cohort
worst-site AUROC and across-site SD.
"""
from __future__ import annotations
import argparse, gzip, io, json, os, sys, time
import numpy as np, torch, torch.nn as nn
import h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_lib as T


def load_pcam(root):
    def gunzip(path):
        if path.endswith(".gz"):
            plain = path[:-3]
            if not os.path.exists(plain):
                with gzip.open(path, "rb") as gz, open(plain, "wb") as out:
                    while True:
                        chunk = gz.read(64 * 1024 * 1024)
                        if not chunk: break
                        out.write(chunk)
            return plain
        return path

    xp = gunzip(os.path.join(root, "camelyonpatch_level_2_split_test_x.h5.gz"))
    yp = gunzip(os.path.join(root, "camelyonpatch_level_2_split_test_y.h5.gz"))
    with h5py.File(xp, "r") as f:
        X = f["x"][:]      # (N, 96, 96, 3) uint8
    with h5py.File(yp, "r") as f:
        y = f["y"][:].squeeze().astype(np.int64)
    # center-crop 96 -> 64
    l = (96 - 64) // 2
    X = X[:, l:l + 64, l:l + 64, :]
    print(f"pcam test set: X={X.shape} y={y.shape}, pos_frac={y.mean():.3f}")
    return X, y


@torch.no_grad()
def score(state, X, dev):
    model = T.Classifier(T.Encoder()).to(dev)
    model.load_state_dict(state)
    model.eval()
    out = []
    for k in range(0, len(X), 512):
        xb = torch.from_numpy(X[k:k + 512])
        xn = (T.to_float(xb).to(dev) - T.IMAGENET_MEAN.to(dev)) / T.IMAGENET_STD.to(dev)
        p = torch.softmax(model(xn), 1)[:, 1].cpu().numpy()
        out.append(p)
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcam_root", default="./data/pcam")
    ap.add_argument("--runs_root", default="./data/runs")
    ap.add_argument("--out", default="results/camelyon17_extended/pcam_external.csv")
    ap.add_argument("--arms", default="baseline sam ssl ssl_sam")
    ap.add_argument("--seeds", default="0")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, y = load_pcam(args.pcam_root)

    rows = []
    for fold in range(5):
        for arm in args.arms.split():
            for seed in [int(s) for s in args.seeds.split()]:
                ck = os.path.join(args.runs_root, f"fold{fold}", f"{arm}_seed{seed}.pt")
                if not os.path.exists(ck):
                    continue
                t0 = time.time()
                state = torch.load(ck, map_location=dev, weights_only=True)
                p = score(state, X, dev)
                m = T.compute_metrics(y, p)
                rows.append(dict(fold=fold, arm=arm, seed=seed,
                                 pcam_auroc=m["auroc"], pcam_ece=m["ece"],
                                 pcam_accuracy=m["accuracy"], pcam_f1=m["f1"],
                                 wall_s=time.time() - t0))
                print(f"fold{fold} {arm} seed{seed}: pcam auroc={m['auroc']:.4f} "
                      f"ece={m['ece']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print("\n== per-arm across-fold summary ==")
    for arm in args.arms.split():
        g = df[df.arm == arm]
        if len(g) == 0: continue
        v = g["pcam_auroc"]
        print(f"{arm:<10}  mean={v.mean():.4f}  SD_across_folds={v.std(ddof=1):.4f}  "
              f"worst={v.min():.4f}  best={v.max():.4f}")


if __name__ == "__main__":
    main()
