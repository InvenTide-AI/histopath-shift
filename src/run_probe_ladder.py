"""Linear probes of frozen backbones spanning a pretraining-scale ladder.

The single-backbone probe in run_arm_probe.py showed that an off-the-shelf
representation can flatten the domain effect on some cohorts and not on others.
That leaves the interesting question open: is the useful variable the size of
the model, or the size of its pretraining corpus? This script separates them.

    backbone          params   pretraining corpus
    resnet18          11.7 M   ImageNet-1K   (1.28 M images)
    resnet50          25.6 M   ImageNet-1K
    convnext_tiny     28.6 M   ImageNet-1K
    vit_b_16          86.6 M   ImageNet-1K
    vit_b_16_swag     86.6 M   IG-3.6 B -> ImageNet-1K
    vit_l_16_swag      304 M   IG-3.6 B -> ImageNet-1K

vit_b_16 against vit_b_16_swag is the controlled pair: identical architecture,
roughly 2800x the pretraining data. Everything downstream is held fixed --
frozen features, a linear head, the same epochs and the same held-out-domain
selection rule as every other arm in the grid.

Usage:
    python src/run_probe_ladder.py --data <fold_dir> --out <run_dir> --arch vit_b_16_swag
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
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_lib as T

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# arch -> (torchvision ctor name, feature dim, SWAG input resolution or None)
LADDER = {
    "resnet18":      ("resnet18", 512, None),
    "resnet50":      ("resnet50", 2048, None),
    "convnext_tiny": ("convnext_tiny", 768, None),
    "vit_b_16":      ("vit_b_16", 768, None),
    "vit_b_16_swag": ("vit_b_16", 768, 384),
    "vit_l_16_swag": ("vit_l_16", 1024, 512),
}
PRETRAIN_IMAGES = {
    "resnet18": 1.28e6, "resnet50": 1.28e6, "convnext_tiny": 1.28e6,
    "vit_b_16": 1.28e6, "vit_b_16_swag": 3.6e9, "vit_l_16_swag": 3.6e9,
}


def build(arch: str, weights_dir: str, dev):
    import torchvision.models as M
    ctor_name, dim, swag_res = LADDER[arch]
    m = getattr(M, ctor_name)(weights=None)
    if swag_res is not None:
        # SWAG checkpoints were trained at a larger input resolution, so the
        # ViT must be constructed at that resolution for the positional
        # embedding to match.
        m = getattr(M, ctor_name)(weights=None, image_size=swag_res)
    state = torch.load(os.path.join(weights_dir, arch + ".pth"),
                       map_location="cpu", weights_only=True)
    m.load_state_dict(state)
    # strip the classifier so the penultimate embedding is exposed
    if hasattr(m, "fc"):
        m.fc = nn.Identity()
    elif hasattr(m, "heads"):
        m.heads = nn.Identity()
    elif hasattr(m, "classifier"):
        m.classifier = nn.Sequential(*list(m.classifier)[:-1], nn.Flatten(1))
    for p in m.parameters():
        p.requires_grad_(False)
    res = swag_res or 224
    return m.eval().to(dev), dim, res


@torch.no_grad()
def embed(backbone, X, dev, res, bs=32):
    out = []
    mean, std = IMAGENET_MEAN.to(dev), IMAGENET_STD.to(dev)
    for k in range(0, len(X), bs):
        xb = torch.from_numpy(X[k:k + bs])
        x = xb.permute(0, 3, 1, 2).float().div_(255.0).to(dev)
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        f = backbone(x)
        out.append(f.flatten(1).cpu())
    return torch.cat(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", required=True, choices=sorted(LADDER))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--weights_dir",
                    default="./data/backbone")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    tag = "probe_%s_seed%d" % (a.arch, a.seed)
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    d = np.load(os.path.join(a.data, "splits.npz"))
    backbone, dim, res = build(a.arch, a.weights_dir, dev)
    print("[%s] %s, %d-d features at %dpx, %.0f pretraining images"
          % (tag, a.arch, dim, res, PRETRAIN_IMAGES[a.arch]), flush=True)

    t0 = time.time()
    Z = {k: embed(backbone, d[k], dev, res) for k in ("Xtr", "Xiv", "Xov", "Xot")}
    print("[%s] embedded in %.0fs" % (tag, time.time() - t0), flush=True)

    ytr = torch.from_numpy(d["ytr"]).long()
    head = nn.Linear(Z["Xtr"].shape[1], 2).to(dev)
    opt = torch.optim.SGD(head.parameters(), lr=a.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    lossf = nn.CrossEntropyLoss()

    best = (-1.0, None)
    for ep in range(1, a.epochs + 1):
        head.train()
        perm = torch.randperm(len(ytr))
        for k in range(0, len(perm), a.bs):
            idx = perm[k:k + a.bs]
            opt.zero_grad()
            loss = lossf(head(Z["Xtr"][idx].to(dev)), ytr[idx].to(dev))
            loss.backward()
            opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            p = torch.softmax(head(Z["Xov"].to(dev)), 1)[:, 1].cpu().numpy()
        auroc = T.compute_metrics(d["yov"], p)["auroc"]
        if auroc > best[0]:
            best = (auroc, {k: v.detach().clone() for k, v in head.state_dict().items()})
        print("[%s] ep%d/%d ood_val_auroc=%.4f" % (tag, ep, a.epochs, auroc), flush=True)

    head.load_state_dict(best[1])
    head.eval()
    res_out = {"arm": "probe_" + a.arch, "seed": a.seed, "arch": a.arch,
               "feature_dim": int(dim), "input_res": int(res),
               "pretrain_images": PRETRAIN_IMAGES[a.arch],
               "epochs": a.epochs, "selected_ood_val_auroc": float(best[0]),
               "wall_s": time.time() - t0, "device": str(dev)}
    for split, Xk, yk in (("ood_test", "Xot", "yot"), ("id_val", "Xiv", "yiv"),
                          ("ood_val", "Xov", "yov")):
        with torch.no_grad():
            p = torch.softmax(head(Z[Xk].to(dev)), 1)[:, 1].cpu().numpy()
        res_out[split] = T.compute_metrics(d[yk], p)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res_out, open(os.path.join(a.out, tag + ".json"), "w"), indent=1)
    print("[%s] DONE ood_test auroc=%.4f ece=%.4f"
          % (tag, res_out["ood_test"]["auroc"], res_out["ood_test"]["ece"]), flush=True)


if __name__ == "__main__":
    main()
