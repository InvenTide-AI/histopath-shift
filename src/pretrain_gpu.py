"""SimCLR pre-training on one fold's TRAINING hospitals only.

Why this exists separately from pretrain_ssl.py: that script pre-trains on the
single-holdout experiment's cache with a small CNN. Here the encoder must match
the sweep's backbone (ResNet-50) and, critically, must see ONLY the training
sites of the fold it will be used on -- pre-training on all sites would leak the
held-out hospital's appearance into the representation, which is exactly the
generalisation the sweep is trying to measure.

One encoder per (fold, seed), written to
    cache/ssl_encoder_<arch>_hold<fold>_seed<seed>.pt
which is the path train_gpu.py --ssl-encoder expects.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gpu import Net, ArrayDataset, _Indexed  # noqa: E402
from train_lib import nt_xent  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--holdout", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arch", default="resnet50")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--cache-size", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--temp", type=float, default=0.5)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--cache", default="cache")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (torch.bfloat16 if (dev.type == "cuda" and not a.no_amp) else None)

    import pandas as pd
    meta = pd.read_parquet(os.path.join(a.cache, "camelyon17_metadata.parquet"))
    meta["center"] = meta["center"].astype(str)
    sp = np.load(os.path.join(a.cache, "hospital_splits.npz"))
    cs = a.cache_size or a.size
    X = np.load(os.path.join(a.cache, f"decoded_{cs}.npy"), mmap_mode="r")
    dec = json.load(open(os.path.join(a.cache, f"decoded_{cs}_index.json")))
    row_to_pos = {int(r): i for i, r in enumerate(dec["row_ids"])}

    # Training rows only. The held-out site never appears here.
    idx = sp[f"{a.holdout}__train"]
    held = set(meta.loc[idx, "center"].unique())
    assert a.holdout not in held, \
        f"held-out site {a.holdout} present in its own pre-training pool"
    if a.limit:
        rs = np.random.default_rng(a.seed)
        idx = idx[np.sort(rs.choice(len(idx), min(a.limit, len(idx)), replace=False))]
    pos = np.array([row_to_pos[int(r)] for r in idx], dtype=np.int64)
    y = meta.label.to_numpy()[idx]
    print(f"pre-training on {len(idx)} patches from sites {sorted(held)} "
          f"(holdout {a.holdout} excluded)", flush=True)

    ds = ArrayDataset(_Indexed(X, pos), y, a.size, True, {"simclr"},
                      seed=a.seed, two_view=True)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=a.bs, shuffle=True, num_workers=a.workers,
        pin_memory=(dev.type == "cuda"), drop_last=True,
        persistent_workers=a.workers > 0)

    model = Net(a.arch, pretrained=False).to(dev).to(memory_format=torch.channels_last)
    proj = torch.nn.Sequential(
        torch.nn.Linear(model.feat_dim, 512), torch.nn.ReLU(inplace=True),
        torch.nn.Linear(512, a.proj_dim)).to(dev)
    opt = torch.optim.SGD(list(model.parameters()) + list(proj.parameters()),
                          lr=a.lr, momentum=0.9, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * len(dl))

    t0, hist = time.time(), []
    for ep in range(1, a.epochs + 1):
        model.train(); proj.train()
        tot, nb = 0.0, 0
        for batch in dl:
            v1, v2 = batch[0], batch[1]
            v1 = v1.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            v2 = v2.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                _, feat = model(torch.cat([v1, v2], 0), return_feat=True)
                z = proj(feat)
                z = F.normalize(z, dim=1)
                loss = nt_xent(z[:len(v1)], z[len(v1):], a.temp)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step(); sched.step()
            tot += float(loss.detach()); nb += 1
        hist.append(tot / max(nb, 1))
        print(f"  epoch {ep}/{a.epochs} nt_xent={hist[-1]:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    if a.limit:
        print(f"[limit={a.limit}] no encoder written (subsampled run)")
        return
    out = os.path.join(a.cache,
                       f"ssl_encoder_{a.arch}_hold{a.holdout}_seed{a.seed}.pt")
    torch.save(model.encoder.state_dict(), out)
    json.dump(dict(holdout=a.holdout, seed=a.seed, arch=a.arch, size=a.size,
                   epochs=a.epochs, bs=a.bs, lr=a.lr, temp=a.temp,
                   n_patches=int(len(idx)), sites_used=sorted(held),
                   loss_per_epoch=hist, wall_s=time.time() - t0),
              open(out.replace(".pt", "_meta.json"), "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
