"""SimCLR pre-training on a chest-radiograph unlabelled pool.

Two pools, identical protocol, so that the pool composition is the only
difference between the arms it feeds:
  S = NIH only            (single-site)
  M = NIH + CheXpert      (site-spanning; no CheXpert *label* is ever read)

Hyperparameters carried over unchanged from pretrain_ssl_crc.py; only the input
resolution (96px) and the pool file differ.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_lib as T
from train_lib import Encoder, ProjectionHead, nt_xent

CACHE = "cache_cxr"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["S", "M"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=5)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    Xu = np.load(f"{CACHE}/cxr_pool_{a.pool}.npz")["Xu"]
    print(f"[ssl_{a.pool}_s{a.seed}] pool n={len(Xu)}", flush=True)

    enc, head = Encoder(), ProjectionHead(256)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=a.lr)
    enc.train(), head.train()

    for ep in range(a.epochs):
        t0 = time.time()
        perm = rng.permutation(len(Xu))
        tot = nb = 0.0
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            xb = T.to_float_gray(torch.from_numpy(Xu[perm[i:i + a.bs]]))
            v1 = T.normalize(T.simclr_view_cxr(xb))
            v2 = T.normalize(T.simclr_view_cxr(xb))
            z1, z2 = head(enc(v1)), head(enc(v2))
            loss = nt_xent(z1, z2)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        print(f"[ssl_{a.pool}_s{a.seed}] ep{ep} loss={tot / max(nb, 1):.4f} "
              f"{time.time() - t0:.0f}s", flush=True)

    os.makedirs("cache_cxr", exist_ok=True)
    out = f"{CACHE}/ssl_encoder_cxr_{a.pool}_seed{a.seed}.pt"
    torch.save(enc.state_dict(), out)
    json.dump(dict(pool=a.pool, seed=a.seed, epochs=a.epochs, n=len(Xu)),
              open(out.replace(".pt", ".json"), "w"), indent=1)
    print(f"[ssl_{a.pool}_s{a.seed}] saved {out}", flush=True)


if __name__ == "__main__":
    main()
