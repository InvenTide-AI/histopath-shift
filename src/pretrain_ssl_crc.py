"""Pillar I: SimCLR pre-training on UNLABELED NCT-CRC-HE tiles.

Trains only on colour-NORMALISED unlabeled tiles disjoint from train/id_val.
The NONORM cohort (the primary shift test condition) and CRC_VAL_HE_7K never
appear here -- pre-training on the shifted appearance would leak the test
condition. Labels are not used.
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

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=6)
p.add_argument("--bs", type=int, default=256)
p.add_argument("--lr", type=float, default=0.05)
p.add_argument("--temp", type=float, default=0.5)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--data", default="cache")
# Pool-size test (DESIGN_crc_pool40k.md). Defaults reproduce the published
# runs exactly: --pool crc_unlabeled.npz, --tag "".
p.add_argument("--pool", default="crc_unlabeled.npz",
               help="unlabeled pool npz inside --data (key 'Xu')")
p.add_argument("--tag", default="",
               help="suffix for encoder/history filenames, e.g. _pool40k")
a = p.parse_args()

torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)
np.random.seed(a.seed)
rng = np.random.RandomState(1000 + a.seed)

Xu = np.load(os.path.join(a.data, a.pool))["Xu"]
print(f"[ssl_crc_seed{a.seed}{a.tag}] pool={a.pool} n={len(Xu)}", flush=True)
enc = T.Encoder()
head = T.ProjectionHead(enc.out_dim)
params = list(enc.parameters()) + list(head.parameters())
opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
spe = len(Xu) // a.bs
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * spe)

hist, t0 = [], time.time()
for ep in range(a.epochs):
    enc.train()
    head.train()
    tot, nb = 0.0, 0
    for xb, _ in T.iterate(Xu, None, a.bs, shuffle=True, rng=rng):
        x = T.to_float(xb)
        z1 = head(enc(T.normalize(T.simclr_view(x))))
        z2 = head(enc(T.normalize(T.simclr_view(x))))
        loss = T.nt_xent(z1, z2, a.temp)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        tot += loss.item()
        nb += 1
    hist.append(dict(epoch=ep + 1, nt_xent=tot / max(nb, 1),
                     elapsed=time.time() - t0))
    print(f"[ssl_crc_seed{a.seed}] ep{ep + 1}/{a.epochs} "
          f"nt_xent={tot / max(nb, 1):.4f} ({time.time() - t0:.0f}s)", flush=True)

torch.save(enc.state_dict(),
           os.path.join(a.data, f"ssl_encoder_crc_seed{a.seed}{a.tag}.pt"))
json.dump(dict(seed=a.seed, epochs=a.epochs, bs=a.bs, lr=a.lr, temp=a.temp,
               pool=a.pool, n_unlabeled=int(len(Xu)),
               chance_loss=float(np.log(2 * a.bs - 1)), history=hist),
          open(os.path.join(a.data,
                            f"ssl_history_crc_seed{a.seed}{a.tag}.json"), "w"),
          indent=1)
print(f"[ssl_crc_seed{a.seed}] saved encoder", flush=True)
