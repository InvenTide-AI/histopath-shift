"""Pillar I: SimCLR pre-training on UNLABELED Camelyon17 patches.

Trains only on unlabeled patches from the training hospitals (0/3/4).
The held-out test hospital (2) never appears here — no leakage.
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_lib as T

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=6)
p.add_argument("--bs", type=int, default=256)
p.add_argument("--lr", type=float, default=0.05)
p.add_argument("--temp", type=float, default=0.5)
p.add_argument("--threads", type=int, default=2)
p.add_argument("--data", default="cache")
a = p.parse_args()

torch.set_num_threads(a.threads)
torch.manual_seed(a.seed); np.random.seed(a.seed)
rng = np.random.RandomState(1000 + a.seed)

Xu = np.load(os.path.join(a.data, "unlabeled.npz"))["Xu"]
enc = T.Encoder()
head = T.ProjectionHead(enc.out_dim)
params = list(enc.parameters()) + list(head.parameters())
opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
spe = len(Xu) // a.bs
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * spe)

hist, t0 = [], time.time()
for ep in range(a.epochs):
    enc.train(); head.train()
    tot, nb = 0.0, 0
    for xb, _ in T.iterate(Xu, None, a.bs, shuffle=True, rng=rng):
        x = T.to_float(xb)
        v1 = T.normalize(T.simclr_view(x))
        v2 = T.normalize(T.simclr_view(x))
        z1 = head(enc(v1)); z2 = head(enc(v2))
        loss = T.nt_xent(z1, z2, a.temp)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tot += loss.item(); nb += 1
    hist.append(dict(epoch=ep + 1, nt_xent=tot / max(nb, 1),
                     elapsed=time.time() - t0))
    print(f"[ssl_seed{a.seed}] ep{ep+1}/{a.epochs} nt_xent={tot/max(nb,1):.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

torch.save(enc.state_dict(), os.path.join(a.data, f"ssl_encoder_seed{a.seed}.pt"))
json.dump(dict(seed=a.seed, epochs=a.epochs, bs=a.bs, lr=a.lr, temp=a.temp,
               n_unlabeled=int(len(Xu)), chance_loss=float(np.log(2 * a.bs - 1)),
               history=hist),
          open(os.path.join(a.data, f"ssl_history_seed{a.seed}.json"), "w"), indent=1)
print(f"[ssl_seed{a.seed}] saved encoder", flush=True)
