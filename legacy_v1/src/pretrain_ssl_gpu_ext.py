"""GPU-aware SimCLR pre-training for Camelyon17-WILDS fold caches.

Mirrors src/pretrain_ssl.py exactly in objective, augmentation, and schedule,
but sets the compute device (default cuda:0) and moves inputs each step so
the H100 does the work. Same output file names / same weight-tensor layout,
so downstream run_arm_ext.py --arm ssl / ssl_sam picks up the encoder
unchanged.
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_lib as T


def simclr_view_gpu(x):
    """Same augmentation as train_lib.simclr_view but keeps every tensor
    on x.device -- the paper's CPU-only version silently allocates CPU
    tensors for colour jitter which fails when x lives on CUDA."""
    dev = x.device
    B, C, H, W = x.shape
    # random resized crop
    s = float(np.random.uniform(0.4, 1.0))
    h = max(8, int(round(H * np.sqrt(s))))
    w = max(8, int(round(W * np.sqrt(s))))
    i = np.random.randint(0, H - h + 1); j = np.random.randint(0, W - w + 1)
    x = F.interpolate(x[:, :, i:i + h, j:j + w], size=(H, W),
                      mode="bilinear", align_corners=False)
    if np.random.rand() < 0.5: x = torch.flip(x, [3])
    if np.random.rand() < 0.5: x = torch.flip(x, [2])
    k = int(np.random.randint(0, 4))
    if k: x = torch.rot90(x, k, [2, 3])
    if np.random.rand() < 0.8:
        b = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.6, 1.4)
        c = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.6, 1.4)
        s = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.6, 1.4)
        x = x * b
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * c + mean
        gray = x.mean(dim=1, keepdim=True)
        x = (x - gray) * s + gray
        hue = torch.empty(B, 3, 1, 1, device=dev).uniform_(-0.08, 0.08)
        x = x + hue
    if np.random.rand() < 0.2:
        x = x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    return x.clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--data", default="cache")
    a = ap.parse_args()

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.RandomState(1000 + a.seed)

    Xu = np.load(os.path.join(a.data, "unlabeled.npz"))["Xu"]
    enc = T.Encoder().to(dev)
    head = T.ProjectionHead(enc.out_dim).to(dev)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
    spe = max(1, len(Xu) // a.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * spe)

    hist, t0 = [], time.time()
    for ep in range(a.epochs):
        enc.train(); head.train()
        tot, nb = 0.0, 0
        # Normalization constants live on CPU in train_lib; move to device once.
        mean = T.IMAGENET_MEAN.to(dev); std = T.IMAGENET_STD.to(dev)
        for xb, _ in T.iterate(Xu, None, a.bs, shuffle=True, rng=rng):
            x = T.to_float(xb).to(dev)
            v1 = (simclr_view_gpu(x) - mean) / std
            v2 = (simclr_view_gpu(x) - mean) / std
            z1 = head(enc(v1)); z2 = head(enc(v2))
            loss = T.nt_xent(z1, z2, a.temp)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item(); nb += 1
        hist.append(dict(epoch=ep + 1, nt_xent=tot / max(nb, 1),
                         elapsed=time.time() - t0))
        print(f"[ssl_seed{a.seed}] ep{ep+1}/{a.epochs} "
              f"nt_xent={tot / max(nb, 1):.4f} ({time.time() - t0:.0f}s)", flush=True)

    torch.save({k: v.detach().cpu() for k, v in enc.state_dict().items()},
               os.path.join(a.data, f"ssl_encoder_seed{a.seed}.pt"))
    json.dump(dict(seed=a.seed, epochs=a.epochs, bs=a.bs, lr=a.lr,
                   temp=a.temp, n_unlabeled=int(len(Xu)),
                   chance_loss=float(np.log(2 * a.bs - 1)),
                   device=str(dev), history=hist),
              open(os.path.join(a.data, f"ssl_history_seed{a.seed}.json"), "w"), indent=1)
    print(f"[ssl_seed{a.seed}] saved encoder to {a.data}", flush=True)


if __name__ == "__main__":
    main()
