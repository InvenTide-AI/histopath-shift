"""SimCLR pre-training on the fold's unlabelled pool Xu (v2).

cnn4      : as v1 (pretrain_ssl_gpu_ext.py): NT-Xent tau 0.5, bs 256, 6 epochs, SGD lr 0.05,
            momentum 0.9, wd 1e-4, cosine per step, views = v1 simclr_view_gpu,
            projection head 256-256-128, fp32.
resnet50  : continued from the ImageNet init, SGD lr 0.01, 6 epochs, bf16 autocast,
            channels_last, projection head 2048-2048-128.
Xu excludes id-val groups, the val domain and the test domain (CONTRACTS.md A, fix D11).

Output: {ssl_root}/{tier}/{cohort}/fold{t}/{backbone}_{init}_seed{s}.pt  (encoder state_dict)
        ... _seed{s}.json  (history incl. steps and images_seen)

CLI:
  python src/v2/ssl_v2.py --tier C --cohort c17 --fold 0 --seeds 0,1,2,3,4
  python src/v2/ssl_v2.py --tier S --cohort c17 --fold 0 --backbone resnet50 --init imagenet --seeds 0,1,2
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
import common as C  # noqa: E402
from arms import simclr_view  # noqa: E402
from backbones import build_net, default_init  # noqa: E402


def nt_xent(z1, z2, temperature=0.5):
    N = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], 0).float(), dim=1)
    sim = (z @ z.T) / temperature
    sim.fill_diagonal_(-1e9)
    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, targets)


def ssl_paths(ssl_root, tier, cohort, fold, backbone, init, seed):
    d = os.path.join(ssl_root, tier, cohort, f"fold{fold}")
    stem = os.path.join(d, f"{backbone}_{init}_seed{seed}")
    return stem + ".pt", stem + ".json"


def default_ssl_cfg(backbone):
    if backbone == "cnn4":
        return dict(epochs=6, bs=256, lr=0.05, temp=0.5, wd=1e-4, momentum=0.9, amp=0, hidden=256, out=128)
    return dict(epochs=6, bs=256, lr=0.01, temp=0.5, wd=1e-4, momentum=0.9, amp=1, hidden=2048, out=128)


def pretrain(Xu, backbone, init, seed, device, cfg=None, log=print):
    """Xu: uint8 NHWC tensor (any device). Returns (encoder state_dict on CPU, history dict)."""
    cfg = {**default_ssl_cfg(backbone), **(cfg or {})}
    C.set_seed(seed)
    net = build_net(backbone, init).to(device)
    head = nn.Sequential(nn.Linear(net.out_dim, cfg["hidden"]), nn.ReLU(inplace=True),
                         nn.Linear(cfg["hidden"], cfg["out"])).to(device)
    cl = backbone != "cnn4"
    amp = bool(cfg["amp"]) and torch.device(device).type == "cuda"
    if cl:
        net = net.to(memory_format=torch.channels_last)
    params = list(net.encoder.parameters()) + list(head.parameters())
    opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["wd"])
    n = int(Xu.shape[0])
    spe = max(1, n // cfg["bs"])
    steps = cfg["epochs"] * spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, C.cosine_lambda(steps))
    rng = np.random.RandomState(1000 + seed)
    hist, t0 = [], time.time()
    net.train(); head.train()
    for ep in range(cfg["epochs"]):
        perm = torch.from_numpy(rng.permutation(n))
        tot = torch.zeros((), device=device)
        for b in range(spe):
            idx = perm[b * cfg["bs"]:(b + 1) * cfg["bs"]].to(Xu.device)
            x = C.to_float(Xu[idx], device)
            v1, v2 = simclr_view(x), simclr_view(x)
            if cl:
                v1 = v1.contiguous(memory_format=torch.channels_last)
                v2 = v2.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                z1, z2 = head(net.features(v1)), head(net.features(v2))
            loss = nt_xent(z1, z2, cfg["temp"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.detach()
        hist.append(dict(epoch=ep + 1, nt_xent=float(tot) / spe, elapsed=time.time() - t0))
        log(f"[ssl {backbone}_{init} s{seed}] ep{ep + 1}/{cfg['epochs']} nt_xent={hist[-1]['nt_xent']:.4f} "
            f"({time.time() - t0:.0f}s)")
    info = dict(backbone=backbone, init=init, seed=seed, cfg=cfg, n_unlabeled=n, steps=steps,
                images_seen=int(steps * 2 * cfg["bs"]), chance_loss=float(np.log(2 * cfg["bs"] - 1)),
                wall_s=time.time() - t0, device=str(device), torch=torch.__version__,
                code_version=C.code_version(), history=hist)
    state = {k: v.detach().cpu() for k, v in net.encoder.state_dict().items()}
    return state, info


def checkpoint_status(pt, js, fingerprint=None):
    """'ok' if both the encoder .pt and the .json exist (and the json's cache fingerprint
    matches `fingerprint` when given), else 'missing' / 'stale'."""
    if not (os.path.exists(pt) and os.path.exists(js)):
        return "missing"
    if fingerprint is not None:
        try:
            info = json.load(open(js))
        except (OSError, ValueError):
            return "missing"
        if info.get("cache_fingerprint") != fingerprint:
            return "stale"
    return "ok"


def run_and_save(Xu, ssl_root, tier, cohort, fold, backbone, init, seed, device, cfg=None, log=print, force=False,
                 fingerprint=None, cache_dir=None):
    """Pre-train unless a complete checkpoint for the same cache fingerprint exists.
    The .pt is written first and the .json last (both atomically, unique temp names);
    a checkpoint counts as present only when both exist, so a crash between the two
    writes just triggers a re-run."""
    pt, js = ssl_paths(ssl_root, tier, cohort, fold, backbone, init, seed)
    st = checkpoint_status(pt, js, fingerprint)
    if st == "ok" and not force:
        return pt, json.load(open(js))
    if st == "stale":
        log(f"[ssl] {js} was trained on a different cache (fingerprint mismatch) -> re-training")
    os.makedirs(os.path.dirname(pt), exist_ok=True)
    state, info = pretrain(Xu, backbone, init, seed, device, cfg, log)
    info["cache_fingerprint"] = fingerprint
    info["cache_dir"] = cache_dir
    C.atomic_torch(state, pt)
    C.atomic_json(info, js)
    return pt, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["C", "S"])
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--init", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--cache_root", default=C.V2)
    ap.add_argument("--cache_dir", default=None, help="override fold cache directory")
    ap.add_argument("--ssl_root", default=os.path.join(C.V2, "ssl"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    bb = a.backbone or ("cnn4" if a.tier == "C" else "resnet50")
    init = a.init or default_init(bb)
    cfg = {}
    if a.epochs is not None:
        cfg["epochs"] = a.epochs
    if a.lr is not None:
        cfg["lr"] = a.lr
    path = a.cache_dir or C.fold_dir(a.cache_root, a.tier, a.cohort, a.fold)
    Xu = C.np.load(os.path.join(path, "Xu.npy"), mmap_mode="r")
    Xu = torch.from_numpy(np.ascontiguousarray(Xu)).to(a.device)
    fp = C.cache_fingerprint(path)
    for s in [int(x) for x in a.seeds.split(",")]:
        pt, info = run_and_save(Xu, a.ssl_root, a.tier, a.cohort, a.fold, bb, init, s, a.device, cfg,
                                log=lambda m: print(m, flush=True), force=a.force, fingerprint=fp, cache_dir=path)
        print(f"saved {pt} images_seen={info['images_seen']} wall={info['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
