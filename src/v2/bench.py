"""GPU timing benchmark for budgeting the v2 sweep (uses the smoke cache).

  python src/v2/bench.py --tier C [--steps 300]   -> prints/writes a JSON table:
  per arm: train seconds per 100 steps (after 20 warm-up steps), eval throughput (img/s),
  plus SSL seconds per step and probe feature-extraction throughput.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import arms as A  # noqa: E402
from backbones import build_net  # noqa: E402
from make_plan import TIER_C_ARMS, TIER_S_ARMS, LADDER  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    P = C.TIER_RES[a.tier]
    fold = C.Fold(os.path.join(C.V2, "smoke", f"cache{P}", "c17", "fold0"), dev, load_u=True)
    Xbig = fold.X["ot"].repeat(25, 1, 1, 1)                     # 25,000 images for eval throughput
    res = {"tier": a.tier, "gpu": torch.cuda.get_device_name(), "arms": {}, "probes": {}, "ssl": {}}
    arms = TIER_C_ARMS if a.tier == "C" else [x for x in TIER_S_ARMS if not x.startswith("probe_")]
    for arm in arms:
        spec = A.resolve_arm(arm, a.tier)
        hp, _ = A.merge_hp(A.default_hp(spec["method"], a.tier, spec["backbone"], spec["init"]), {})
        ms = dict(p=hp["ms_p"], alpha=hp["ms_alpha"]) if spec["method"] == "mixstyle" else None
        model = build_net(spec["backbone"], spec["init"], mixstyle=ms).to(dev)
        cl = bool(hp["channels_last"])
        if cl:
            model = model.to(memory_format=torch.channels_last)
        tr = A.Trainer(spec["method"], model, hp, fold.K, a.steps + 20, dev)
        smp = C.EpochSampler(fold.n("tr"), hp["bs"], 0, dev)
        X = fold.X["tr"]
        for i in range(a.steps + 20):
            if i == 20:
                sync()
                t0 = time.time()
            idx = smp.next()
            tr.step(C.to_float(X[idx], dev), fold.yt["tr"][idx], fold.d_tr[idx])
        sync()
        t_train = (time.time() - t0) / a.steps * 100
        C.predict(model, Xbig[:2048], dev, amp=False, channels_last=cl)   # fp32 eval as in run_v2
        sync()
        t0 = time.time()
        C.predict(model, Xbig, dev, amp=False, channels_last=cl)
        sync()
        ips = len(Xbig) / (time.time() - t0)
        res["arms"][arm] = dict(train_s_per_100=t_train, eval_img_per_s=ips, backbone=spec["backbone"])
        print(f"{arm:22s} {spec['backbone']:9s} train {t_train:6.2f}s/100 steps  eval {ips:9.0f} img/s", flush=True)
        del model, tr
        torch.cuda.empty_cache()
    if a.tier == "S":
        for arm in LADDER:
            spec = A.resolve_arm(arm, "S")
            net = build_net(spec["backbone"], spec["init"], probe=True).to(dev).eval()
            with torch.no_grad():   # fp32 extraction as in run_v2.extract
                net.features(C.to_float(Xbig[:512], dev))
                sync()
                t0 = time.time()
                for k in range(0, 10240, 512):
                    net.features(C.to_float(Xbig[k:k + 512], dev))
                sync()
            ips = 10240 / (time.time() - t0)
            res["probes"][arm] = dict(extract_img_per_s=ips)
            print(f"{arm:30s} extract {ips:8.0f} img/s", flush=True)
            del net
            torch.cuda.empty_cache()
    # SSL step time
    import ssl_v2
    bb = "cnn4" if a.tier == "C" else "resnet50"
    init = "random" if bb == "cnn4" else "imagenet"
    Xu = fold.Xu
    t0 = time.time()
    _, info = ssl_v2.pretrain(Xu, bb, init, 0, dev, cfg=dict(epochs=2), log=lambda m: None)
    per_step = (info["history"][1]["elapsed"] - info["history"][0]["elapsed"]) / (info["steps"] / 2)
    res["ssl"] = dict(backbone=bb, s_per_step=per_step, s_per_epoch_40k=per_step * (40000 // 256))
    print(f"SSL {bb}: {per_step * 1000:.1f} ms/step -> {per_step * 156:.1f}s per epoch on 40k images", flush=True)
    out = a.out or os.path.join(C.V2, "smoke", f"bench_{a.tier}.json")
    C.atomic_json(res, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
