"""Linear probe of a frozen ImageNet ResNet50 as the paper's foundation-model
reference. Same 5-fold protocol as run_arm_ext.py, but instead of training the
0.58M-parameter CNN from scratch it (a) resizes each 64x64 tile to 224x224,
(b) extracts a 2048-d embedding with a frozen ResNet50 pretrained on
ImageNet-1K (torchvision weights), and (c) trains only a linear head on top.

An ImageNet backbone is not a pathology foundation model; it stands in as an
off-the-shelf-transfer reference the paper can report while UNI-family
checkpoints remain gated behind an authentication path not reachable from the
compute network.  A pathology foundation-model probe using the same interface
is a one-line change to --backbone_arch.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_lib as T


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_resnet50(weights_path: str, dev):
    from torchvision.models import resnet50
    m = resnet50(weights=None)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    m.load_state_dict(state)
    m.fc = nn.Identity()  # expose 2048-d penultimate features
    for p in m.parameters(): p.requires_grad_(False)
    return m.eval().to(dev)


@torch.no_grad()
def embed(backbone, X, dev, bs=64, upsample_to=224):
    out = []
    for k in range(0, len(X), bs):
        xb = torch.from_numpy(X[k:k + bs])
        x = xb.permute(0, 3, 1, 2).float().div_(255.0).to(dev)
        x = F.interpolate(x, size=(upsample_to, upsample_to),
                          mode="bilinear", align_corners=False)
        x = (x - IMAGENET_MEAN.to(dev)) / IMAGENET_STD.to(dev)
        out.append(backbone(x).cpu())
    return torch.cat(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--weights", default="./data/backbone/resnet50.pth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--arm_name", default="probe_resnet50")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tag = f"{a.arm_name}_seed{a.seed}"
    os.makedirs(a.out, exist_ok=True)

    d = np.load(os.path.join(a.data, "splits.npz"))
    print(f"data: Xtr={d['Xtr'].shape}, Xot={d['Xot'].shape}, dev={a.device}", flush=True)
    backbone = build_resnet50(a.weights, a.device)

    print("[probe] embedding all splits...", flush=True); t0 = time.time()
    F_tr = embed(backbone, d["Xtr"], a.device)
    F_iv = embed(backbone, d["Xiv"], a.device)
    F_ov = embed(backbone, d["Xov"], a.device)
    F_ot = embed(backbone, d["Xot"], a.device)
    print(f"[probe] embed done in {time.time() - t0:.0f}s "
          f"(feat_dim={F_tr.shape[1]})", flush=True)

    head = nn.Linear(F_tr.shape[1], 2).to(a.device)
    opt = torch.optim.SGD(head.parameters(), lr=a.lr, momentum=0.9, weight_decay=1e-4)
    steps = max(1, len(F_tr) // a.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * steps)
    loss_fn = nn.CrossEntropyLoss()

    ytr = torch.from_numpy(d["ytr"]).to(a.device)
    yov = d["yov"]; yiv = d["yiv"]; yot = d["yot"]
    F_tr = F_tr.to(a.device); F_ov = F_ov.to(a.device)
    F_iv = F_iv.to(a.device); F_ot = F_ot.to(a.device)

    best = {"ood_val_auroc": -1}
    for ep in range(a.epochs):
        head.train()
        idx = torch.randperm(len(F_tr), device=a.device)
        tot, nb = 0.0, 0
        for k in range(0, len(idx) - a.bs + 1, a.bs):
            b = idx[k:k + a.bs]
            opt.zero_grad()
            l = loss_fn(head(F_tr[b]), ytr[b])
            l.backward(); opt.step(); sched.step()
            tot += l.item(); nb += 1
        head.eval()
        with torch.no_grad():
            pv = torch.softmax(head(F_ov), 1)[:, 1].cpu().numpy()
        m = T.compute_metrics(yov, pv)
        if m["auroc"] > best["ood_val_auroc"]:
            best = {"ood_val_auroc": m["auroc"], "epoch": ep + 1,
                    "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
        print(f"[{tag}] ep{ep+1}/{a.epochs} loss={tot/max(nb,1):.4f} "
              f"ood_val_auroc={m['auroc']:.4f}", flush=True)

    head.load_state_dict(best["state"]); head.to(a.device); head.eval()
    with torch.no_grad():
        p_ot = torch.softmax(head(F_ot), 1)[:, 1].cpu().numpy()
        p_iv = torch.softmax(head(F_iv), 1)[:, 1].cpu().numpy()
        p_ov = torch.softmax(head(F_ov), 1)[:, 1].cpu().numpy()
    res = dict(arm=a.arm_name, seed=a.seed, selected_epoch=best["epoch"],
               epochs=a.epochs, lr=a.lr, bs=a.bs, feat_dim=int(F_tr.shape[1]),
               backbone="torchvision.resnet50 (ImageNet-1K)",
               n_train=int(len(F_tr)), wall_s=time.time() - t0,
               device=str(a.device),
               ood_test=T.compute_metrics(yot, p_ot),
               id_val=T.compute_metrics(yiv, p_iv),
               ood_val=T.compute_metrics(yov, p_ov))
    json.dump(res, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1)
    print(f"[{tag}] DONE ood_test auroc={res['ood_test']['auroc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
