"""v2 arms (PLAN §3) with the audit fixes D1-D11.

An arm name resolves to (method, backbone, init):
  <method>                     tier C: cnn4/random; tier S: resnet50/imagenet
  <method>_<lunit init>        e.g. erm_lunit_bt (resnet50), he_jitter_lunit_dino (vit_s16)
  probe_<backbone>_<init>      frozen linear probe, e.g. probe_resnet50_lunit_bt, probe_vit_b16_swag

Methods: erm sam simclr simclr_sam augonly compute_matched groupdro irm coral mixstyle
         fish lisa macenko he_jitter tia_style macenko_tile   (rotinv dropped, D3)
macenko / tia_style use group-level ("per-slide") Macenko (mac_level='group', stain.py);
macenko_tile is the per-tile variant (= macenko with mac_level='tile', appendix). The group-level
arms fit the target with the same group estimator (mac_ref='train_group'), per slide x domain
(mac_group_unit='slide_domain') and with the pooled 99th-percentile maxC (mac_group_maxc='pooled').
Test-time adaptation (bnadapt, tent) is a per-job option evaluated on the ID-selected
checkpoint (run_v2.py), with a seeded shuffled test stream (D8).

Every hyperparameter has a default in `default_hp` and can be overridden with
--hp key=value; the full dict is recorded in the run JSON.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import bn_modules, default_init
from common import cosine_lambda

METHODS = ("erm", "sam", "simclr", "simclr_sam", "augonly", "compute_matched", "groupdro", "irm",
           "coral", "mixstyle", "fish", "lisa", "macenko", "he_jitter", "tia_style", "macenko_tile")
LUNIT_INITS = ("lunit_bt", "lunit_swav", "lunit_mocov2", "lunit_dino")
PROBE_BACKBONES = ("convnext_tiny", "resnet18", "resnet50", "vit_b16", "vit_s16")
USES_DOMAINS = ("groupdro", "irm", "coral", "fish", "lisa")
USES_MACENKO = ("macenko", "tia_style", "macenko_tile")
USES_SSL = ("simclr", "simclr_sam")
USES_SAM = ("sam", "simclr_sam")


# ============================================================================ arm resolution
def resolve_arm(arm: str, tier: str) -> dict:
    if arm.startswith("probe_"):
        rest = arm[len("probe_"):]
        for bb in sorted(PROBE_BACKBONES, key=len, reverse=True):
            if rest.startswith(bb + "_"):
                return dict(method="probe", backbone=bb, init=rest[len(bb) + 1:], probe=True)
        raise ValueError(f"cannot parse probe arm {arm!r}: expected probe_<backbone>_<init>")
    for ini in LUNIT_INITS:
        if arm.endswith("_" + ini):
            method = arm[: -len(ini) - 1]
            bb = "vit_s16" if ini == "lunit_dino" else "resnet50"
            assert method in METHODS, arm
            return dict(method=method, backbone=bb, init=ini, probe=False)
    if arm not in METHODS:
        raise ValueError(f"unknown arm {arm!r}; methods={METHODS}")
    bb = "cnn4" if tier == "C" else "resnet50"
    return dict(method=arm, backbone=bb, init=default_init(bb), probe=False)


def default_hp(method: str, tier: str, backbone: str, init: str) -> dict:
    vit = backbone.startswith("vit")
    hp = dict(steps=1200, bs=128, eval_every=0,  # 0 -> total_steps // 10
              opt="adamw" if vit else "sgd",
              lr=(3e-5 if vit else (0.02 if tier == "C" else 3e-3)),
              momentum=0.9, wd=(0.05 if vit else 1e-4),
              warmup_frac=(0.05 if vit else 0.0),
              amp=int(tier == "S"), channels_last=int(backbone in ("resnet50", "resnet18")),
              aug="dihedral")
    if method == "probe":
        # features are extracted in fp32 (run_v2.extract); amp is not used by probes
        hp.update(lr=0.05, wd=1e-4, opt="sgd", warmup_frac=0.0, feat_std=1, amp=0, channels_last=0,
                  aug="none")
        return hp
    if method in USES_SAM:
        hp["rho"] = 0.05
    if method == "groupdro":
        hp["eta"] = 0.01
    if method == "irm":
        hp.update(irm_lambda=10.0, irm_anneal_frac=0.25, irm_reset_opt=1)
    if method == "coral":
        hp["coral_lambda"] = 1.0
    if method == "mixstyle":
        hp.update(ms_p=0.5, ms_alpha=0.1)
    if method == "fish":
        hp["meta_lr"] = 0.1
    if method == "lisa":
        hp.update(lisa_alpha=2.0, p_sel=0.5)
    if method == "he_jitter":
        hp["jitter"] = 0.4
    if method == "tia_style":
        hp["jitter"] = 0.3
    if method in USES_MACENKO:
        grp = method != "macenko_tile"
        hp.update(# target fit: 'train_group' = the group estimator on every training group (median over
                  # groups; same estimator as the source), 'train' = per-tile estimator on mac_nref tiles
                  mac_ref=("train_group" if grp else "train"), mac_nref=256, mac_io=240.0, mac_alpha=1.0,
                  mac_beta=0.15, mac_min_tissue_frac=0.05,
                  mac_nonneg=1,   # 1: concentrations clamped >= 0 (stain.py); records without it are pre-fix
                  # 'group': one stain matrix per group id (slide / image), 'tile': per tile. Records
                  # without mac_level are per-tile (stale for macenko / tia_style -> re-run).
                  mac_level=("group" if grp else "tile"),
                  mac_group_tiles=256, mac_group_px=200_000, mac_group_min_px=2048,
                  # group maxC = 99th percentile over all pixels of the group's sampled tiles (standard)
                  mac_group_maxc="pooled",
                  # group = slide (c17: WILDS slide, not patient) x domain id (canine tr/iv: x scanner)
                  mac_group_unit="slide_domain")
    if method == "augonly":
        hp.update(rrc_min=0.4, cj_p=0.8, cj=0.4, hue=0.08, gray_p=0.2)
    if method in USES_SSL or method == "compute_matched":
        cnn = backbone == "cnn4"
        hp.update(ssl_epochs=6, ssl_bs=256, ssl_lr=(0.05 if cnn else 0.01), ssl_temp=0.5)
    return hp


def parse_hp_value(v: str):
    if isinstance(v, (int, float)):
        return v
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def merge_hp(defaults: dict, overrides: dict) -> tuple[dict, str]:
    """Returns (full hp, hpkey). Unknown keys raise. hpkey = sorted non-default k=v."""
    hp = dict(defaults)
    diff = []
    for k, v in (overrides or {}).items():
        if k not in defaults:
            raise KeyError(f"unknown hyperparameter {k!r}; known: {sorted(defaults)}")
        v = parse_hp_value(v)
        d = defaults[k]
        if isinstance(d, float) and isinstance(v, int):
            v = float(v)
        hp[k] = v
        if v != d:
            diff.append(k)
    from common import fmt_val
    key = ",".join(f"{k}={fmt_val(hp[k])}" for k in sorted(diff)) or "default"
    return hp, key


def ssl_images_seen(n_u: int, epochs: int, bs: int) -> int:
    """images through the encoder during SimCLR pre-training: steps * 2 views * bs."""
    return int(epochs * (n_u // bs) * 2 * bs)


def total_steps(method: str, hp: dict, n_u: int | None = None) -> int:
    """Step budget. compute_matched: 1200 + ceil(ssl_images_seen / bs) (D9)."""
    if method == "compute_matched":
        assert n_u is not None
        return int(hp["steps"] + math.ceil(ssl_images_seen(n_u, hp["ssl_epochs"], hp["ssl_bs"]) / hp["bs"]))
    return int(hp["steps"])


# ============================================================================ augmentation
def sup_augment(x):
    """Shared by every supervised arm (as v1): random flip H/V + rot90 (per batch)."""
    if np.random.rand() < 0.5:
        x = torch.flip(x, [3])
    if np.random.rand() < 0.5:
        x = torch.flip(x, [2])
    k = int(np.random.randint(0, 4))
    if k:
        x = torch.rot90(x, k, [2, 3])
    return x


def simclr_view(x, rrc_min=0.4, cj_p=0.8, cj=0.4, hue=0.08, gray_p=0.2):
    """Identical to v1 pretrain_ssl_gpu_ext.simclr_view_gpu at default arguments."""
    dev = x.device
    B, C, H, W = x.shape
    s = float(np.random.uniform(rrc_min, 1.0))
    h = max(8, int(round(H * np.sqrt(s))))
    w = max(8, int(round(W * np.sqrt(s))))
    i = np.random.randint(0, H - h + 1)
    j = np.random.randint(0, W - w + 1)
    x = F.interpolate(x[:, :, i:i + h, j:j + w], size=(H, W), mode="bilinear", align_corners=False)
    if np.random.rand() < 0.5:
        x = torch.flip(x, [3])
    if np.random.rand() < 0.5:
        x = torch.flip(x, [2])
    k = int(np.random.randint(0, 4))
    if k:
        x = torch.rot90(x, k, [2, 3])
    if np.random.rand() < cj_p:
        b = torch.empty(B, 1, 1, 1, device=dev).uniform_(1 - cj, 1 + cj)
        c = torch.empty(B, 1, 1, 1, device=dev).uniform_(1 - cj, 1 + cj)
        s = torch.empty(B, 1, 1, 1, device=dev).uniform_(1 - cj, 1 + cj)
        x = x * b
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * c + mean
        gray = x.mean(dim=1, keepdim=True)
        x = (x - gray) * s + gray
        x = x + torch.empty(B, 3, 1, 1, device=dev).uniform_(-hue, hue)
    if np.random.rand() < gray_p:
        x = x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    return x.clamp(0, 1)


def he_jitter(x, strength=0.4):
    """Per-example brightness/contrast/saturation U[1-s,1+s] and hue +-s/5 (as v1)."""
    if strength <= 0:
        return x
    B, dev = x.shape[0], x.device
    b = torch.empty(B, 1, 1, 1, device=dev).uniform_(1 - strength, 1 + strength)
    c = torch.empty(B, 1, 1, 1, device=dev).uniform_(1 - strength, 1 + strength)
    s = torch.empty(B, 1, 1, 1, device=dev).uniform_(1 - strength, 1 + strength)
    hue = torch.empty(B, 3, 1, 1, device=dev).uniform_(-strength / 5, strength / 5)
    x = x * b
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * c + m
    g = x.mean(dim=1, keepdim=True)
    x = (x - g) * s + g
    return (x + hue).clamp(0, 1)


# ============================================================================ objectives
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al., ICLR 2021)."""

    def __init__(self, params, base_optimizer_cls, rho=0.05, **kw):
        defaults = dict(rho=rho, **kw)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kw)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self):
        gn = torch.norm(torch.stack([p.grad.norm(2) for g in self.param_groups for p in g["params"]
                                     if p.grad is not None]), 2)
        for g in self.param_groups:
            scale = g["rho"] / (gn + 1e-12)
            for p in g["params"]:
                if p.grad is None:
                    continue
                e = p.grad * scale.to(p)
                p.add_(e)
                self.state[p]["e_w"] = e
        self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if "e_w" in self.state[p]:
                    p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        self.zero_grad(set_to_none=True)


def set_bn_momentum(model, on: bool):
    for m in bn_modules(model):
        if not on:
            m._saved_momentum = m.momentum
            m.momentum = 0.0
        elif hasattr(m, "_saved_momentum"):
            m.momentum = m._saved_momentum


def irm_penalty(logits, y):
    """IRMv1 penalty, DomainBed's unbiased form (product of the dummy-scale gradients
    of two disjoint halves)."""
    scale = torch.ones((), device=logits.device, requires_grad=True)
    l1 = F.cross_entropy(logits[::2] * scale, y[::2])
    l2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
    g1 = torch.autograd.grad(l1, [scale], create_graph=True)[0]
    g2 = torch.autograd.grad(l2, [scale], create_graph=True)[0]
    return (g1 * g2).sum()


def irm_weight(step: int, anneal_steps: int, lam: float) -> float:
    """DomainBed schedule: 1.0 for the first `anneal_steps` updates, then lambda."""
    return float(lam) if step >= anneal_steps else 1.0


def coral_penalty(f, d, K):
    """DomainBed CORAL: mean over training-domain pairs of
    mean((mu_i-mu_j)^2) + mean((Cov_i-Cov_j)^2)."""
    f = f.float()
    stats = []
    for k in range(K):
        fk = f[d == k]
        if fk.shape[0] < 2:
            continue
        mu = fk.mean(0, keepdim=True)
        c = fk - mu
        stats.append((mu, c.T @ c / (fk.shape[0] - 1)))
    if len(stats) < 2:
        return f.new_zeros(())
    pen, n = f.new_zeros(()), 0
    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            pen = pen + (stats[i][0] - stats[j][0]).pow(2).mean() + (stats[i][1] - stats[j][1]).pow(2).mean()
            n += 1
    return pen / n


def groupdro_loss(per, d, q, eta, K):
    """Sagawa et al. 2020 / WILDS: q_k <- q_k exp(eta L_k), loss = sum_k q_k L_k.
    Groups are the contiguous training-domain indices 0..K-1 (fix D6)."""
    cnt = torch.bincount(d, minlength=K).float()
    Lk = torch.zeros(K, device=per.device).scatter_add_(0, d, per.float()) / cnt.clamp_min(1)
    present = (cnt > 0).float()
    with torch.no_grad():
        q.mul_(torch.exp(eta * Lk.detach() * present))
        q.div_(q.sum())
    return (q * Lk).sum()


def lisa_mix(x, y, d, alpha=2.0, p_sel=0.5, n_classes=2, intra_label=None):
    """LISA (Yao et al., ICML 2022) selective augmentation, vectorised partner search.
    With probability p_sel (per batch) intra-label: partner has the same label and a
    different domain; otherwise intra-domain: same domain, different label (labels mixed).
    Rows without a valid partner fall back to same-label (intra-label) / different-label
    (intra-domain) partners from any domain, else to themselves."""
    B = y.shape[0]
    if intra_label is None:
        intra_label = bool(np.random.rand() < p_sel)
    same_y = y[:, None] == y[None, :]
    same_d = d[:, None] == d[None, :]
    eye = torch.eye(B, dtype=torch.bool, device=y.device)
    if intra_label:
        M, Fb = same_y & ~same_d, same_y & ~eye
    else:
        M, Fb = same_d & ~same_y, ~same_y
    R = torch.rand(B, B, device=y.device)
    p1 = torch.where(M, R, torch.full_like(R, -1.0)).argmax(1)
    p2 = torch.where(Fb, R, torch.full_like(R, -1.0)).argmax(1)
    ar = torch.arange(B, device=y.device)
    partner = torch.where(M.any(1), p1, torch.where(Fb.any(1), p2, ar))
    lam = torch.distributions.Beta(alpha, alpha).sample((B,)).to(x.device)
    xm = lam.view(B, 1, 1, 1) * x + (1 - lam.view(B, 1, 1, 1)) * x[partner]
    yo = F.one_hot(y, n_classes).float()
    ym = lam.view(B, 1) * yo + (1 - lam.view(B, 1)) * yo[partner]
    return xm, ym, partner, intra_label


class FishState:
    """Fish (Shi et al., ICLR 2022), DomainBed/WILDS-style implementation (fix D7):
    an inner model is reset to the current weights each meta step, takes one SGD step
    per training domain present in the batch (random order) with a *persistent* inner
    optimizer (momentum + weight decay, scheduled lr), then
        theta <- theta + meta_lr * (theta_inner - theta)    (parameters only)
        buffers <- buffers_inner                              (BN statistics copied)."""

    def __init__(self, model, make_opt, meta_lr):
        self.inner = copy.deepcopy(model)
        self.opt = make_opt([p for p in self.inner.parameters() if p.requires_grad])
        self.meta_lr = float(meta_lr)

    def step(self, model, x, y, d, autocast_ctx, min_per_domain=2):
        mp, ip = list(model.parameters()), list(self.inner.parameters())
        mb, ib = list(model.buffers()), list(self.inner.buffers())
        with torch.no_grad():
            torch._foreach_copy_(ip, mp)
            torch._foreach_copy_(ib, mb)
        self.inner.train()
        cnt = torch.bincount(d).tolist()                      # one host sync per meta step
        doms = [k for k in np.random.permutation(len(cnt)) if cnt[k] >= min_per_domain]
        tot = torch.zeros((), device=x.device)
        for k in doms:
            idx = (d == k).nonzero(as_tuple=True)[0]
            xk = x.index_select(0, idx)
            if x.is_contiguous(memory_format=torch.channels_last):
                xk = xk.contiguous(memory_format=torch.channels_last)
            with autocast_ctx():
                loss = F.cross_entropy(self.inner(xk).float(), y.index_select(0, idx))
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            tot = tot + loss.detach()
        with torch.no_grad():
            diff = torch._foreach_sub(ip, mp)
            torch._foreach_add_(mp, diff, alpha=self.meta_lr)   # params: theta + meta_lr (inner - theta)
            torch._foreach_copy_(mb, ib)                         # buffers: copied from the inner model
        return tot / max(len(doms), 1)


# ============================================================================ trainer
class Trainer:
    def __init__(self, method, model, hp, K, n_steps, device):
        self.method, self.model, self.hp, self.K, self.dev = method, model, hp, K, device
        self.amp = bool(hp["amp"]) and torch.device(device).type == "cuda"
        self.cl = bool(hp["channels_last"])
        warm = int(round(hp["warmup_frac"] * n_steps))
        self.n_steps = n_steps

        def make_opt(params):
            if hp["opt"] == "adamw":
                return torch.optim.AdamW(params, lr=hp["lr"], weight_decay=hp["wd"])
            return torch.optim.SGD(params, lr=hp["lr"], momentum=hp["momentum"], weight_decay=hp["wd"])

        params = [p for p in model.parameters() if p.requires_grad]
        if method == "fish":
            self.fish = FishState(model, make_opt, hp["meta_lr"])
            self.opt = self.base = self.fish.opt
        elif method in USES_SAM:
            assert hp["opt"] == "sgd"
            self.opt = SAM(params, torch.optim.SGD, rho=hp["rho"], lr=hp["lr"], momentum=hp["momentum"],
                           weight_decay=hp["wd"])
            self.base = self.opt.base_optimizer
        else:
            self.opt = self.base = make_opt(params)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.base, cosine_lambda(n_steps, warm))
        if method == "groupdro":
            self.q = torch.full((K,), 1.0 / K, device=device)
        if method == "irm":
            self.irm_anneal = int(round(hp["irm_anneal_frac"] * n_steps))
        self.it = 0

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp)

    def augment(self, x):
        hp = self.hp
        if hp["aug"] == "dihedral":
            x = sup_augment(x)
        if self.method == "augonly":
            x = simclr_view(x, hp["rrc_min"], hp["cj_p"], hp["cj"], hp["hue"], hp["gray_p"])
        elif self.method in ("he_jitter", "tia_style"):
            x = he_jitter(x, hp["jitter"])
        if self.cl:
            x = x.contiguous(memory_format=torch.channels_last)
        return x

    def _ce_step(self, loss):
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

    def step(self, x, y, d):
        """x: float [0,1] NCHW (un-augmented), y, d: int64. Returns detached loss tensor."""
        m, hp, model = self.method, self.hp, self.model
        model.train()
        x = self.augment(x)
        if m == "fish":
            loss = self.fish.step(model, x, y, d, self.autocast)
        elif m in USES_SAM:
            set_bn_momentum(model, True)
            with self.autocast():
                loss = F.cross_entropy(model(x).float(), y)
            loss.backward()
            self.opt.first_step()
            set_bn_momentum(model, False)          # no BN-statistics update at the perturbed point
            with self.autocast():
                F.cross_entropy(model(x).float(), y).backward()
            self.opt.second_step()
            set_bn_momentum(model, True)
        elif m == "groupdro":
            with self.autocast():
                logits = model(x).float()
            loss = groupdro_loss(F.cross_entropy(logits, y, reduction="none"), d, self.q, hp["eta"], self.K)
            self._ce_step(loss)
        elif m == "irm":
            if self.it == self.irm_anneal and hp["irm_reset_opt"] and self.irm_anneal > 0:
                self.base.state.clear()            # DomainBed resets the optimizer at the anneal step
            w = irm_weight(self.it, self.irm_anneal, hp["irm_lambda"])
            with self.autocast():
                logits = model(x).float()
            nll = F.cross_entropy(logits, y)
            pens = [irm_penalty(logits[d == k], y[d == k]) for k in range(self.K) if int((d == k).sum()) >= 4]
            pen = torch.stack(pens).mean() if pens else logits.new_zeros(())
            loss = nll + w * pen
            self._ce_step(loss)
        elif m == "coral":
            with self.autocast():
                f = model.features(x)
                logits = model.fc(f).float()
            loss = F.cross_entropy(logits, y) + hp["coral_lambda"] * coral_penalty(f, d, self.K)
            self._ce_step(loss)
        elif m == "lisa":
            xm, ym, _, _ = lisa_mix(x, y, d, hp["lisa_alpha"], hp["p_sel"])
            with self.autocast():
                logits = model(xm).float()
            loss = -(ym * F.log_softmax(logits, 1)).sum(1).mean()
            self._ce_step(loss)
        else:  # erm, simclr, augonly, compute_matched, mixstyle, macenko, he_jitter, tia_style
            with self.autocast():
                loss = F.cross_entropy(model(x).float(), y)
            self._ce_step(loss)
        self.sched.step()
        self.it += 1
        return loss.detach()

    def lr(self):
        return self.base.param_groups[0]["lr"]


# ============================================================================ test-time adaptation
def tta_order(n, seed=0):
    """Fixed-seed shuffled test stream (fix D8): never feed a label-sorted stream."""
    return np.random.RandomState(10_000 + seed).permutation(n)


@torch.no_grad()
def tta_bnadapt(model, X_u8, device, bs=128, seed=0, amp=False, cl=False):
    """Re-estimate BN running statistics (cumulative average) on the shuffled test
    stream, then predict in eval mode. Returns probabilities in the original order."""
    from common import predict, to_float
    bns = bn_modules(model)
    assert bns, "bnadapt needs BatchNorm layers"
    order = torch.from_numpy(tta_order(X_u8.shape[0], seed))
    model.eval()
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    for k in range(0, len(order), bs):
        x = to_float(X_u8[order[k:k + bs].to(X_u8.device)], device)
        if cl:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            model(x)
    model.eval()
    return predict(model, X_u8, device, amp=amp, channels_last=cl)


def tta_tent(model, X_u8, device, bs=128, lr=1e-3, steps=1, seed=0, amp=False, cl=False, record=None):
    """TENT (Wang et al., ICLR 2021), online: BN in train mode with batch statistics,
    only BN affine parameters adapted, one Adam step per batch of the shuffled stream;
    the prediction for a batch is the one made before its update (as in the reference
    code). Returns probabilities in the original order."""
    from common import to_float
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for m in bn_modules(model):
        m.train()
        m.track_running_stats = False
        m.running_mean, m.running_var = None, None
        for p in (m.weight, m.bias):
            if p is not None:
                p.requires_grad_(True)
                params.append(p)
    assert params, "tent needs BatchNorm layers"
    opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    n = X_u8.shape[0]
    order = torch.from_numpy(tta_order(n, seed))
    probs = torch.empty(n, dtype=torch.float64)
    for k in range(0, n, bs):
        idx = order[k:k + bs]
        if record is not None:
            record.append(idx.clone())
        x = to_float(X_u8[idx.to(X_u8.device)], device)
        if cl:
            x = x.contiguous(memory_format=torch.channels_last)
        for s in range(steps):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(x).float()
            if s == 0:
                probs[idx] = torch.softmax(logits.detach().double(), 1)[:, 1].cpu()
            p = logits.softmax(1)
            loss = -(p * logits.log_softmax(1)).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return probs.numpy()
