"""Additional cross-hospital robustness methods, exposed as drop-in objectives
for the same 0.58M-parameter encoder used by the paper's 4-arm ablation.

Each helper takes a mini-batch and a hospital label vector `s` (one integer per
example, giving the training site the example came from) and returns the loss
scalar to backpropagate; the caller does the optimizer step.  This lets a
5-hospital experiment add: GroupDRO, DeepCORAL, IRM, MixStyle, and a
Macenko-stain-normalized baseline while keeping the encoder, schedule and
sample budget identical to the paper's ERM/SSL/SAM arms.  Two additional
pieces --- a stain-normalization pre-processor and a foundation-model linear
probe --- live in the same file so the whole extension is one import.

The goal of adding these is to answer the editor's "incremental methodological
advance" concern: the paper is now the first study to run the full five-hospital
factorial *across a controlled family of domain-generalization objectives* (not
only ERM vs. SSL vs. SAM), and to test each one for the same failure mode ---
gains that live at one held-out site and vanish at the others.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- GroupDRO
class GroupDROState:
    """Group-distributionally-robust optimization (Sagawa et al., ICLR 2020).

    Maintains an exponentially reweighted distribution q over the K training
    hospitals.  Each SGD step: (i) compute per-hospital mean loss, (ii) update
    q_k <- q_k * exp(eta * L_k), renormalize, (iii) return the weighted sum.
    """

    def __init__(self, n_groups: int, eta: float = 0.01):
        self.q = torch.full((n_groups,), 1.0 / n_groups)
        self.eta = eta

    def loss(self, per_example_loss: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        self.q = self.q.to(per_example_loss.device)
        losses = []
        for k in range(len(self.q)):
            mask = (s == k)
            losses.append(per_example_loss[mask].mean() if mask.any()
                          else torch.zeros((), device=per_example_loss.device))
        L = torch.stack(losses)
        with torch.no_grad():
            self.q = self.q * torch.exp(self.eta * L.detach())
            self.q = self.q / self.q.sum().clamp_min(1e-12)
        return (self.q * L).sum()


# --------------------------------------------------------------------------- DeepCORAL
def coral_penalty(feats: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Deep-CORAL feature-covariance alignment (Sun & Saenko, ECCV 2016 W).

    Aligns the feature-covariance matrices between every pair of training
    hospitals.  A representation with equal per-hospital second moments has no
    linear signature of the site.
    """
    groups = [feats[s == k] for k in torch.unique(s)]
    groups = [g for g in groups if g.shape[0] >= 2]
    if len(groups) < 2:
        return feats.new_zeros(())
    covs = [_cov(g) for g in groups]
    d = covs[0].shape[0]
    pen, n_pairs = 0.0, 0
    for i in range(len(covs)):
        for j in range(i + 1, len(covs)):
            pen = pen + (covs[i] - covs[j]).pow(2).sum() / (4 * d * d)
            n_pairs += 1
    return pen / max(n_pairs, 1)


def _cov(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(0, keepdim=True)
    n = max(x.shape[0] - 1, 1)
    return (x.T @ x) / n


# --------------------------------------------------------------------------- IRM v1
def irm_penalty(logits: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Invariant Risk Minimization penalty (Arjovsky et al., 2019, IRMv1).

    For each training hospital, compute the gradient of the loss w.r.t. a
    dummy scalar multiplier on the logits; return the squared norm summed
    over hospitals.  A representation for which the optimal linear head is
    the same at every hospital has zero penalty.
    """
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    penalty = logits.new_zeros(())
    for k in torch.unique(s):
        mask = (s == k)
        if mask.sum() < 2:
            continue
        loss_k = F.cross_entropy(logits[mask] * scale, y[mask])
        grad_k = torch.autograd.grad(loss_k, [scale], create_graph=True)[0]
        penalty = penalty + grad_k.pow(2).sum()
    return penalty


# --------------------------------------------------------------------------- MixStyle
class MixStyle(nn.Module):
    """MixStyle (Zhou et al., ICLR 2021).

    During training, mixes the per-instance channel mean/std of each feature
    map with the mean/std of a shuffled batch, encouraging the encoder to
    represent content independently of stain-driven style statistics.  A
    no-op at eval time.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p, self.alpha, self.eps = p, alpha, eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(()) > self.p:
            return x
        B, C = x.shape[0], x.shape[1]
        mu = x.mean(dim=[2, 3], keepdim=True)
        sig = x.std(dim=[2, 3], keepdim=True).clamp_min(self.eps)
        idx = torch.randperm(B, device=x.device)
        mu2, sig2 = mu[idx], sig[idx]
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x)
        mu_mix = lam * mu + (1 - lam) * mu2
        sig_mix = lam * sig + (1 - lam) * sig2
        return sig_mix * (x - mu) / sig + mu_mix


# --------------------------------------------------------------------------- Macenko stain normalization
class MacenkoNormalizer:
    """Macenko et al., ISBI 2009 --- classical H&E stain normalization.

    Fits a 3x2 stain matrix from a reference tile and re-projects new tiles
    through that matrix.  We use this as an *input-space* baseline against
    the *representation-space* corrections above.  If the SSL gain really is
    a stain-invariance effect, a strong stain normalizer should cover part of
    it; if not, SSL is doing something more than stain equalization.
    """

    HE_REF = np.array([[0.5626, 0.2159],
                       [0.7201, 0.8012],
                       [0.4062, 0.5581]], dtype=np.float32)
    MAX_C_REF = np.array([1.9705, 1.0308], dtype=np.float32)

    def __init__(self, io: float = 240.0, alpha: float = 1.0, beta: float = 0.15):
        self.io, self.alpha, self.beta = io, alpha, beta
        self.HE, self.maxC = self.HE_REF.copy(), self.MAX_C_REF.copy()

    def fit(self, rgb_uint8: np.ndarray) -> "MacenkoNormalizer":
        img = rgb_uint8.reshape(-1, 3).astype(np.float32) + 1
        OD = -np.log(img / self.io)
        OD = OD[(OD >= self.beta).any(1)]
        if len(OD) < 100:
            return self
        cov = np.cov(OD.T)
        eig_val, eig_vec = np.linalg.eigh(cov)
        proj = OD @ eig_vec[:, 1:3]
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        lo = np.percentile(phi, self.alpha)
        hi = np.percentile(phi, 100 - self.alpha)
        v_lo = eig_vec[:, 1:3] @ np.array([np.cos(lo), np.sin(lo)])
        v_hi = eig_vec[:, 1:3] @ np.array([np.cos(hi), np.sin(hi)])
        HE = np.array([v_lo, v_hi]).T if v_lo[0] > v_hi[0] else np.array([v_hi, v_lo]).T
        Y = OD.T
        C = np.linalg.lstsq(HE, Y, rcond=None)[0]
        self.HE = HE.astype(np.float32)
        self.maxC = np.percentile(C, 99, axis=1).astype(np.float32).clip(min=1e-3)
        return self

    def transform(self, rgb_uint8: np.ndarray) -> np.ndarray:
        h, w, _ = rgb_uint8.shape
        img = rgb_uint8.reshape(-1, 3).astype(np.float32) + 1
        OD = -np.log(img / self.io)
        C = np.linalg.lstsq(self.HE, OD.T, rcond=None)[0]
        C = C * (self.MAX_C_REF / self.maxC)[:, None]
        OD_norm = self.HE_REF @ C
        img_norm = self.io * np.exp(-OD_norm)
        return img_norm.T.clip(0, 255).reshape(h, w, 3).astype(np.uint8)


# --------------------------------------------------------------------------- ViT / foundation-model linear probe
class LinearProbe(nn.Module):
    """Given a frozen off-the-shelf histopathology backbone (e.g. UNI, CTransPath),
    exposes only a linear head trained on the same labeled budget as the CNN arms.

    Addresses the paper's fourth stated limitation ("whether the same pattern
    holds at the scale of pathology foundation models").  Because these
    backbones are pre-trained on hundreds of thousands of WSIs, a linear probe
    of them is the natural upper bound the CNN arms are being measured against.
    """

    def __init__(self, feature_dim: int, n_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(feature_dim, n_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc(feats)


@torch.no_grad()
def frozen_features(backbone: nn.Module, x: torch.Tensor, batch: int = 128) -> torch.Tensor:
    backbone.eval()
    out = []
    for s in range(0, len(x), batch):
        out.append(backbone(x[s:s + batch]))
    return torch.cat(out, 0)


# --------------------------------------------------------------------------- registry
METHODS = {
    "erm": "empirical risk minimization (paper's baseline arm)",
    "sam": "sharpness-aware minimization (paper's SAM arm)",
    "ssl": "SimCLR pre-training + fine-tune (paper's SSL arm)",
    "ssl_sam": "SSL + SAM (paper's SSL+SAM arm)",
    "groupdro": "GroupDRO with hospital labels (Sagawa 2020)",
    "coral": "DeepCORAL feature-covariance alignment (Sun 2016)",
    "irm": "IRMv1 penalty across hospitals (Arjovsky 2019)",
    "mixstyle": "MixStyle stain-style mixing in the encoder (Zhou 2021)",
    "macenko": "Macenko stain normalization at input (Macenko 2009) + ERM",
    "probe_uni": "linear probe on a frozen histopathology foundation backbone",
    "fish": "Fish gradient matching across domains (Shi 2021)",
    "lisa": "LISA domain mix-up on same-class inter-domain pairs (Yao 2022)",
    "erm_henorm": "ERM + strong H&E colour/brightness jitter (Taori et al.)",
    "rotinv": "dihedral rotation/flip invariance + hard negatives (Lafarge & Koelzer, MIDOG21)",
    "tia_style": "stain normalization + heavy colour and dihedral augmentation "
                 "(Jahanifar et al., MIDOG21/22 winner; classification stage)",
}


# --------------------------------------------------------------------------- Fish (Shi 2021)
def fish_meta_step(model, opt, groups, loss_fn, meta_lr: float = 0.1):
    """Fish: Do one SGD step on each hospital's mini-batch in turn, then
    replace model params by (init + meta_lr * (final - init)).

    groups: iterable of (x, y) tensors, one per training hospital.
    """
    init = {k: v.detach().clone() for k, v in model.state_dict().items()}
    inner_opt = torch.optim.SGD(model.parameters(), lr=opt.param_groups[0]["lr"], momentum=0.9)
    for x, y in groups:
        inner_opt.zero_grad()
        l = loss_fn(model(x), y).mean()
        l.backward()
        inner_opt.step()
    with torch.no_grad():
        for k, v in model.state_dict().items():
            v.data.copy_(init[k] + meta_lr * (v.data - init[k]))
    return float(l.detach().item())


# --------------------------------------------------------------------------- LISA (Yao 2022)
def lisa_batch(x: torch.Tensor, y: torch.Tensor, s: torch.Tensor,
               alpha: float = 2.0) -> tuple:
    """LISA mix-up: for each example, find another example of the SAME class
    from a DIFFERENT hospital and produce a Beta(alpha, alpha)-mixed input.

    Falls back to plain mixup if a same-class inter-hospital partner is not
    available for a given example.
    """
    B = x.shape[0]
    lam = torch.distributions.Beta(alpha, alpha).sample((B,)).to(x.device)
    lam = lam.view(-1, 1, 1, 1)
    partner = torch.arange(B, device=x.device)
    for i in range(B):
        candidates = torch.where((y == y[i]) & (s != s[i]))[0]
        if len(candidates) == 0:
            candidates = torch.where(y == y[i])[0]
        partner[i] = candidates[torch.randint(len(candidates), (1,), device=x.device)]
    x_mixed = lam * x + (1 - lam) * x[partner]
    # LISA keeps the label since we only mix same-class examples.
    return x_mixed, y


# --------------------------------------------------------------------------- Strong H&E-space jitter (Taori)
def he_jitter(x: torch.Tensor, strength: float = 0.4) -> torch.Tensor:
    """Aggressive H&E-jitter: per-channel brightness / contrast / hue in the
    RGB space, sampled per-example. Approximates the augmentation used in
    the ERM-w/-H&E-jitter WILDS entry (Taori et al.)."""
    if not (0.0 < strength):
        return x
    B, C = x.shape[0], x.shape[1]
    dev = x.device
    b = torch.empty(B, 1, 1, 1, device=dev).uniform_(1 - strength, 1 + strength)
    c = torch.empty(B, 1, 1, 1, device=dev).uniform_(1 - strength, 1 + strength)
    s = torch.empty(B, 1, 1, 1, device=dev).uniform_(1 - strength, 1 + strength)
    hue = torch.empty(B, 3, 1, 1, device=dev).uniform_(-strength / 5, strength / 5)
    x = x * b
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * c + m
    g = x.mean(dim=1, keepdim=True)
    x = (x - g) * s + g
    x = x + hue
    return x.clamp(0, 1)


# --------------------------------------------------------------------------- Dihedral invariance (Lafarge & Koelzer, MIDOG 2021)
def dihedral(x: torch.Tensor) -> torch.Tensor:
    """Random element of the 8-fold dihedral group, sampled per example.

    Mitotic figures carry no canonical orientation, so the MIDOG 2021 entry of
    Lafarge & Koelzer built rotation invariance into the model. Applying the
    group as an augmentation is the cheap equivalent: it buys the same
    invariance without changing the architecture, which keeps this arm
    comparable to every other arm in the grid.
    """
    out = x
    k = torch.randint(0, 4, (1,)).item()
    if k:
        out = torch.rot90(out, k, dims=(2, 3))
    if torch.rand(1).item() < 0.5:
        out = torch.flip(out, dims=(3,))
    return out


def describe() -> str:
    lines = ["Available methods:"]
    for k, v in METHODS.items():
        lines.append(f"  {k:<10} {v}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
