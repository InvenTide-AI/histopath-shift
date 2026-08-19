"""Training library for the Camelyon17-WILDS two-pillar ablation.

Pillar I  : self-supervised pre-training (SimCLR / NT-Xent)  -> "SSL"
Pillar II : sharpness-aware minimization (Foret et al. 2021) -> "SAM"
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F


# ---------------------------------------------------------------- encoder
def conv_bn(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Encoder(nn.Module):
    """Compact 4-stage CNN (~0.58M params), sized for CPU training at 64px."""
    def __init__(self, w=(32, 64, 128, 256)):
        super().__init__()
        a, b, c, d = w
        self.features = nn.Sequential(
            conv_bn(3, a), conv_bn(a, a), nn.MaxPool2d(2),
            conv_bn(a, b), conv_bn(b, b), nn.MaxPool2d(2),
            conv_bn(b, c), conv_bn(c, c), nn.MaxPool2d(2),
            conv_bn(c, d), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.out_dim = d

    def forward(self, x):
        return self.features(x)


class Classifier(nn.Module):
    def __init__(self, encoder=None, n_classes=2):
        super().__init__()
        self.encoder = encoder if encoder is not None else Encoder()
        self.fc = nn.Linear(self.encoder.out_dim, n_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


class ProjectionHead(nn.Module):
    """2-layer MLP projection head (SimCLR)."""
    def __init__(self, in_dim, hidden=256, out=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------- Pillar II: SAM
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al., ICLR 2021).

    Two-step update per batch:
      1. e(w) = rho * g / ||g||_2      (ascent to the local worst case)
      2. w <- w - lr * grad L(w + e(w)) (descent computed at the perturbed point)
    """
    def __init__(self, params, base_optimizer_cls, rho=0.05, **kw):
        assert rho >= 0
        defaults = dict(rho=rho, **kw)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kw)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    def _grad_norm(self):
        return torch.norm(torch.stack([
            p.grad.norm(p=2) for g in self.param_groups
            for p in g["params"] if p.grad is not None]), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        gn = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (gn + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)                        # climb to w + e(w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])       # back to w
        self.base_optimizer.step()                 # descent with the perturbed grad
        if zero_grad:
            self.zero_grad()


def sam_step(model, x, y, opt, loss_fn):
    """One SAM update. BN stats are frozen on the ascent pass so the
    perturbation does not double-update running statistics."""
    enable_running_stats(model, False)
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.first_step(zero_grad=True)
    enable_running_stats(model, True)
    loss_fn(model(x), y).backward()
    opt.second_step(zero_grad=True)
    return loss.item()


def enable_running_stats(model, enable):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if not enable:
                m._saved_momentum = m.momentum
                m.momentum = 0.0
            elif hasattr(m, "_saved_momentum"):
                m.momentum = m._saved_momentum


# ---------------------------------------------------------------- Pillar I: SimCLR
def nt_xent(z1, z2, temperature=0.5):
    """Normalized temperature-scaled cross-entropy (SimCLR).
    2N views; positives are (i, i+N). Self-similarities masked out."""
    N = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], 0), dim=1)
    sim = (z @ z.T) / temperature
    sim.fill_diagonal_(-1e9)
    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, targets)


# ---- augmentations (uint8 NHWC tensor batches -> float NCHW), CPU-cheap ----
def _rand_resized_crop(x, scale=(0.4, 1.0)):
    """x: (B,3,H,W) float. Per-batch random resized crop."""
    B, C, H, W = x.shape
    s = float(np.random.uniform(*scale))
    h = max(8, int(round(H * np.sqrt(s))))
    w = max(8, int(round(W * np.sqrt(s))))
    i = np.random.randint(0, H - h + 1)
    j = np.random.randint(0, W - w + 1)
    return F.interpolate(x[:, :, i:i + h, j:j + w], size=(H, W),
                         mode="bilinear", align_corners=False)


def simclr_view(x):
    """Histopathology-appropriate SimCLR view. Colour jitter + grayscale are
    the operations that target the stain/scanner-intensity acquisition
    shortcut the position paper identifies (Sec 2.1)."""
    x = _rand_resized_crop(x)
    if np.random.rand() < 0.5:
        x = torch.flip(x, [3])
    if np.random.rand() < 0.5:
        x = torch.flip(x, [2])
    k = int(np.random.randint(0, 4))
    if k:
        x = torch.rot90(x, k, [2, 3])
    if np.random.rand() < 0.8:                       # colour jitter
        B = x.shape[0]
        b = torch.empty(B, 1, 1, 1).uniform_(0.6, 1.4)
        c = torch.empty(B, 1, 1, 1).uniform_(0.6, 1.4)
        s = torch.empty(B, 1, 1, 1).uniform_(0.6, 1.4)
        x = x * b
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * c + mean
        gray = x.mean(dim=1, keepdim=True)
        x = (x - gray) * s + gray
        hue = torch.empty(B, 3, 1, 1).uniform_(-0.08, 0.08)
        x = x + hue
    if np.random.rand() < 0.2:                       # grayscale
        x = x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    return x.clamp(0, 1)


def sup_augment(x):
    """Light augmentation shared by ALL supervised arms (kept identical so the
    only varying factor between arms is the intervention under test)."""
    if np.random.rand() < 0.5:
        x = torch.flip(x, [3])
    if np.random.rand() < 0.5:
        x = torch.flip(x, [2])
    k = int(np.random.randint(0, 4))
    if k:
        x = torch.rot90(x, k, [2, 3])
    return x


# ---------------------------------------------------------------- data
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_float(batch_u8):
    """(B,H,W,3) uint8 -> (B,3,H,W) float in [0,1]."""
    return batch_u8.permute(0, 3, 1, 2).float().div_(255.0)


def normalize(x):
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def iterate(X, y, bs, shuffle=True, rng=None):
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        (rng or np.random).shuffle(idx)
    for s in range(0, n - (bs - 1 if shuffle else 0), bs):
        b = idx[s:s + bs]
        yield torch.from_numpy(X[b]), (None if y is None else torch.from_numpy(y[b]))


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def predict(model, X, bs=512):
    model.eval()
    out = []
    for xb, _ in iterate(X, None, bs, shuffle=False):
        out.append(torch.softmax(model(normalize(to_float(xb))), 1)[:, 1].numpy())
    return np.concatenate(out)


def compute_metrics(y_true, p):
    from sklearn.metrics import (accuracy_score, roc_auc_score,
                                 average_precision_score, f1_score,
                                 balanced_accuracy_score, confusion_matrix,
                                 log_loss)
    yhat = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    m = dict(
        accuracy=accuracy_score(y_true, yhat),
        auroc=roc_auc_score(y_true, p),
        avg_precision=average_precision_score(y_true, p),
        f1=f1_score(y_true, yhat, zero_division=0),
        balanced_accuracy=balanced_accuracy_score(y_true, yhat),
        sensitivity=tp / max(tp + fn, 1),
        specificity=tn / max(tn + fp, 1),
        precision=tp / max(tp + fp, 1),
        log_loss=log_loss(y_true, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )
    # expected calibration error (15 equal-width bins)
    bins = np.linspace(0, 1, 16)
    conf = np.where(yhat == 1, p, 1 - p)
    correct = (yhat == y_true).astype(float)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        msk = (conf > lo) & (conf <= hi)
        if msk.sum():
            ece += msk.mean() * abs(correct[msk].mean() - conf[msk].mean())
    m["ece"] = float(ece)
    return m
