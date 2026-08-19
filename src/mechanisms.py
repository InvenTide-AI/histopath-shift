"""Mechanism suite for cross-site generalization, beyond SSL and SAM.

The earlier experiments tested two mechanisms mapped to two closure conditions:
  condition (i)  -- learn features that survive the appearance change  -> SimCLR
  condition (ii) -- reach a solution whose risk is flat under that change -> SAM

The finding that motivated this extension: the two are super-additive under
shift (positive interaction on every shift axis, both datasets) but each is
individually *costly*, so a net gain requires the interaction to exceed the
summed cost of the mechanisms applied alone. That immediately raises the
question this module exists to answer -- is the super-additivity a property of
SimCLR+SAM specifically, or of the (i)x(ii) PAIRING? If it is the pairing, then
any condition-(i) mechanism should be super-additive with any condition-(ii)
mechanism, and the effect should be absent within a class.

So the suite is organised BY CONDITION, not as a flat list:

  condition (i) -- representation-level invariance to site appearance
    simclr   contrastive pre-training on unlabeled patches (already studied)
    mae      masked autoencoding: a reconstruction objective rather than an
             instance-discrimination one, so it does not rely on the colour
             jitter that arguably hands SimCLR the stain invariance for free
    stainaug supervised stain/colour augmentation in optical-density space:
             the direct, cheap intervention that targets the same invariance
             without any pre-training at all -- the control that tells us
             whether SSL is doing anything a good augmentation cannot
    dann     adversarial domain-adversarial training on the site label: makes
             invariance an explicit objective rather than a hoped-for
             by-product (uses TRAIN-site labels only; the held-out hospital is
             never seen, so this stays a legitimate cross-site method)

  condition (ii) -- flatness of the achieved solution
    sam      sharpness-aware minimization (already studied)
    swa      stochastic weight averaging: reaches a flat region by averaging
             iterates rather than by minimising a worst-case perturbation.
             A second condition-(ii) mechanism with a completely different
             mechanism of action, which is what makes the pairing test possible

Composability is the whole point: any subset may be enabled at once, so the
2x2 factorial structure the interaction decomposition needs is preserved for
every (i)x(ii) pair, not just SimCLR x SAM.

Everything here is deliberately implemented against the primitives already in
train_lib.py (SAM, nt_xent, the augmentation stack) rather than re-deriving
them, so the mechanisms studied in the published runs are bit-identical to the
ones studied here.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Condition (i): stain augmentation in optical-density space
# ---------------------------------------------------------------------------
# H&E absorbance is roughly linear in optical density, OD = -log10(I/I0), so a
# stain-strength change is close to a per-channel affine map in OD space. That
# makes this a physically-motivated augmentation rather than generic colour
# jitter, which matters because the whole question is invariance to a *stain*
# change.

def rgb_to_od(x, eps=1e-6):
    """x in [0,1], NCHW -> optical density."""
    return -torch.log10(x.clamp_min(eps))


def od_to_rgb(od):
    return torch.pow(10.0, -od).clamp(0.0, 1.0)


def stain_augment(x, sigma=0.25, bias=0.05, generator=None):
    """Per-image, per-channel affine jitter in OD space (Tellez et al. style).

    sigma: multiplicative scale, drawn ~ U(1-sigma, 1+sigma)
    bias:  additive offset, drawn ~ U(-bias, bias)
    """
    n, c = x.shape[0], x.shape[1]
    dev = x.device
    a = 1.0 + (torch.rand(n, c, 1, 1, device=dev, generator=generator) * 2 - 1) * sigma
    b = (torch.rand(n, c, 1, 1, device=dev, generator=generator) * 2 - 1) * bias
    return od_to_rgb(rgb_to_od(x) * a + b)


# ---------------------------------------------------------------------------
# Condition (i): masked autoencoding
# ---------------------------------------------------------------------------
class MAEDecoder(nn.Module):
    """Lightweight conv decoder for masked-patch reconstruction.

    Deliberately small: the point is to shape the ENCODER, and a heavy decoder
    would both dominate the compute budget and let the decoder compensate for a
    weak encoder, which would confound the comparison against SimCLR.
    """

    def __init__(self, in_dim, out_ch=3, out_hw=64):
        super().__init__()
        self.out_hw = out_hw
        self.fc = nn.Linear(in_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, out_ch, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(z.shape[0], 256, 4, 4)
        out = self.net(h)
        if out.shape[-1] != self.out_hw:
            out = F.interpolate(out, size=(self.out_hw, self.out_hw), mode="bilinear",
                                align_corners=False)
        return out


def random_block_mask(x, patch=8, ratio=0.6, generator=None):
    """Mask a fraction of non-overlapping square blocks.

    Returns (masked_input, mask) with mask 1 where content was REMOVED, so the
    loss can be taken on masked regions only -- reconstructing visible pixels is
    trivial and would dilute the objective.
    """
    n, c, h, w = x.shape
    gh, gw = h // patch, w // patch
    keep = torch.rand(n, 1, gh, gw, device=x.device, generator=generator) >= ratio
    mask = (~keep).float()
    mask_full = F.interpolate(mask, size=(h, w), mode="nearest")
    return x * (1.0 - mask_full), mask_full


def mae_loss(recon, target, mask_full):
    """MSE on masked pixels only."""
    denom = mask_full.sum().clamp_min(1.0) * target.shape[1]
    return (((recon - target) ** 2) * mask_full).sum() / denom


# ---------------------------------------------------------------------------
# Condition (i): domain-adversarial training (DANN)
# ---------------------------------------------------------------------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class DomainHead(nn.Module):
    """Predicts which TRAINING site a patch came from, through a reversed
    gradient, so the encoder is pushed toward site-invariant features.

    Uses only the sites present in training. The held-out hospital is never
    supplied, so this remains a valid leave-one-hospital-out method (it is
    domain generalization, not domain adaptation).
    """

    def __init__(self, in_dim, n_domains, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(True),
                                 nn.Linear(hidden, n_domains))

    def forward(self, z, lambd=1.0):
        return self.net(grad_reverse(z, lambd))


def dann_lambda(step, total_steps, max_lambda=1.0):
    """Ramp the adversarial weight up from 0 (Ganin et al.).

    Early in training the domain classifier is uninformative, so a full-strength
    reversed gradient injects noise into the encoder; the ramp is what makes
    DANN trainable in practice rather than an optional refinement.
    """
    p = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return float(max_lambda * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0))


# ---------------------------------------------------------------------------
# Condition (ii): stochastic weight averaging
# ---------------------------------------------------------------------------
class SWA:
    """Equal-weight running average of parameters, collected late in training.

    Distinct from SAM in mechanism: SAM minimises a worst-case perturbed loss
    during optimisation; SWA does nothing to the objective and instead averages
    iterates that wander around a basin. If flatness is what matters for
    cross-site transfer, both should help and neither should be super-additive
    with the other -- which is the falsifiable prediction that makes including
    SWA worthwhile rather than merely additional.

    BN running statistics are NOT an average of the models' statistics, so they
    must be recomputed with a forward pass over training data before evaluation
    (`update_bn`). Skipping that step is the classic SWA bug and silently
    degrades accuracy.
    """

    def __init__(self, model, start_frac=0.6):
        self.start_frac = start_frac
        self.n = 0
        self.avg = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
        self.active = False

    def maybe_update(self, model, epoch, total_epochs):
        if (epoch + 1) / total_epochs < self.start_frac:
            return False
        sd = model.state_dict()
        if self.n == 0:
            for k in self.avg:
                self.avg[k].copy_(sd[k].detach().float())
        else:
            for k in self.avg:
                if self.avg[k].is_floating_point():
                    self.avg[k].mul_(self.n / (self.n + 1)).add_(
                        sd[k].detach().float() / (self.n + 1))
                else:
                    self.avg[k].copy_(sd[k].detach().float())
        self.n += 1
        self.active = True
        return True

    def load_into(self, model):
        if not self.active:
            return False
        sd = model.state_dict()
        model.load_state_dict({k: self.avg[k].to(sd[k].dtype) for k in sd})
        return True

    @staticmethod
    @torch.no_grad()
    def update_bn(model, batches, device="cpu"):
        """Recompute BN running stats for the averaged weights."""
        mods = [m for m in model.modules()
                if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        if not mods:
            return 0
        for m in mods:
            m.reset_running_stats()
            m.momentum = None          # cumulative moving average
        model.train()
        seen = 0
        for xb in batches:
            model(xb.to(device))
            seen += xb.shape[0]
        return seen


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
CONDITION_I = ("simclr", "mae", "stainaug", "dann")
CONDITION_II = ("sam", "swa")
ALL_MECHANISMS = CONDITION_I + CONDITION_II


def parse_mechanisms(spec):
    """'simclr+sam' -> {'simclr','sam'}; 'baseline'/'' -> set()."""
    if not spec or spec in ("baseline", "erm", "none"):
        return set()
    out = set()
    for tok in str(spec).replace(",", "+").split("+"):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok not in ALL_MECHANISMS:
            raise ValueError(f"unknown mechanism {tok!r}; known: {ALL_MECHANISMS}")
        out.add(tok)
    return out


def arm_name(mechs):
    """Canonical, order-independent arm name."""
    if not mechs:
        return "baseline"
    return "+".join(m for m in ALL_MECHANISMS if m in mechs)


def factorial_pairs():
    """The (i)x(ii) pairs whose 2x2 factorial supports an interaction estimate."""
    return [(a, b) for a in CONDITION_I for b in CONDITION_II]
