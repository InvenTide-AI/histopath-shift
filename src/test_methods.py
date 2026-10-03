"""Smoke tests for src/methods.py. Runs on CPU in a few seconds."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch, torch.nn as nn
import methods as M
import train_lib as T


def test_groupdro():
    dro = M.GroupDROState(n_groups=3, eta=0.1)
    per = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.05, 0.05])
    s = torch.tensor([0, 0, 1, 1, 2, 2])
    l = dro.loss(per, s)
    assert l.item() > 0
    # q should shift mass toward the highest-loss group
    q0 = dro.q.clone()
    for _ in range(20):
        dro.loss(per, s)
    q1 = dro.q
    # group 1 has the highest mean loss (0.5); its share must have grown
    assert q1[1] > q0[1] > 1.0/3 - 1e-6
    print("groupdro ok  q:", q1.tolist())


def test_coral():
    torch.manual_seed(0)
    # Two hospitals whose feature covariances differ: penalty should be > 0.
    a = torch.randn(64, 8)
    b = torch.randn(64, 8) * 2.0  # different scale => different covariance
    feats = torch.cat([a, b], 0)
    s = torch.cat([torch.zeros(64), torch.ones(64)]).long()
    pen = M.coral_penalty(feats, s)
    # And zero (or very small) when both groups are identically distributed.
    feats2 = torch.cat([a, a.clone()], 0)
    pen2 = M.coral_penalty(feats2, s)
    assert pen.item() > pen2.item() + 1e-3
    print(f"coral ok  pen(different)={pen.item():.3f}  pen(same)={pen2.item():.3f}")


def test_irm():
    torch.manual_seed(0)
    B, C = 32, 2
    logits = torch.randn(B, C, requires_grad=True)
    y = torch.randint(0, C, (B,))
    s = torch.randint(0, 3, (B,))
    pen = M.irm_penalty(logits, y, s)
    assert pen.item() >= 0
    # IRM penalty should differentiate cleanly through `logits`.
    g = torch.autograd.grad(pen, logits)[0]
    assert g.shape == logits.shape
    print(f"irm ok  pen={pen.item():.4f}")


def test_mixstyle():
    torch.manual_seed(0)
    ms = M.MixStyle(p=1.0, alpha=0.5); ms.train()
    x = torch.randn(8, 3, 16, 16)
    y = ms(x)
    assert y.shape == x.shape
    ms.eval()
    y2 = ms(x)
    assert torch.equal(y2, x)  # no-op at eval
    print("mixstyle ok")


def test_macenko():
    rng = np.random.default_rng(0)
    img = rng.integers(80, 240, size=(64, 64, 3), dtype=np.uint8)
    m = M.MacenkoNormalizer().fit(img)
    out = m.transform(img)
    assert out.dtype == np.uint8 and out.shape == img.shape
    assert (out >= 0).all() and (out <= 255).all()
    print(f"macenko ok  out range=[{out.min()},{out.max()}]")


def test_probe():
    backbone = nn.Sequential(nn.Conv2d(3, 8, 3, 1, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    head = M.LinearProbe(feature_dim=8, n_classes=2)
    x = torch.randn(4, 3, 32, 32)
    feats = backbone(x)
    logits = head(feats)
    assert logits.shape == (4, 2)
    print("probe ok")


if __name__ == "__main__":
    test_groupdro()
    test_coral()
    test_irm()
    test_mixstyle()
    test_macenko()
    test_probe()
    print("all smoke tests passed")
