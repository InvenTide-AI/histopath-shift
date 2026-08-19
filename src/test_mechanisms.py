"""Unit tests for mechanisms.py.

Each mechanism must demonstrably change the part of the training step it claims
to change. These are cheap (CPU, seconds) and exist because a mechanism that
silently no-ops would produce a null result indistinguishable from a real one --
the most expensive possible failure in a sweep this size.
"""
import numpy as np
import torch
import torch.nn as nn

import mechanisms as M


def test_od_roundtrip():
    x = torch.rand(4, 3, 16, 16).clamp(0.02, 1.0)
    back = M.od_to_rgb(M.rgb_to_od(x))
    assert torch.allclose(x, back, atol=1e-4), "OD transform is not invertible"


def test_stain_augment_changes_colour_not_geometry():
    torch.manual_seed(0)
    x = torch.rand(8, 3, 32, 32).clamp(0.1, 0.9)
    y = M.stain_augment(x, sigma=0.3, bias=0.08)
    assert y.shape == x.shape
    assert not torch.allclose(x, y), "stain_augment did nothing"
    # per-channel means should move; spatial structure should be preserved, so
    # per-image channel correlation with the original stays high
    for i in range(x.shape[0]):
        for c in range(3):
            a, b = x[i, c].flatten(), y[i, c].flatten()
            r = np.corrcoef(a.numpy(), b.numpy())[0, 1]
            assert r > 0.9, f"structure destroyed (r={r:.3f}) -- not a stain jitter"
    dmean = (y.mean((2, 3)) - x.mean((2, 3))).abs().mean().item()
    assert dmean > 1e-3, "channel means unchanged"


def test_mask_ratio_and_loss_is_masked_only():
    torch.manual_seed(0)
    x = torch.rand(16, 3, 64, 64)
    xm, mask = M.random_block_mask(x, patch=8, ratio=0.6)
    frac = mask.mean().item()
    assert 0.45 < frac < 0.75, f"mask fraction {frac:.2f} far from requested 0.6"
    assert torch.allclose(xm[mask.expand_as(x) > 0], torch.zeros(1)), "masked pixels not zeroed"
    # loss must ignore visible pixels: perturbing them changes nothing
    recon = torch.rand_like(x)
    l1 = M.mae_loss(recon, x, mask)
    recon2 = recon.clone()
    recon2[mask.expand_as(x) == 0] += 5.0
    l2 = M.mae_loss(recon2, x, mask)
    assert torch.allclose(l1, l2), "mae_loss is not restricted to masked pixels"


def test_grad_reverse_flips_sign():
    z = torch.randn(4, 8, requires_grad=True)
    out = M.grad_reverse(z, 2.0).sum()
    out.backward()
    assert torch.allclose(z.grad, -2.0 * torch.ones_like(z)), "gradient not reversed/scaled"


def test_dann_head_pushes_encoder_toward_invariance():
    """With gradient reversal, the encoder gradient must oppose domain
    discriminability. Check the sign relationship directly."""
    torch.manual_seed(0)
    enc = nn.Linear(6, 8)
    head = M.DomainHead(8, n_domains=3, hidden=16)
    x = torch.randn(32, 6)
    d = torch.randint(0, 3, (32,))
    z = enc(x)
    loss = nn.functional.cross_entropy(head(z, lambd=1.0), d)
    ge = torch.autograd.grad(loss, enc.weight, retain_graph=True)[0]
    # same loss without reversal
    z2 = enc(x)
    loss2 = nn.functional.cross_entropy(head.net(z2), d)
    ge2 = torch.autograd.grad(loss2, enc.weight)[0]
    cos = (ge.flatten() @ ge2.flatten()) / (ge.norm() * ge2.norm() + 1e-12)
    assert cos < -0.99, f"encoder gradient not reversed (cos={cos:.3f})"


def test_dann_lambda_ramp():
    assert M.dann_lambda(0, 100) == 0.0
    mid, end = M.dann_lambda(50, 100), M.dann_lambda(100, 100)
    assert 0 < mid < end <= 1.0, f"ramp not monotone: {mid}, {end}"


def test_swa_averages_and_starts_late():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    swa = M.SWA(m, start_frac=0.6)
    total = 5
    # early epochs must not be collected
    assert not swa.maybe_update(m, 0, total), "SWA collected before start_frac"
    assert not swa.maybe_update(m, 1, total)
    snaps = []
    for ep in (3, 4):
        with torch.no_grad():
            m[0].weight.add_(1.0)
        snaps.append(m[0].weight.detach().clone())
        assert swa.maybe_update(m, ep, total), "SWA did not collect late epoch"
    expect = torch.stack(snaps).mean(0)
    swa.load_into(m)
    assert torch.allclose(m[0].weight, expect, atol=1e-6), "SWA average incorrect"


def test_swa_update_bn_recomputes_stats():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4))
    m.train()
    for _ in range(3):
        m(torch.rand(8, 3, 8, 8))
    before = m[1].running_mean.detach().clone()
    batches = [torch.rand(8, 3, 8, 8) + 10.0 for _ in range(3)]  # very different data
    seen = M.SWA.update_bn(m, batches)
    assert seen == 24, f"update_bn saw {seen} samples"
    assert not torch.allclose(before, m[1].running_mean), "BN stats not recomputed"


def test_parse_and_naming_is_order_independent():
    assert M.parse_mechanisms("baseline") == set()
    assert M.parse_mechanisms("") == set()
    assert M.parse_mechanisms("simclr+sam") == {"simclr", "sam"}
    assert M.parse_mechanisms("sam,simclr") == {"simclr", "sam"}
    assert M.arm_name({"sam", "simclr"}) == M.arm_name({"simclr", "sam"})
    assert M.arm_name(set()) == "baseline"
    try:
        M.parse_mechanisms("nope")
        raise AssertionError("unknown mechanism was accepted")
    except ValueError:
        pass


def test_factorial_pairs_cover_conditions():
    pairs = M.factorial_pairs()
    assert len(pairs) == len(M.CONDITION_I) * len(M.CONDITION_II) == 8
    assert ("simclr", "sam") in pairs, "the already-published pair must be included"
    for a, b in pairs:
        assert a in M.CONDITION_I and b in M.CONDITION_II


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL {name}: {e}")
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    raise SystemExit(1 if fails else 0)
