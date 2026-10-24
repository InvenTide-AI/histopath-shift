"""True per-tile Macenko stain normalisation, vectorised in torch (fix D2).

Macenko et al., ISBI 2009. For every tile independently:
  1. OD = -log((I+1)/Io); tissue pixels = all three OD channels >= beta.
  2. Plane of the two leading eigenvectors of the tissue-OD covariance.
  3. Angles of the projected tissue OD; the alpha / (100-alpha) percentiles give
     the two extreme stain directions -> per-tile stain matrix HE. The H/E labelling
     of the two directions is decided by matching to HE_ref (the column permutation
     with the larger summed cosine), not by the 'larger red OD = H' heuristic, which
     is unstable when the two directions have similar red OD.
  4. Concentrations C = max(pinv(HE) @ OD, 0) (all pixels; non-negative, since stain
     amounts are physically >= 0); per-stain 99th percentile maxC.
  5. Normalise: C * maxC_ref / maxC, reconstruct with the reference HE_ref.
A tile with fewer than `min_tissue` tissue pixels keeps HE_ref and maxC_ref
(i.e. it is only projected onto the reference stain plane).

Non-negativity (nonneg=True, the default): per-tile H and E vectors are nearly collinear
(cos 0.95-0.98), so the unconstrained least-squares fit (Macenko's MATLAB `HE\\Y`,
torchstain's lstsq) assigns large negative H to eosin-rich pixels whenever a tile's E
vector is tilted toward H. Rebuilt with HE_ref after the maxC rescale, those pixels become
saturated pink. On Camelyon17 center 4 (fold 4) this artefact is label-correlated (90% of
tumour vs 54% of normal tiles; 2% vs 20% in training) and drove the macenko arm's
off-site AUROC to 0.42 (clamping C >= 0 at test time only: 0.95 with the same weights).
Retrained with C >= 0 it reaches only 0.57 (5 seeds): the remaining gap comes from the
per-tile stain estimate on pale center-4 tiles (27-34% tissue pixels vs 58-83% elsewhere);
reconstructing center 4 with HE_ref instead gives 0.93 with the retrained weights.
nonneg=False reproduces the pre-fix behaviour (records without hparam mac_nonneg).

Reference (mac_ref='train'): elementwise median of the per-tile stain vectors of
`n_ref` fixed training tiles (each tile's H/E labelling aligned to the canonical
H&E pair, then iteratively to the running median), and the median of their per-tile
99th-percentile concentrations. A single Macenko estimate on the pooled pixels of
many tiles is NOT used: pooling tiles with different stain vectors gives a
non-physical 'E' direction (red OD 0.4-0.55 on Camelyon17 instead of ~0.2-0.3) that
differs from fold to fold.

v1 (methods.MacenkoNormalizer.transform) re-used the reference HE for every tile,
which is a single fixed colour transform; this module estimates HE per tile.

Group-level ("per-slide") Macenko (level='group', hparam mac_level='group', the default of
the macenko / tia_style arms; 'macenko_tile' = the per-tile path above, mac_level='tile').
Standard practice estimates the stain matrix once per slide / WSI; per-tile estimates on
small pale tiles are unreliable (c17 center 4, see above). Groups (run_v2.split_groups,
hparam mac_group_unit='slide_domain'): the slide -- c17: WILDS metadata 'slide' at the
rows.npz row ids (the cache g*.npy is the patient; 6/43 patients have 2-3 slides) -- crossed
with the split's domain id where it has one (canine tr/iv: a canine group is a slide imaged
on 3 scanners, pooled it would keep the scanner differences; MIDOG images and c17 slides lie
within one domain); MIDOG: image; PCam: the 'wsi' column. For every split and group,
estimate_group():
  1. orders the group's tiles by a content hash keyed by the group seed (blake2b of the
     tile bytes; seed = crc32 of the group id) and keeps the first `max_tiles` (a
     deterministic pseudo-random subsample that does not depend on the row order);
  2. pools the tissue pixels (OD >= beta in all channels) of those tiles in that order and,
     if there are more than `max_px`, keeps a seeded random subset of `max_px` pixels;
  3. runs the same Macenko estimator (estimate_stains) once on the pooled pixels, labels
     H/E by matching HE_ref (align_to);
  4. maxC_g (maxc='pooled', hparam mac_group_maxc='pooled'): per-stain 99th percentile of the
     concentrations (C >= 0 if nonneg) over ALL pixels of the sampled tiles -- the
     torchstain / StainTools / TIAToolbox statistic ('tile_median': median of the per-tile
     99th percentiles, the first group-level version).
Every tile of the group is then normalised with (HE_g, maxC_g): C = max(pinv(HE_g) OD, 0),
C * maxC_ref / maxC_g, reconstructed with HE_ref. A group with fewer than `min_px` pooled
tissue pixels, no tile with >= min_tissue tissue pixels, or a non-finite estimate keeps
(HE_ref, maxC_ref) (counted as a fallback).
Target (hparam mac_ref='train_group', fit_reference_groups): the SAME group estimator on every
training group, HE_ref = median of the group stain vectors (labelling aligned iteratively),
maxC_ref = median of the group maxC. Source and target then share one estimator (as in
torchstain / StainTools / TIAToolbox, whose fit() and transform() call the same routine), so
a typical training slide is left nearly unchanged; the per-tile reference (mac_ref='train')
has a different maxC scale (pooled H scale 0.7-0.9 vs per-tile 1.6-1.9 on c17 fold 4
training slides, i.e. a ~2x H rescale of every slide). Pooling here is over the pixels of
ONE slide, the standard Macenko input; the 'non-physical E' remark above concerns pooling
tiles of many slides.
No labels are used. The estimate is transductive (each test slide's own tiles) and depends
on the tile makeup of the slide in the cache: c17 caches are class-balanced patches from
annotated regions, so a test slide's estimate is computed from a label-selected tile set,
not from a label-independent tissue sample of the WSI as in deployment.
"""
from __future__ import annotations

import hashlib
import zlib

import numpy as np
import torch

# Canonical reference (as used by the common Macenko implementations / v1).
HE_CANON = torch.tensor([[0.5626, 0.2159],
                         [0.7201, 0.8012],
                         [0.4062, 0.5581]], dtype=torch.float32)
MAXC_CANON = torch.tensor([1.9705, 1.0308], dtype=torch.float32)


def _od(X_u8: torch.Tensor, io: float) -> torch.Tensor:
    """(B,H,W,3) uint8 -> (B,N,3) optical density."""
    B = X_u8.shape[0]
    I = X_u8.reshape(B, -1, 3).float()
    return -torch.log((I + 1.0) / io)


def estimate_stains(OD: torch.Tensor, alpha=1.0, beta=0.15, min_tissue=64):
    """OD (B,N,3) -> HE (B,3,2), ok (B,) bool (enough tissue + finite). Columns are
    provisionally ordered 'larger red OD first'; callers re-label them with align_to."""
    B, N, _ = OD.shape
    mask = (OD >= beta).all(dim=2)                                   # (B,N)
    nt = mask.sum(1)                                                  # (B,)
    mf = mask.float().unsqueeze(2)
    cnt = nt.clamp_min(2).float().view(B, 1, 1)
    mu = (OD * mf).sum(1, keepdim=True) / cnt
    Z = (OD - mu) * mf
    cov = Z.transpose(1, 2) @ Z / (cnt - 1)                           # (B,3,3)
    cov = cov + 1e-8 * torch.eye(3, device=OD.device)
    _, evec = torch.linalg.eigh(cov.double())                          # ascending
    E = evec[:, :, 1:3].float()                                       # (B,3,2)
    sgn = torch.where(E[:, 0:1, :] < 0, -1.0, 1.0)                     # first component positive
    E = E * sgn
    T = OD @ E                                                        # (B,N,2)
    phi = torch.atan2(T[..., 1], T[..., 0])
    phi = torch.where(mask, phi, torch.full_like(phi, float("nan")))
    lo = torch.nanquantile(phi, alpha / 100.0, dim=1)
    hi = torch.nanquantile(phi, 1 - alpha / 100.0, dim=1)
    v_lo = (E @ torch.stack([torch.cos(lo), torch.sin(lo)], 1).unsqueeze(2)).squeeze(2)   # (B,3)
    v_hi = (E @ torch.stack([torch.cos(hi), torch.sin(hi)], 1).unsqueeze(2)).squeeze(2)
    h_first = (v_lo[:, 0] > v_hi[:, 0]).view(B, 1)
    vH = torch.where(h_first, v_lo, v_hi)
    vE = torch.where(h_first, v_hi, v_lo)
    HE = torch.stack([vH, vE], 2)                                     # (B,3,2)
    HE = HE * torch.where(HE.sum(1, keepdim=True) < 0, -1.0, 1.0)
    HE = HE / HE.norm(dim=1, keepdim=True).clamp_min(1e-8)
    ok = (nt >= min_tissue) & torch.isfinite(HE).all(dim=(1, 2))
    return HE, ok


def concentrations(OD: torch.Tensor, HE: torch.Tensor, nonneg: bool = True) -> torch.Tensor:
    """Least-squares stain concentrations (B,2,N), clamped to >= 0 if `nonneg`
    (see the module docstring; nonneg=False is the pre-fix unconstrained fit)."""
    pinv = torch.linalg.pinv(HE)                                      # (B,2,3)
    C = pinv @ OD.transpose(1, 2)
    return C.clamp_min(0.0) if nonneg else C


def _quantile_rows(C: torch.Tensor, q: float, chunk=256):
    out = []
    for k in range(0, C.shape[0], chunk):
        out.append(torch.quantile(C[k:k + chunk], q, dim=2))
    return torch.cat(out, 0)


def _pooled_quantile(C: torch.Tensor, q: float, max_n=1 << 24) -> torch.Tensor:
    """(2,n) -> (2,) q-quantile of each row (torch.quantile needs n <= 2^24; above that a
    fixed evenly spaced subset of the columns is used)."""
    if C.shape[1] > max_n:
        C = C[:, torch.linspace(0, C.shape[1] - 1, max_n, device=C.device).long()]
    return torch.quantile(C, q, dim=1)


def _align_swap(HE: torch.Tensor, HE_ref: torch.Tensor) -> torch.Tensor:
    """(B,) bool: True where swapping the two columns of HE (B,3,2) matches HE_ref better."""
    R = HE_ref.to(HE.device, HE.dtype)
    ident = (HE[:, :, 0] @ R[:, 0]) + (HE[:, :, 1] @ R[:, 1])
    swap = (HE[:, :, 0] @ R[:, 1]) + (HE[:, :, 1] @ R[:, 0])
    return swap > ident


def align_to(HE: torch.Tensor, HE_ref: torch.Tensor) -> torch.Tensor:
    """Swap the two columns of each per-tile HE (B,3,2) where the swapped labelling
    matches HE_ref (3,2) better (larger summed cosine)."""
    R = HE_ref.to(HE.device, HE.dtype)
    ident = (HE[:, :, 0] @ R[:, 0]) + (HE[:, :, 1] @ R[:, 1])
    swap = (HE[:, :, 0] @ R[:, 1]) + (HE[:, :, 1] @ R[:, 0])
    return torch.where((swap > ident).view(-1, 1, 1), HE.flip(2), HE)


def _unit_cols(HE: torch.Tensor) -> torch.Tensor:
    return HE / HE.norm(dim=-2, keepdim=True).clamp_min(1e-8)


def fit_reference(X_u8: torch.Tensor, io=240.0, alpha=1.0, beta=0.15, min_tissue=64, n_iter=3, min_ok=8,
                  nonneg=True):
    """Fit the target (HE_ref, maxC_ref) on a set of reference tiles.
    HE_ref  : elementwise median over the tissue tiles of the per-tile stain vectors,
              each tile's H/E labelling aligned first to HE_CANON and then (n_iter times)
              to the current median; columns renormalised to unit length.
    maxC_ref: median over the same tiles of their per-tile 99th-percentile
              concentrations under their own (aligned) HE, so that a typical training
              tile keeps its intensity after normalisation.
    Falls back to the canonical constants if fewer than `min_ok` tiles have tissue."""
    dev = X_u8.device
    OD = _od(X_u8, io)
    HE, ok = estimate_stains(OD, alpha, beta, min_tissue)
    if int(ok.sum()) < min_ok:
        return HE_CANON.to(dev).clone(), MAXC_CANON.to(dev).clone()
    HE, OD = HE[ok], OD[ok]
    ref = HE_CANON.to(dev)
    for _ in range(max(1, n_iter)):
        HE = align_to(HE, ref)
        ref = _unit_cols(HE.median(0).values)
    HE = align_to(HE, ref)
    ref = _unit_cols(HE.median(0).values)
    maxC = _quantile_rows(concentrations(OD, HE, nonneg), 0.99).median(0).values.clamp_min(1e-3)
    return ref, maxC


@torch.no_grad()
def macenko_normalize(X_u8: torch.Tensor, HE_ref: torch.Tensor, maxC_ref: torch.Tensor,
                      io=240.0, alpha=1.0, beta=0.15, min_tissue=64, chunk=1024,
                      return_he=False, nonneg=True):
    """Per-tile Macenko normalisation of a uint8 NHWC tensor (any device).
    Returns uint8 NHWC tensor on the same device (and optionally the per-tile HE / ok)."""
    dev = X_u8.device
    HE_ref = HE_ref.to(dev).float()
    maxC_ref = maxC_ref.to(dev).float()
    out = torch.empty_like(X_u8)
    hes, oks = [], []
    for k in range(0, X_u8.shape[0], chunk):
        xb = X_u8[k:k + chunk]
        B, H, W, _ = xb.shape
        OD = _od(xb, io)
        HE, ok = estimate_stains(OD, alpha, beta, min_tissue)
        HE = align_to(HE, HE_ref)                                    # H/E labelling w.r.t. the reference
        HE = torch.where(ok.view(B, 1, 1), HE, HE_ref.expand(B, 3, 2))
        C = concentrations(OD, HE, nonneg)                           # (B,2,N), >= 0 if nonneg
        maxC = _quantile_rows(C, 0.99).clamp_min(1e-3)               # (B,2)
        maxC = torch.where(ok.view(B, 1), maxC, maxC_ref.expand(B, 2))
        C = C * (maxC_ref.view(1, 2) / maxC).unsqueeze(2)
        I = io * torch.exp(-(HE_ref.unsqueeze(0) @ C))               # (B,3,N)
        I = I.transpose(1, 2).reshape(B, H, W, 3)
        I = torch.nan_to_num(I, nan=255.0, posinf=255.0, neginf=0.0)
        out[k:k + chunk] = I.clamp(0, 255).round().to(torch.uint8)
        if return_he:
            hes.append(HE)
            oks.append(ok)
    if return_he:
        return out, torch.cat(hes), torch.cat(oks)
    return out


def make_macenko_fn(Xtr_u8: torch.Tensor, ref="train", n_ref=256, io=240.0, alpha=1.0, beta=0.15,
                    min_tissue=64, seed=0, nonneg=True, level="tile", group_tiles=256, group_px=200_000,
                    group_min_px=2048, group_maxc="tile_median", ref_groups=None):
    """Build the split transform used by the macenko / tia_style arms. The target
    (HE_ref, maxC_ref) is fitted on training data only (no test data):
      ref='train'       : per-tile estimator on `n_ref` training tiles (fixed seed; fit_reference),
      ref='train_group' : the group estimator (estimate_group, same settings as fn.group) on every
                          training group `ref_groups` (group id per row of Xtr_u8), median over
                          groups (fit_reference_groups) -- the source and target then use the same
                          estimator, so a typical training group is left (nearly) unchanged;
      ref='canonical'   : the canonical constants.
    fn(X) is always the per-tile transform; fn.group(X, groups) -> (X_norm, stats) is the
    group-level transform (level='group'); fn.level records which one the arm uses;
    fn.ref_info describes the reference fit."""
    if level not in ("tile", "group"):
        raise ValueError(f"mac_level must be 'tile' or 'group', got {level!r}")
    ref_info = dict(ref=ref)
    if ref == "canonical":
        HE_ref, maxC_ref = HE_CANON.clone(), MAXC_CANON.clone()
    elif ref == "train_group":
        if ref_groups is None:
            raise ValueError("mac_ref='train_group' needs the training group ids (ref_groups)")
        HE_ref, maxC_ref, ref_info = fit_reference_groups(Xtr_u8, ref_groups, io, alpha, beta, min_tissue, nonneg,
                                                          group_tiles, group_px, group_min_px, group_maxc)
        if HE_ref is None:   # too few usable training groups: per-tile reference (recorded)
            g = np.random.RandomState(seed)
            idx = torch.from_numpy(g.choice(Xtr_u8.shape[0], size=min(n_ref, Xtr_u8.shape[0]), replace=False))
            HE_ref, maxC_ref = fit_reference(Xtr_u8[idx.to(Xtr_u8.device)], io, alpha, beta, min_tissue,
                                             nonneg=nonneg)
            ref_info["fallback"] = "train"
        ref_info["ref"] = ref
    elif ref == "train":
        g = np.random.RandomState(seed)
        idx = torch.from_numpy(g.choice(Xtr_u8.shape[0], size=min(n_ref, Xtr_u8.shape[0]), replace=False))
        HE_ref, maxC_ref = fit_reference(Xtr_u8[idx.to(Xtr_u8.device)], io, alpha, beta, min_tissue,
                                         nonneg=nonneg)
    else:
        raise ValueError(f"mac_ref must be 'train', 'train_group' or 'canonical', got {ref!r}")

    def fn(X):
        return macenko_normalize(X, HE_ref, maxC_ref, io, alpha, beta, min_tissue, nonneg=nonneg)

    def group_fn(X, groups):
        return macenko_normalize_groups(X, groups, HE_ref, maxC_ref, io, alpha, beta, min_tissue, nonneg=nonneg,
                                        max_tiles=group_tiles, max_px=group_px, min_px=group_min_px,
                                        maxc=group_maxc)
    fn.group, fn.level, fn.ref_info = group_fn, level, ref_info
    fn.HE_ref, fn.maxC_ref = HE_ref.cpu(), maxC_ref.cpu()
    return fn


# ============================================================================ group-level Macenko
def group_seed(gid) -> int:
    """Deterministic 31-bit seed of a group id (same on every device / process)."""
    if isinstance(gid, (np.integer, np.floating)):
        gid = gid.item()
    return zlib.crc32(f"macenko-group:{gid}".encode()) & 0x7FFFFFFF


def _tile_order(X_u8: torch.Tensor, seed: int) -> np.ndarray:
    """Row indices of X_u8 sorted by a seeded content hash of each tile: a pseudo-random
    order that depends only on the set of tiles, not on their row order."""
    A = np.ascontiguousarray(X_u8.cpu().numpy())
    key = int(seed).to_bytes(8, "little")
    h = np.fromiter((int.from_bytes(hashlib.blake2b(A[i].tobytes(), digest_size=8, key=key).digest(), "little")
                     for i in range(A.shape[0])), dtype=np.uint64, count=A.shape[0])
    return np.argsort(h, kind="stable")


@torch.no_grad()
def estimate_group(X_u8: torch.Tensor, HE_ref: torch.Tensor, maxC_ref: torch.Tensor, io=240.0, alpha=1.0,
                   beta=0.15, min_tissue=64, seed=0, max_tiles=256, max_px=200_000, min_px=2048, nonneg=True,
                   maxc="tile_median"):
    """Stain matrix (3,2) and max concentrations (2,) of ONE group (all its tiles, any row
    order), see the module docstring. maxc='pooled': 99th percentile of the concentrations
    over ALL pixels of the sampled tiles (torchstain / StainTools / TIAToolbox statistic);
    'tile_median': median over the sampled tissue tiles of their per-tile 99th percentiles.
    Returns (HE, maxC, info); on fallback HE = HE_ref, maxC = maxC_ref and info['ok'] = False
    with info['reason']."""
    if maxc not in ("pooled", "tile_median"):
        raise ValueError(f"mac_group_maxc must be 'pooled' or 'tile_median', got {maxc!r}")
    dev = X_u8.device
    HE_ref = HE_ref.to(dev).float()
    maxC_ref = maxC_ref.to(dev).float()
    n = int(X_u8.shape[0])
    order = _tile_order(X_u8, seed)[:max_tiles]
    Xs = X_u8[torch.from_numpy(order).to(dev)]
    OD = _od(Xs, io)                                                  # (m,N,3)
    mask = (OD >= beta).all(dim=2)                                    # (m,N)
    T = OD[mask]                                                      # (npx,3), canonical order
    npx = int(T.shape[0])
    info = dict(n_tiles=n, n_used=int(len(order)), n_px=npx, ok=False, reason=None)
    if npx > max_px:
        sel = np.sort(np.random.RandomState(seed).choice(npx, max_px, replace=False))
        T = T[torch.from_numpy(sel).to(dev)]
    info["n_px_used"] = int(T.shape[0])
    if npx < max(min_px, 2):
        info["reason"] = "few_tissue_px"
        return HE_ref.clone(), maxC_ref.clone(), info
    HE, ok = estimate_stains(T.unsqueeze(0), alpha, beta, min_tissue=max(min_px, 2))
    tile_ok = mask.sum(1) >= min_tissue
    if not bool(ok[0]):
        info["reason"] = "nonfinite"
        return HE_ref.clone(), maxC_ref.clone(), info
    if not bool(tile_ok.any()):
        info["reason"] = "no_tissue_tile"
        return HE_ref.clone(), maxC_ref.clone(), info
    HE = align_to(HE, HE_ref)[0]                                      # (3,2), H/E w.r.t. the reference
    m = int(tile_ok.sum())
    if maxc == "pooled":
        C = concentrations(OD.reshape(1, -1, 3), HE.unsqueeze(0), nonneg)[0]   # (2, m_all*N), every pixel
        maxC = _pooled_quantile(C, 0.99).clamp_min(1e-3)                   # (2,)
    else:
        C = concentrations(OD[tile_ok], HE.expand(m, 3, 2), nonneg)      # (m,2,N)
        maxC = _quantile_rows(C, 0.99).median(0).values.clamp_min(1e-3)  # (2,)
    if not bool(torch.isfinite(maxC).all()):
        info["reason"] = "nonfinite"
        return HE_ref.clone(), maxC_ref.clone(), info
    info["ok"], info["n_tissue_tiles"] = True, m
    return HE, maxC, info


@torch.no_grad()
def apply_stains(X_u8: torch.Tensor, HE_rows: torch.Tensor, maxC_rows: torch.Tensor, HE_ref: torch.Tensor,
                 maxC_ref: torch.Tensor, io=240.0, nonneg=True, chunk=1024) -> torch.Tensor:
    """Normalise every tile with a GIVEN source stain matrix HE_rows (B,3,2) and max
    concentrations maxC_rows (B,2): C = pinv(HE) OD (>= 0 if nonneg), rescaled by
    maxC_ref / maxC, reconstructed with HE_ref. uint8 NHWC in, uint8 NHWC out."""
    dev = X_u8.device
    HE_ref = HE_ref.to(dev).float()
    maxC_ref = maxC_ref.to(dev).float()
    HE_rows = HE_rows.to(dev).float()
    maxC_rows = maxC_rows.to(dev).float()
    out = torch.empty_like(X_u8)
    for k in range(0, X_u8.shape[0], chunk):
        xb = X_u8[k:k + chunk]
        B, H, W, _ = xb.shape
        OD = _od(xb, io)
        C = concentrations(OD, HE_rows[k:k + chunk], nonneg)
        C = C * (maxC_ref.view(1, 2) / maxC_rows[k:k + chunk]).unsqueeze(2)
        I = io * torch.exp(-(HE_ref.unsqueeze(0) @ C))
        I = I.transpose(1, 2).reshape(B, H, W, 3)
        I = torch.nan_to_num(I, nan=255.0, posinf=255.0, neginf=0.0)
        out[k:k + chunk] = I.clamp(0, 255).round().to(torch.uint8)
    return out


def _cos_cols(HE: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """(G,3,2), (3,2) -> (G,2) cosine of the H and E columns with the reference."""
    return (_unit_cols(HE) * _unit_cols(R).unsqueeze(0)).sum(1)


def _q(v: np.ndarray) -> dict | None:
    if v.size == 0:
        return None
    return dict(min=float(v.min()), p10=float(np.quantile(v, 0.1)), median=float(np.median(v)),
                p90=float(np.quantile(v, 0.9)), max=float(v.max()))


@torch.no_grad()
def macenko_normalize_groups(X_u8: torch.Tensor, groups, HE_ref: torch.Tensor, maxC_ref: torch.Tensor,
                             io=240.0, alpha=1.0, beta=0.15, min_tissue=64, nonneg=True, max_tiles=256,
                             max_px=200_000, min_px=2048, chunk=1024, return_he=False, maxc="tile_median"):
    """Group-level Macenko of a uint8 NHWC tensor. `groups` (n,) group id per row (no labels).
    Returns (X_norm uint8 NHWC on the same device, stats) [+ (HE_rows, group_index) if return_he].
    stats: n_groups, n_fallback, n_rows_fallback, fallback reasons, quantiles of the per-group
    cos(H/E, reference), maxC and pooled tissue pixels, and per-group values."""
    dev = X_u8.device
    HE_ref = HE_ref.to(dev).float()
    maxC_ref = maxC_ref.to(dev).float()
    g = np.asarray(groups)
    assert g.shape == (X_u8.shape[0],), (g.shape, tuple(X_u8.shape))
    uniq, inv = np.unique(g, return_inverse=True)
    G = len(uniq)
    HEg = torch.empty(G, 3, 2, device=dev)
    maxCg = torch.empty(G, 2, device=dev)
    infos = []
    for k, gid in enumerate(uniq):
        idx = torch.from_numpy(np.nonzero(inv == k)[0]).to(dev)
        HEg[k], maxCg[k], info = estimate_group(X_u8[idx], HE_ref, maxC_ref, io, alpha, beta, min_tissue,
                                                group_seed(gid), max_tiles, max_px, min_px, nonneg, maxc)
        infos.append(info)
    inv_t = torch.from_numpy(inv.astype(np.int64)).to(dev)
    out = apply_stains(X_u8, HEg[inv_t], maxCg[inv_t], HE_ref, maxC_ref, io, nonneg, chunk)
    ok = np.array([i["ok"] for i in infos], bool)
    cos = _cos_cols(HEg, HE_ref).cpu().numpy()
    mc = maxCg.cpu().numpy()
    he = HEg.cpu().numpy()
    reasons = {}
    for i in infos:
        if not i["ok"]:
            reasons[i["reason"]] = reasons.get(i["reason"], 0) + 1
    sizes = np.bincount(inv, minlength=G)
    stats = dict(level="group", maxc=maxc, n_rows=int(len(g)), n_groups=int(G), n_fallback=int((~ok).sum()),
                 n_rows_fallback=int(sizes[~ok].sum()), fallback_reasons=reasons,
                 group_size=_q(sizes.astype(float)), n_px=_q(np.array([i["n_px"] for i in infos], float)),
                 cos_H=_q(cos[ok, 0]), cos_E=_q(cos[ok, 1]), cos_HE_ref_pair=float(HE_ref[:, 0] @ HE_ref[:, 1]),
                 cos_HE_pair=_q((he[ok, :, 0] * he[ok, :, 1]).sum(1)),
                 maxC_H=_q(mc[ok, 0]), maxC_E=_q(mc[ok, 1]),
                 frac_groups_cos_ge_095=(float(((cos[ok] >= 0.95).all(1)).mean()) if ok.any() else None),
                 per_group={str(gid.item() if hasattr(gid, "item") else gid):
                            dict(n=int(sizes[k]), ok=bool(ok[k]), n_px=int(infos[k]["n_px"]),
                                 HE=np.round(he[k], 4).tolist(), maxC=np.round(mc[k], 4).tolist(),
                                 cos=np.round(cos[k], 4).tolist())
                            for k, gid in enumerate(uniq)})
    if return_he:
        return out, stats, HEg[inv_t], inv
    return out, stats


@torch.no_grad()
def fit_reference_groups(X_u8: torch.Tensor, groups, io=240.0, alpha=1.0, beta=0.15, min_tissue=64, nonneg=True,
                         max_tiles=256, max_px=200_000, min_px=2048, maxc="pooled", n_iter=3, min_ok=3):
    """Target (HE_ref, maxC_ref) fitted with the GROUP estimator (mac_ref='train_group'):
    estimate_group on every training group (labelled w.r.t. HE_CANON), then -- as in
    fit_reference, but over groups instead of tiles -- the elementwise median of the group
    stain vectors (H/E labelling iteratively aligned to the running median, columns
    renormalised) and the median of the group maxC. Every group counts once.
    Returns (HE_ref, maxC_ref, info), or (None, None, info) if fewer than `min_ok` groups
    give an estimate."""
    g = np.asarray(groups)
    assert g.shape == (X_u8.shape[0],), (g.shape, tuple(X_u8.shape))
    dev = X_u8.device
    uniq, inv = np.unique(g, return_inverse=True)
    HEs, mCs = [], []
    for k, gid in enumerate(uniq):
        idx = torch.from_numpy(np.nonzero(inv == k)[0]).to(dev)
        HE, mC, inf = estimate_group(X_u8[idx], HE_CANON, MAXC_CANON, io, alpha, beta, min_tissue,
                                     group_seed(gid), max_tiles, max_px, min_px, nonneg, maxc)
        if inf["ok"]:
            HEs.append(HE)
            mCs.append(mC)
    info = dict(ref="train_group", maxc=maxc, n_groups=int(len(uniq)), n_groups_used=len(HEs))
    if len(HEs) < min_ok:
        return None, None, info
    HE, mC = torch.stack(HEs), torch.stack(mCs)                        # (G,3,2), (G,2)
    ref = HE_CANON.to(dev)
    for _ in range(max(1, n_iter) + 1):
        sw = _align_swap(HE, ref)
        HE = torch.where(sw.view(-1, 1, 1), HE.flip(2), HE)
        mC = torch.where(sw.view(-1, 1), mC.flip(1), mC)
        ref = _unit_cols(HE.median(0).values)
    maxC = mC.median(0).values.clamp_min(1e-3)
    cos = _cos_cols(HE, ref).cpu().numpy()
    info.update(cos_H=_q(cos[:, 0]), cos_E=_q(cos[:, 1]), maxC_H=_q(mC[:, 0].cpu().numpy()),
                maxC_E=_q(mC[:, 1].cpu().numpy()))
    return ref, maxC, info


def synth_tiles(HE: np.ndarray, C: np.ndarray, io=240.0, H=32) -> np.ndarray:
    """Render tiles from a stain matrix (3,2) and concentrations (B,2,H*H) -> uint8 (B,H,H,3).
    Used by the unit test."""
    OD = np.einsum("ij,bjn->bin", HE, C)
    I = io * np.exp(-OD) - 1.0
    B = C.shape[0]
    return np.clip(np.round(I.transpose(0, 2, 1).reshape(B, H, H, 3)), 0, 255).astype(np.uint8)
