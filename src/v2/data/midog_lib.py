"""Patch extraction shared by prepare_midog21.py and prepare_midogpp.py.

Per image (one worker process per image):
  * labelled patches: centred on each annotation bbox centre (always; no
    clamping), crop of cp = round(P * s) px, resized to P (PIL bilinear) when
    cp != P; produced for every P in SIZES from the same centre. Objects whose
    crop would leave the image are cut from a reflect-padded copy of the image
    (np.pad mode="reflect", margin ceil(96*s)+8 px) so the object stays at the
    patch centre at every size; the number of padded px per row and size is
    returned (`pad`) and stored as rows.npz pad_<split> / manifest
    n_edge_padded_<split>. (v1 clamped the crop inside the image instead, which
    moved near-border objects off-centre by up to cp/2 and made the 64- and
    96-px crops of one object non-concentric.)
  * unlabelled tissue crops: up to n_unlab random centres per image, drawn with
    an image-specific RNG (default_rng([seed, image_number])) inside
    [ceil(96*s), W - ceil(96*s)) so that every size fits without padding;
    accepted iff the 64-px crop passes v1's MIDOG tissue filter
    (mean < 220, std > 12) and the centre was not drawn before in this image
    (no duplicate crops); max tries_per*n_unlab tries (v1: 8).
s = target_mpp / image_mpp (scale normalisation) or 1.0.
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
SIZES = (64, 96)


def _origin(cx, cy, cp):
    half = cp // 2
    return int(round(cx)) - half, int(round(cy)) - half


def _overhang(x, y, cp, W, H):
    """px by which a crop with top-left (x, y) leaves the W x H image (0 = inside)."""
    return max(0, -x, -y, x + cp - W, y + cp - H)


def _crop_padded(imgp, M, x, y, cp):
    """crop with top-left (x, y) in original-image coordinates from imgp = image padded by M."""
    assert x + M >= 0 and y + M >= 0 and x + M + cp <= imgp.shape[1] and y + M + cp <= imgp.shape[0], \
        (x, y, cp, M, imgp.shape)
    return imgp[y + M:y + M + cp, x + M:x + M + cp]


def _fit(p, P):
    if p.shape[0] == P:
        return np.ascontiguousarray(p)
    return np.asarray(Image.fromarray(np.ascontiguousarray(p)).resize((P, P), Image.BILINEAR))


def is_tissue(p):
    return p.mean() < 220.0 and p.std() > 12.0


def work(job):
    """job = dict(path, num, objs=[(cx, cy, label, ann_id)], s, seed, n_unlab[, tries_per, rng_key]).
    rng_key (default [seed, num]) seeds the unlabelled-centre RNG."""
    img = np.asarray(Image.open(job["path"]).convert("RGB"))
    H, W = img.shape[:2]
    s = job["s"]
    cps = {P: int(round(P * s)) for P in SIZES}
    M = int(math.ceil(max(SIZES) * s)) + 8
    org = [{P: _origin(cx, cy, cps[P]) for P in SIZES} for cx, cy, _, _ in job["objs"]]
    pad = {P: np.asarray([_overhang(*o[P], cps[P], W, H) for o in org], np.int64) for P in SIZES}
    need_pad = any(len(v) and v.max() > 0 for v in pad.values())
    imgp = np.pad(img, ((M, M), (M, M), (0, 0)), mode="reflect") if need_pad else None
    lab = {P: [] for P in SIZES}
    y, aid, cxy = [], [], []
    for k, (cx, cy, l, a) in enumerate(job["objs"]):
        for P in SIZES:
            x0, y0 = org[k][P]
            if pad[P][k] > 0:
                p = _crop_padded(imgp, M, x0, y0, cps[P])
            else:
                p = img[y0:y0 + cps[P], x0:x0 + cps[P]]
            lab[P].append(_fit(p, P))
        y.append(l); aid.append(a); cxy.append((cx, cy))
    del imgp
    unl = {P: [] for P in SIZES}
    ucxy = []
    n = job["n_unlab"]
    n_tries = 0
    if n > 0:
        rng = np.random.default_rng(job.get("rng_key", [int(job["seed"]), int(job["num"])]))
        m = int(math.ceil(96 * s)) + 1
        seen = set()
        while len(ucxy) < n and n_tries < n * job.get("tries_per", 8):
            n_tries += 1
            cx = int(rng.integers(m, W - m)); cy = int(rng.integers(m, H - m))
            if (cx, cy) in seen:
                continue
            o64 = _origin(cx, cy, cps[64])
            p64 = _fit(img[o64[1]:o64[1] + cps[64], o64[0]:o64[0] + cps[64]], 64)
            if not is_tissue(p64):
                continue
            seen.add((cx, cy))
            unl[64].append(p64)
            o96 = _origin(cx, cy, cps[96])
            unl[96].append(_fit(img[o96[1]:o96[1] + cps[96], o96[0]:o96[0] + cps[96]], 96))
            ucxy.append((cx, cy))
    stack = lambda L, P: np.stack(L) if L else np.zeros((0, P, P, 3), np.uint8)
    return dict(num=job["num"], H=H, W=W, s=s, crop_px=cps,
                lab={P: stack(lab[P], P) for P in SIZES}, y=np.asarray(y, np.int64), pad=pad,
                n_tries=n_tries,
                ann_id=np.asarray(aid, np.int64), cxy=np.asarray(cxy, float).reshape(-1, 2),
                unl={P: stack(unl[P], P) for P in SIZES}, ucxy=np.asarray(ucxy, np.int64).reshape(-1, 2))


def run_all(jobs, workers):
    from multiprocessing import Pool
    out = {}
    with Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(work, jobs)):
            out[r["num"]] = r
            if (k + 1) % 25 == 0:
                print("  %d/%d images" % (k + 1, len(jobs)), flush=True)
    return out


def assemble_pad(res, nums, P):
    """Per-row edge padding (px) of the labelled patches of images `nums` at size P."""
    return np.concatenate([res[n]["pad"][P] for n in nums]) if nums else np.zeros(0, np.int64)


def assemble(res, nums, P, what="lab"):
    """Concatenate per-image arrays for images `nums` (order kept). Returns X, y, img."""
    X = [res[n][what][P] for n in nums]
    img = [np.full(len(res[n][what][P]), n, np.int64) for n in nums]
    X = np.concatenate(X) if X else np.zeros((0, P, P, 3), np.uint8)
    img = np.concatenate(img) if img else np.zeros(0, np.int64)
    if what == "lab":
        y = np.concatenate([res[n]["y"] for n in nums]) if nums else np.zeros(0, np.int64)
        return X, y, img
    return X, None, img
