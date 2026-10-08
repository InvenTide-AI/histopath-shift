"""Build scanner-complete fold caches from the MIDOG 2021 training data.

MIDOG 2021 (Aubreville et al., Med Image Anal 84:102699) is 200 breast-cancer
regions cropped from whole slide images, 50 from each of four scanners, all
from one laboratory (UMC Utrecht). Because the lab is held constant, every
visible difference between the four subsets is attributable to the acquisition
device. That makes the scanner a clean stand-in for the site variable in
Camelyon17, with the stain and case-mix components removed.

Annotations exist for three of the four scanners (images 001-150) and give a
binary label per object: "mitotic figure" versus "not mitotic figure", the
latter being hard look-alikes. Classifying those two against each other is
exactly the second stage of the challenge-winning pipeline, so the task here
sits inside the published problem rather than beside it.

Fold layout, K = 3 scanners. Unlike Camelyon17 there is no spare labelled
domain to serve as an out-of-distribution validation set: holding one scanner
out for test and another for validation would leave a single training domain
and degenerate every group-based method. So the two remaining scanners are both
used for training and epochs are selected on a held-out split of them. This is
recorded in the manifest and is the same deviation already documented for
NCT-CRC-HE.

Output per fold, matching the Camelyon17 cache contract exactly so the existing
arm runner needs no changes:
    <out>/fold{t}/splits.npz     Xtr ytr str Xiv yiv Xov yov Xot yot
    <out>/fold{t}/unlabeled.npz  Xu
    <out>/fold{t}/manifest.json
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SCANNERS = ["HamamatsuXR", "HamamatsuS360", "AperioCS2", "LeicaGT450"]
PATCH = 64
N_ANNOTATED = 150          # images 001-150 carry annotations
UNLAB_PER_IMAGE = 300      # random tissue crops per training image


def scanner_of(image_number: int) -> int:
    """Images 1-50 are scanner 0, 51-100 scanner 1, 101-150 scanner 2, 151-200 scanner 3."""
    return (image_number - 1) // 50


def crop(im: Image.Image, cx: float, cy: float, size: int = PATCH) -> np.ndarray:
    """Centre crop of `size` at (cx, cy), clamped to stay inside the image."""
    half = size // 2
    x = int(round(cx)) - half
    y = int(round(cy)) - half
    x = max(0, min(x, im.width - size))
    y = max(0, min(y, im.height - size))
    return np.asarray(im.crop((x, y, x + size, y + size)).convert("RGB"))


def is_tissue(patch: np.ndarray) -> bool:
    """Reject near-white glass and flat regions."""
    return patch.mean() < 220.0 and patch.std() > 12.0


def extract(src: str, ann_path: str, verbose: bool = True):
    """Return per-scanner labelled patches plus per-image unlabelled tissue crops."""
    ann = json.load(open(ann_path))
    by_image = {}
    for a in ann["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    num_of = {im["id"]: int(im["file_name"].split(".")[0]) for im in ann["images"]}

    X = {s: [] for s in range(3)}
    y = {s: [] for s in range(3)}
    unlab = {}                       # image number -> array of tissue crops
    rng = np.random.default_rng(0)

    for image_id in sorted(num_of):
        n = num_of[image_id]
        path = os.path.join(src, "%03d.tiff" % n)
        if not os.path.exists(path):
            if verbose:
                print("  missing %s, skipping" % os.path.basename(path), flush=True)
            continue
        im = Image.open(path)
        s = scanner_of(n)

        if n <= N_ANNOTATED:
            for a in by_image.get(image_id, []):
                x0, y0, x1, y1 = a["bbox"]
                p = crop(im, (x0 + x1) / 2.0, (y0 + y1) / 2.0)
                X[s].append(p)
                # category 1 = mitotic figure (positive), 2 = look-alike (negative)
                y[s].append(1 if a["category_id"] == 1 else 0)

        # unlabelled tissue pool: random crops away from the annotated objects
        keep = []
        tries = 0
        while len(keep) < UNLAB_PER_IMAGE and tries < UNLAB_PER_IMAGE * 8:
            tries += 1
            cx = rng.integers(PATCH, im.width - PATCH)
            cy = rng.integers(PATCH, im.height - PATCH)
            p = crop(im, cx, cy)
            if is_tissue(p):
                keep.append(p)
        unlab[n] = np.asarray(keep, dtype=np.uint8)
        if verbose:
            print("  %03d.tiff scanner=%-14s labelled=%4d unlab=%d"
                  % (n, SCANNERS[s], len(by_image.get(image_id, [])), len(keep)),
                  flush=True)

    out_X = {s: np.asarray(X[s], dtype=np.uint8) for s in range(3)}
    out_y = {s: np.asarray(y[s], dtype=np.int64) for s in range(3)}
    return out_X, out_y, unlab


def build_folds(out_root: str, X, y, unlab, id_val_frac: float = 0.15,
                unlab_cap: int = 40000, seed: int = 0):
    rng = np.random.default_rng(seed)
    for t in range(3):
        train_scanners = [s for s in range(3) if s != t]
        Xtr_parts, ytr_parts, str_parts = [], [], []
        Xiv_parts, yiv_parts = [], []
        for s in train_scanners:
            idx = rng.permutation(len(X[s]))
            n_val = int(round(id_val_frac * len(idx)))
            val_idx, tr_idx = idx[:n_val], idx[n_val:]
            Xtr_parts.append(X[s][tr_idx]); ytr_parts.append(y[s][tr_idx])
            str_parts.append(np.full(len(tr_idx), s, dtype=np.int64))
            Xiv_parts.append(X[s][val_idx]); yiv_parts.append(y[s][val_idx])

        Xtr = np.concatenate(Xtr_parts); ytr = np.concatenate(ytr_parts)
        s_tr = np.concatenate(str_parts)
        Xiv = np.concatenate(Xiv_parts); yiv = np.concatenate(yiv_parts)

        # unlabelled pool: training scanners only, never the test scanner
        pool = [unlab[n] for n in sorted(unlab)
                if scanner_of(n) in train_scanners and len(unlab[n])]
        Xu = np.concatenate(pool) if pool else np.zeros((0, PATCH, PATCH, 3), np.uint8)
        if len(Xu) > unlab_cap:
            Xu = Xu[rng.permutation(len(Xu))[:unlab_cap]]

        d = os.path.join(out_root, "fold%d" % t)
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(
            os.path.join(d, "splits.npz"),
            Xtr=Xtr, ytr=ytr, str=s_tr,
            Xiv=Xiv, yiv=yiv,
            # K=3 leaves no spare labelled domain: the selection set is the
            # held-out split of the training scanners, not a held-out scanner.
            Xov=Xiv, yov=yiv,
            Xot=X[t], yot=y[t])
        np.savez_compressed(os.path.join(d, "unlabeled.npz"), Xu=Xu)
        json.dump({
            "dataset": "MIDOG2021",
            "domain_variable": "slide scanner (lab held constant: UMC Utrecht)",
            "fold": t,
            "test_scanner": SCANNERS[t],
            "train_scanners": [SCANNERS[s] for s in train_scanners],
            "n_train": int(len(Xtr)), "n_id_val": int(len(Xiv)),
            "n_ood_test": int(len(X[t])), "n_unlabeled": int(len(Xu)),
            "pos_rate_train": float(ytr.mean()), "pos_rate_test": float(y[t].mean()),
            "patch": PATCH,
            "task": "mitotic figure vs annotated non-mitotic look-alike",
            "model_selection": ("held-out split of the training scanners; K=3 "
                                "leaves no spare labelled domain for an "
                                "out-of-distribution validation set"),
        }, open(os.path.join(d, "manifest.json"), "w"), indent=1)
        print("fold %d: test=%s train=%s  n_train=%d n_val=%d n_test=%d n_unlab=%d"
              % (t, SCANNERS[t], [SCANNERS[s] for s in train_scanners],
                 len(Xtr), len(Xiv), len(X[t]), len(Xu)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="./data/midog")
    ap.add_argument("--ann", default=None, help="MIDOG.json (defaults to <src>/MIDOG.json)")
    ap.add_argument("--out", default="./data/midog_cache")
    a = ap.parse_args()
    ann = a.ann or os.path.join(a.src, "MIDOG.json")
    print("extracting patches ...", flush=True)
    X, y, unlab = extract(a.src, ann)
    for s in range(3):
        print("scanner %-14s n=%5d  positives=%.3f"
              % (SCANNERS[s], len(X[s]), y[s].mean() if len(y[s]) else float("nan")))
    os.makedirs(a.out, exist_ok=True)
    build_folds(a.out, X, y, unlab)


if __name__ == "__main__":
    main()
