"""v2 shared utilities: fold-cache loading (CONTRACTS.md A), metrics, seeding,
code-version hash, run-record helpers (CONTRACTS.md B).

Images are kept as uint8 NHWC tensors (GPU-resident when they fit) and converted
to float [0,1] NCHW per batch; normalisation / resizing is done inside the model
(see backbones.Preproc), so every arm sees the same [0,1] input space.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

DATA_ROOT = os.environ.get("HISTOPATH_DATA", "data")
V2 = os.path.join(DATA_ROOT, "v2")
BACKBONE_DIR = os.path.join(DATA_ROOT, "backbone")
PCAM_X = os.path.join(DATA_ROOT, "pcam", "camelyonpatch_level_2_split_test_x.h5")
PCAM_Y = os.path.join(DATA_ROOT, "pcam", "camelyonpatch_level_2_split_test_y.h5")
HERE = os.path.dirname(os.path.abspath(__file__))

# Primary cohorts. midog21sn = MIDOG 2021 scale-normalised to constant physical extent (as canine and
# MIDOG++); the un-normalised "midog21" cache is kept for a tier-C sensitivity analysis.
COHORTS = ("c17", "canine", "midog21sn", "midogpp")
N_FOLDS = {"c17": 5, "canine": 5, "midog21": 3, "midog21sn": 3, "midogpp": 7}
TIER_RES = {"C": 64, "S": 96}

# Files that determine training behaviour. stats*/data/ (other tracks) and the
# planning scripts are deliberately excluded so that editing them does not change
# the code_version of otherwise identical runs.
CODE_FILES = ("common.py", "backbones.py", "arms.py", "stain.py", "ssl_v2.py", "run_v2.py")


# ----------------------------------------------------------------------------- misc
def code_version() -> str:
    h = hashlib.sha256()
    for name in CODE_FILES:
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            h.update(name.encode())
            with open(p, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


class rng_isolated:
    """Context manager: code inside consumes python / numpy / torch (CPU + CUDA) global RNG
    state without affecting what the caller draws afterwards (state restored on exit)."""

    def __enter__(self):
        self.py, self.np, self.cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
        self.cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        return self

    def __exit__(self, *exc):
        random.setstate(self.py)
        np.random.set_state(self.np)
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state_all(self.cuda)
        return False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fold_dir(cache_root: str, tier: str, cohort: str, fold: int) -> str:
    return os.path.join(cache_root, f"cache{TIER_RES[tier]}", cohort, f"fold{fold}")


def _tmp_name(path: str) -> str:
    """Process- and call-unique temp name next to `path` (concurrent writers never share it)."""
    import uuid
    d, b = os.path.split(path)
    return os.path.join(d, f".{b}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")


def atomic_write(path: str, writer):
    """writer(tmp_path) writes the file; then atomically rename onto `path`. Concurrent
    writers of the same target each use their own temp file, so the last rename wins
    and the target is always a complete file."""
    tmp = _tmp_name(path)
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_json(obj, path):
    def w(tmp):
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=1, default=_json_default)
    atomic_write(path, w)


def atomic_npy(arr, path):
    def w(tmp):
        with open(tmp, "wb") as f:
            np.save(f, arr)
    atomic_write(path, w)


def atomic_npz(path, **arrays):
    def w(tmp):
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
    atomic_write(path, w)


def atomic_torch(obj, path):
    atomic_write(path, lambda tmp: torch.save(obj, tmp))


def eval_steps(n_steps: int, eval_every: int = 0, n_ckpt: int = 10) -> list:
    """Checkpoint steps. Default (eval_every=0): exactly `n_ckpt` checkpoints at
    round(k * n_steps / n_ckpt), k = 1..n_ckpt (the last one at n_steps), for every budget
    (1200 -> 120, 240, ..., 1200; compute_matched 4944 -> 494, 989, ..., 4944).
    eval_every > 0: every eval_every steps, plus n_steps if not a multiple."""
    if eval_every and eval_every > 0:
        s = list(range(int(eval_every), n_steps + 1, int(eval_every)))
        return s if (s and s[-1] == n_steps) else s + [n_steps]
    return sorted({max(1, int(round(k * n_steps / n_ckpt))) for k in range(1, n_ckpt + 1)})


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    raise TypeError(type(o))


def fmt_val(v) -> str:
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        return "%g" % v
    return str(v)


# ----------------------------------------------------------------------------- data
SPLIT_FILES = {
    "tr": ("Xtr", "ytr", "dtr", "gtr"),
    "iv": ("Xiv", "yiv", "div", "giv"),
    "ov": ("Xov", "yov", None, None),
    "ot": ("Xot", "yot", None, None),
}


class Fold:
    """One fold cache (CONTRACTS.md A), loaded once.

    X[split] : uint8 NHWC torch tensor on `device` (or CPU when it does not fit)
    y[split] : int64 numpy array; yt[split] the same on device
    d_tr     : int64 tensor on device, training-domain index 0..K-1 (asserted contiguous)
    """

    def __init__(self, path: str, device="cuda", load_u: bool = False, gpu_budget_gb: float = 40.0):
        self.path = path
        self.device = torch.device(device)
        mpath = os.path.join(path, "manifest.json")
        self.manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
        self.X, self.y, self.yt, self.g = {}, {}, {}, {}
        self.d_np = {}
        arrays = {}
        for sp, (xk, yk, dk, gk) in SPLIT_FILES.items():
            fx = os.path.join(path, xk + ".npy")
            if not os.path.exists(fx):
                if sp == "ov":
                    continue
                raise FileNotFoundError(fx)
            arrays[sp] = (np.load(fx, mmap_mode="r"), np.load(os.path.join(path, yk + ".npy")).astype(np.int64),
                          np.load(os.path.join(path, dk + ".npy")).astype(np.int64)
                          if dk and os.path.exists(os.path.join(path, dk + ".npy")) else None,
                          np.load(os.path.join(path, gk + ".npy"))
                          if gk and os.path.exists(os.path.join(path, gk + ".npy")) else None)
        nbytes = sum(a[0].nbytes for a in arrays.values())
        on_gpu = self.device.type == "cuda" and nbytes / 1e9 < gpu_budget_gb
        self.img_device = self.device if on_gpu else torch.device("cpu")
        for sp, (X, y, d, g) in arrays.items():
            self.X[sp] = torch.from_numpy(np.array(X)).to(self.img_device)
            self.y[sp] = y
            self.yt[sp] = torch.from_numpy(y).to(self.device)
            if d is not None:
                self.d_np[sp] = d
            if g is not None:
                self.g[sp] = g
        self.has_ov = "ov" in self.X
        d = self.d_np["tr"]
        self.K = int(len(self.manifest.get("train_domains", [])) or (d.max() + 1))
        uniq = np.unique(d)
        assert uniq.min() >= 0 and uniq.max() < self.K and len(uniq) == self.K, \
            f"dtr must be contiguous 0..K-1 (K={self.K}), got {uniq.tolist()}"
        self.d_tr = torch.from_numpy(d).to(self.device)
        self.P = int(self.X["tr"].shape[1])
        self._Xu = None
        if load_u:
            self.Xu  # noqa: B018 (force load)
        self._transformed = {}

    @property
    def Xu(self):
        if self._Xu is None:
            fu = os.path.join(self.path, "Xu.npy")
            Xu = np.load(fu, mmap_mode="r")
            dev = self.img_device
            if dev.type == "cuda" and Xu.nbytes / 1e9 > 30:
                dev = torch.device("cpu")
            self._Xu = torch.from_numpy(np.array(Xu)).to(dev)
        return self._Xu

    def n(self, sp):
        return int(self.X[sp].shape[0])

    @property
    def fingerprint(self) -> str:
        if getattr(self, "_fp", None) is None:
            self._fp = cache_fingerprint(self.path)
        return self._fp

    def info(self):
        m = self.manifest
        return dict(test_domain=m.get("test_domain"), val_domain=m.get("val_domain"),
                    train_domains=m.get("train_domains"))

    def transformed(self, key, fn):
        """Cache a per-split transformed copy of the image tensors (e.g. Macenko)."""
        if key not in self._transformed:
            t0 = time.time()
            self._transformed[key] = {sp: fn(X) for sp, X in self.X.items()}
            self._transformed[key]["_time"] = time.time() - t0
        return self._transformed[key]


FP_FILES = ("manifest.json", "rows.npz", "ytr.npy", "dtr.npy", "gtr.npy", "yiv.npy", "div.npy", "giv.npy",
            "yov.npy", "gov.npy", "yot.npy", "got.npy", "du.npy", "gu.npy")
FP_IMAGES = ("Xtr", "Xiv", "Xov", "Xot", "Xu")


def cache_fingerprint(path: str, n_rows: int = 256) -> str:
    """Identity of a fold cache: sha256 over manifest.json, every label / domain / group /
    row-id array, and the shape plus `n_rows` evenly spaced rows of every image array.
    A rebuilt cache (new manifest `created`, different rows or labels) gets a new
    fingerprint; reading it costs well under a second."""
    h = hashlib.sha256()
    for name in FP_FILES:
        p = os.path.join(path, name)
        if os.path.exists(p):
            h.update(name.encode())
            with open(p, "rb") as f:
                h.update(f.read())
    for name in FP_IMAGES:
        p = os.path.join(path, name + ".npy")
        if not os.path.exists(p):
            continue
        X = np.load(p, mmap_mode="r")
        h.update(f"{name}{X.shape}{X.dtype}".encode())
        if X.shape[0]:
            idx = np.unique(np.linspace(0, X.shape[0] - 1, min(n_rows, X.shape[0])).round().astype(np.int64))
            h.update(np.ascontiguousarray(X[idx]).tobytes())
    return h.hexdigest()


def to_float(xb_u8: torch.Tensor, device=None) -> torch.Tensor:
    """(B,H,W,3) uint8 -> (B,3,H,W) float32 in [0,1] (on `device`)."""
    if device is not None:
        xb_u8 = xb_u8.to(device, non_blocking=True)
    return xb_u8.permute(0, 3, 1, 2).float().div_(255.0)


class EpochSampler:
    """Sampling without replacement, reshuffled every epoch, last partial batch dropped."""

    def __init__(self, n, bs, seed, device):
        self.n, self.bs, self.device = n, bs, device
        self.g = torch.Generator(device="cpu").manual_seed(int(seed) + 12345)
        self.perm, self.pos = None, n

    def next(self):
        if self.pos + self.bs > self.n:
            self.perm = torch.randperm(self.n, generator=self.g).to(self.device)
            self.pos = 0
        b = self.perm[self.pos:self.pos + self.bs]
        self.pos += self.bs
        return b


def load_pcam(device, crop: int | None):
    import h5py
    with h5py.File(PCAM_X, "r") as f:
        X = f["x"][:]
    with h5py.File(PCAM_Y, "r") as f:
        y = f["y"][:].reshape(-1).astype(np.int64)
    if crop is not None and crop < X.shape[1]:
        l = (X.shape[1] - crop) // 2
        X = X[:, l:l + crop, l:l + crop, :]
    return torch.from_numpy(np.ascontiguousarray(X)).to(device), y


# ----------------------------------------------------------------------------- inference
@torch.no_grad()
def predict(model, X_u8, device, bs=1024, amp=False, channels_last=False):
    """Positive-class probabilities, float64 numpy, in the row order of X_u8.
    The v2 drivers always call this with amp=False: evaluation is fp32 (TF32 matmuls)
    on both tiers, even when training used bf16 autocast."""
    model.eval()
    out = []
    for k in range(0, X_u8.shape[0], bs):
        x = to_float(X_u8[k:k + bs], device)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and x.is_cuda):
            logits = model(x)
        out.append(torch.softmax(logits.double(), 1)[:, 1])   # float64 softmax: no fp32 saturation ties
    return torch.cat(out).double().cpu().numpy()


# ----------------------------------------------------------------------------- metrics
METRIC_KEYS = ("auroc", "ap", "acc", "bacc", "f1", "ece", "brier", "nll")


def ece_score(y, p, n_bins=15):
    """Expected calibration error, 15 equal-width bins on the confidence of the
    predicted class (threshold 0.5) -- same definition as v1."""
    yhat = (p >= 0.5).astype(int)
    conf = np.where(yhat == 1, p, 1 - p)
    correct = (yhat == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def sanitize_probs(p):
    """(p with non-finite entries replaced by 0.5 = chance, number replaced)."""
    p = np.asarray(p, dtype=np.float64)
    bad = ~np.isfinite(p)
    n_bad = int(bad.sum())
    if n_bad:
        p = np.where(bad, 0.5, p)
    return p, n_bad


def metrics(y, p) -> dict:
    """The 8 metrics. Non-finite probabilities (diverged model) are scored as 0.5 (chance)
    instead of raising; run_v2 flags such runs with status='diverged'."""
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
    y = np.asarray(y).astype(int)
    p, _ = sanitize_probs(p)
    yhat = (p >= 0.5).astype(int)
    two = len(np.unique(y)) == 2
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return dict(
        auroc=float(roc_auc_score(y, p)) if two else None,
        ap=float(average_precision_score(y, p)) if two else None,
        acc=float((yhat == y).mean()),
        bacc=float(balanced_accuracy_score(y, yhat)) if two else None,
        f1=float(f1_score(y, yhat, zero_division=0)),
        ece=ece_score(y, p),
        brier=float(np.mean((p - y) ** 2)),
        nll=float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
    )


def cosine_lambda(total, warmup=0):
    def f(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return f


# ----------------------------------------------------------------------------- contract B
RECORD_KEYS = ("schema", "arm", "tier", "backbone", "init", "cohort", "fold", "test_domain", "val_domain", "seed",
               "hparams", "phase", "input_res", "n_params", "n_train", "wall_s", "checkpoints", "selected",
               "test_at", "code_version", "torch", "device")


def validate_record(rec: dict, preds: dict | None = None):
    """Raise AssertionError if `rec` (and optionally the preds npz dict) violates CONTRACTS.md B."""
    for k in RECORD_KEYS:
        assert k in rec, f"missing key {k}"
    assert rec["schema"] == "v2" and rec["tier"] in ("C", "S")
    assert isinstance(rec["hparams"], dict) and {"lr", "steps", "bs"} <= set(rec["hparams"])
    has_ov = rec["checkpoints"][0]["ood_val"] is not None
    rules = {"id", "last", "oracle"} | ({"ood"} if has_ov else set())
    assert set(rec["selected"]) == rules, (rec["selected"], rules)
    assert set(rec["test_at"]) == rules
    steps = [c["step"] for c in rec["checkpoints"]]
    assert steps == sorted(steps) and len(set(steps)) == len(steps)
    for r in rules:
        assert rec["selected"][r] in steps
    for c in rec["checkpoints"]:
        assert "train_loss" in c
        for sp in ("id_val", "ood_val", "ood_test"):
            if sp == "ood_val" and not has_ov:
                assert c[sp] is None
                continue
            assert set(METRIC_KEYS) <= set(c[sp]), (sp, c[sp])
    status = rec.get("status", "ok")
    assert status in ("ok", "diverged"), status
    if status == "ok" and not int(rec["hparams"].get("eval_every", 0) or 0):
        assert len(steps) == 10, f"expected 10 checkpoints, got {len(steps)}"
    if "total_steps" in rec["hparams"]:
        assert status == "diverged" or steps[-1] == rec["hparams"]["total_steps"]
    for r in rules:
        assert set(METRIC_KEYS) <= set(rec["test_at"][r])
    for opt in ("tta", "pcam"):
        if opt in rec:
            for v in rec[opt].values():
                assert set(METRIC_KEYS) <= set(v)
    if "pcam" in rec:
        assert rec["cohort"] == "c17" and "id" in rec["pcam"]
    if preds is not None:
        need = {"y_ood_test", "p_ood_test_id", "p_ood_test_last"} | ({"p_ood_test_ood"} if has_ov else set())
        need |= {f"p_tta_{t}" for t in rec.get("tta", {})}
        assert need <= set(preds), (need, set(preds))
        n = len(preds["y_ood_test"])
        for k in need:
            if k.startswith(("p_ood_test", "p_tta")):
                assert len(preds[k]) == n
        for sp in ("id_val", "ood_val"):   # optional per-row predictions at the selected checkpoints
            ks = [k for k in preds if k.startswith(f"p_{sp}_")]
            if ks:
                assert f"y_{sp}" in preds
                for k in ks:
                    assert len(preds[k]) == len(preds[f"y_{sp}"])
    return True


def list_code_files():
    return sorted(glob.glob(os.path.join(HERE, "*.py")))
