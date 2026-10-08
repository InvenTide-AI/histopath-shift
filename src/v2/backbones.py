"""v2 backbones.

Every model is a `Net`: Preproc (resize + per-init normalisation, no parameters,
non-persistent buffers) -> encoder -> fc. Inputs are float [0,1] NCHW at the cache
resolution; Net handles everything else, so arms/augmentations are backbone-agnostic.

Input-resolution policy (PLAN §3):
  cnn4                 native (64 px tier C, 96 px tier S)
  resnet50 fine-tune   native (96 px)
  ViT (fine-tune)      upsampled to 224 (bilinear)
  every frozen probe   upsampled to 224 (bilinear) -- including vit_b16_swag, whose
                       384-px position embedding is interpolated to 14x14.

Normalisation constants:
  ImageNet: torchvision defaults.
  Lunit (Kang et al., CVPR 2023): "Image statistics" section of the release notes,
  https://github.com/lunit-io/benchmark-ssl-pathology/releases/tag/pretrained-weights
  mean [0.70322989, 0.53606487, 0.66096631], std [0.21716536, 0.26081574, 0.20723464].
  cnn4 / random init uses ImageNet constants (identical to v1).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import BACKBONE_DIR

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
# https://github.com/lunit-io/benchmark-ssl-pathology/releases/tag/pretrained-weights
LUNIT_MEAN, LUNIT_STD = (0.70322989, 0.53606487, 0.66096631), (0.21716536, 0.26081574, 0.20723464)

WEIGHT_FILES = {
    ("resnet50", "imagenet"): "resnet50.pth",
    ("resnet50", "lunit_bt"): "lunit_bt_rn50_ep200.torch",
    ("resnet50", "lunit_swav"): "lunit_swav_rn50_ep200.torch",
    ("resnet50", "lunit_mocov2"): "lunit_mocov2_rn50_ep200.torch",
    ("vit_s16", "lunit_dino"): "lunit_dino_vit_small_patch16_ep200.torch",
    ("resnet18", "imagenet"): "resnet18.pth",
    ("convnext_tiny", "imagenet"): "convnext_tiny.pth",
    ("vit_b16", "imagenet"): "vit_b_16.pth",
    ("vit_b16", "swag"): "vit_b_16_swag.pth",
}
# images seen in pre-training (for the representation ladder table)
PRETRAIN_IMAGES = {"imagenet": 1.28e6, "swag": 3.6e9, "lunit_bt": 19e6, "lunit_swav": 19e6,
                   "lunit_mocov2": 19e6, "lunit_dino": 19e6, "random": 0}
BACKBONES = ("cnn4", "resnet50", "vit_s16", "resnet18", "convnext_tiny", "vit_b16")
INITS = ("random", "imagenet", "lunit_bt", "lunit_swav", "lunit_mocov2", "lunit_dino", "swag")


def norm_for(init):
    return (LUNIT_MEAN, LUNIT_STD) if init.startswith("lunit") else (IMAGENET_MEAN, IMAGENET_STD)


class Preproc(nn.Module):
    def __init__(self, mean, std, resize_to=None):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.resize_to = resize_to

    def forward(self, x):
        if self.resize_to is not None and x.shape[-1] != self.resize_to:
            x = F.interpolate(x, size=(self.resize_to, self.resize_to), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std


class MixStyle(nn.Module):
    """MixStyle (Zhou et al., ICLR 2021), 'random' mixing as in the reference code
    (github.com/KaiyangZhou/mixstyle-release): statistics detached, Beta(alpha, alpha)
    mixing weights, applied with probability p. Identity in eval mode -- it is a
    registered submodule, so model.eval() reaches it (fix D1)."""

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p, self.alpha, self.eps = float(p), float(alpha), eps

    def forward(self, x):
        if not self.training or torch.rand(()) > self.p:
            return x
        B = x.shape[0]
        xf = x.float()
        mu = xf.mean(dim=[2, 3], keepdim=True)
        sig = (xf.var(dim=[2, 3], keepdim=True) + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        xn = (xf - mu) / sig
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device)
        perm = torch.randperm(B, device=x.device)
        mu2, sig2 = mu[perm], sig[perm]
        out = xn * (lam * sig + (1 - lam) * sig2) + (lam * mu + (1 - lam) * mu2)
        return out.to(x.dtype)

    def extra_repr(self):
        return f"p={self.p}, alpha={self.alpha}"


# ----------------------------------------------------------------------------- cnn4
def conv_bn(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class CNN4(nn.Module):
    """Same architecture and state_dict keys as v1 train_lib.Encoder (583,394 params
    with the 2-class head). MixStyle (if any) is applied after the stage-1 and
    stage-2 MaxPool (features[2], features[5])."""
    MS_AFTER = (2, 5)

    def __init__(self, w=(32, 64, 128, 256), mixstyle=None):
        super().__init__()
        a, b, c, d = w
        self.features = nn.Sequential(
            conv_bn(3, a), conv_bn(a, a), nn.MaxPool2d(2),
            conv_bn(a, b), conv_bn(b, b), nn.MaxPool2d(2),
            conv_bn(b, c), conv_bn(c, c), nn.MaxPool2d(2),
            conv_bn(c, d), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.out_dim = d
        self.mixstyle = MixStyle(**mixstyle) if mixstyle else None

    def forward(self, x):
        if self.mixstyle is None:
            return self.features(x)
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.MS_AFTER:
                x = self.mixstyle(x)
        return x


# ----------------------------------------------------------------------------- torchvision ResNets
class ResNetEnc(nn.Module):
    """torchvision ResNet trunk (fc removed). MixStyle after layer1 and layer2."""

    def __init__(self, arch="resnet50", mixstyle=None):
        super().__init__()
        import torchvision.models as tvm
        self.body = getattr(tvm, arch)(weights=None)
        self.out_dim = self.body.fc.in_features
        self.body.fc = nn.Identity()
        self.mixstyle = MixStyle(**mixstyle) if mixstyle else None

    def forward(self, x):
        b = self.body
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        x = b.layer1(x)
        if self.mixstyle is not None:
            x = self.mixstyle(x)
        x = b.layer2(x)
        if self.mixstyle is not None:
            x = self.mixstyle(x)
        x = b.layer4(b.layer3(x))
        return torch.flatten(b.avgpool(x), 1)

    def load_init(self, path):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
        self.body.load_state_dict(sd, strict=True)


class TimmViTS(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.body = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
        self.out_dim = self.body.num_features

    def forward(self, x):
        return self.body(x)

    def load_init(self, path):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        self.body.load_state_dict(sd, strict=True)


class TVGeneric(nn.Module):
    """torchvision convnext_tiny / vit_b_16 (incl. SWAG) trunk for frozen probes."""

    def __init__(self, arch, init):
        super().__init__()
        import torchvision.models as tvm
        self.arch, self.init = arch, init
        if arch == "convnext_tiny":
            self.body = tvm.convnext_tiny(weights=None)
            self.out_dim = 768
        elif arch == "vit_b16":
            self.body = tvm.vit_b_16(weights=None, image_size=224)
            self.out_dim = 768
        else:
            raise ValueError(arch)

    def load_init(self, path):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if self.arch == "vit_b16":
            pe = sd["encoder.pos_embedding"]
            if pe.shape[1] != self.body.encoder.pos_embedding.shape[1]:
                from torchvision.models.vision_transformer import interpolate_embeddings
                sd = interpolate_embeddings(224, 16, sd)
        self.body.load_state_dict(sd, strict=True)
        if self.arch == "convnext_tiny":
            self.body.classifier = nn.Sequential(*list(self.body.classifier)[:-1], nn.Flatten(1))
        else:
            self.body.heads = nn.Identity()

    def forward(self, x):
        return self.body(x)


# ----------------------------------------------------------------------------- Net
class Net(nn.Module):
    def __init__(self, encoder, mean, std, resize_to=None, n_classes=2):
        super().__init__()
        self.pre = Preproc(mean, std, resize_to)
        self.encoder = encoder
        self.out_dim = encoder.out_dim
        self.fc = nn.Linear(encoder.out_dim, n_classes)

    def features(self, x):
        return self.encoder(self.pre(x))

    def forward(self, x):
        return self.fc(self.features(x))


def default_init(backbone):
    return {"cnn4": "random", "resnet50": "imagenet", "vit_s16": "lunit_dino", "resnet18": "imagenet",
            "convnext_tiny": "imagenet", "vit_b16": "imagenet"}[backbone]


def input_res(backbone, probe=False):
    if probe or backbone in ("vit_s16", "vit_b16"):
        return 224
    return None  # native


def build_net(backbone="cnn4", init="random", mixstyle=None, probe=False, weights_dir=BACKBONE_DIR,
              n_classes=2) -> Net:
    """mixstyle: None or dict(p=..., alpha=...). probe=True -> 224 px input (frozen-probe rule)."""
    if backbone == "cnn4":
        assert init == "random", "cnn4 has no pretrained init (SimCLR weights are loaded by the arm)"
        enc = CNN4(mixstyle=mixstyle)
    elif backbone in ("resnet50", "resnet18"):
        enc = ResNetEnc(backbone, mixstyle=mixstyle)
        if init != "random":
            enc.load_init(os.path.join(weights_dir, WEIGHT_FILES[(backbone, init)]))
    elif backbone == "vit_s16":
        assert mixstyle is None, "MixStyle has no defined insertion point in a ViT"
        enc = TimmViTS()
        if init != "random":
            enc.load_init(os.path.join(weights_dir, WEIGHT_FILES[(backbone, init)]))
    elif backbone in ("convnext_tiny", "vit_b16"):
        enc = TVGeneric(backbone, init)
        enc.load_init(os.path.join(weights_dir, WEIGHT_FILES[(backbone, init)]))
    else:
        raise ValueError(backbone)
    mean, std = norm_for(init)
    return Net(enc, mean, std, resize_to=input_res(backbone, probe), n_classes=n_classes)


def n_params(m: nn.Module, trainable_only=False):
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only))


def bn_modules(model):
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
