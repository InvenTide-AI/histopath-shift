"""Download the pretrained weights used in the paper into $HISTOPATH_DATA/backbone/.

ImageNet / SWAG weights are the torchvision releases whose hashes match the files used for every run
(resnet18-f37072fd, resnet50-11ad3fa6, convnext_tiny-983f1562, vit_b_16-c867db91, vit_b_16_swag-9ac1b537).
Lunit weights (Kang et al., CVPR 2023) come from the GitHub release of lunit-io/benchmark-ssl-pathology and are
licensed for NON-COMMERCIAL research use only (Lunit Inc. Public License). SWAG weights are CC BY-NC 4.0.

    python scripts/fetch_backbones.py            # all
    python scripts/fetch_backbones.py --skip-lunit
"""
import argparse
import hashlib
import os
import re
import urllib.request

import torchvision.models as tvm

TV = {"resnet18.pth": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1),
      "resnet50.pth": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2),
      "convnext_tiny.pth": (tvm.convnext_tiny, tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
      "vit_b_16.pth": (tvm.vit_b_16, tvm.ViT_B_16_Weights.IMAGENET1K_V1),
      "vit_b_16_swag.pth": (tvm.vit_b_16, tvm.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)}
LUNIT = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights/"
LUNIT_FILES = ["lunit_bt_rn50_ep200.torch", "lunit_swav_rn50_ep200.torch", "lunit_mocov2_rn50_ep200.torch",
               "lunit_dino_vit_small_patch16_ep200.torch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.environ.get("HISTOPATH_DATA", "data"), "backbone"))
    ap.add_argument("--skip-lunit", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, (fn, w) in TV.items():
        p = os.path.join(a.out, name)
        if not os.path.exists(p):
            # the raw torchvision checkpoint file, byte-identical to the one used in the paper; torchvision
            # file names end in the first 8 hex digits of the file's SHA-256
            urllib.request.urlretrieve(w.url, p)
            want = re.search(r"-([0-9a-f]{8})\.pth$", w.url).group(1)
            got = hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]
            if got != want:
                os.remove(p)
                raise SystemExit(f"hash mismatch for {name}: {got} != {want}")
            print("wrote", p)
    if not a.skip_lunit:
        for f in LUNIT_FILES:
            p = os.path.join(a.out, f)
            if not os.path.exists(p):
                urllib.request.urlretrieve(LUNIT + f, p)
                print("wrote", p)


if __name__ == "__main__":
    main()
