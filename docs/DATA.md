# Data and pretrained weights

Nothing below is redistributed in this repository. Download each item yourself and accept its licence.
All paths are relative to `$HISTOPATH_DATA` (default `./data`).

## Cohorts

| Cohort | Source | Licence | Expected layout |
|---|---|---|---|
| Camelyon17-WILDS (Koh et al., 2021; Bándi et al., 2019) | `pip install wilds`, then `python -c "from wilds import get_dataset; get_dataset('camelyon17', download=True, root_dir='$HISTOPATH_DATA/wilds')"` | CC0 1.0 | `wilds/camelyon17_v1.0/{metadata.csv,patches/}` |
| Canine cutaneous SCC, 44 slides × 5 scanners (Wilm et al., 2023) | Zenodo [10.5281/zenodo.7418555](https://doi.org/10.5281/zenodo.7418555); `python scripts/fetch_zenodo.py 7418555 $HISTOPATH_DATA/canine_scc` | CC BY 4.0 | `canine_scc/{scc_NN_<scanner>.tif, scc.json}` |
| MIDOG 2021 training set (Aubreville et al., 2023) | Zenodo [10.5281/zenodo.4643381](https://doi.org/10.5281/zenodo.4643381); `python scripts/fetch_midog.py` (writes `./data/midog`) | CC BY 4.0 | `midog/{001..200.tiff, MIDOG.json}` |
| MIDOG++ (Aubreville et al., Sci. Data 2023) | figshare collection [10.6084/m9.figshare.c.6615571.v1](https://doi.org/10.6084/m9.figshare.c.6615571.v1); `python src/v2/data/fetch_midogpp.py` (reuses the MIDOG 2021 images, verifies md5) | collection CC BY 4.0, files CC0 1.0 | `midogpp/{*.tiff, MIDOG++.json, image_meta.json}` |
| PatchCamelyon test split (Veeling et al., 2018), secondary check only | Zenodo [10.5281/zenodo.2546921](https://doi.org/10.5281/zenodo.2546921) | CC0 1.0 | `pcam/camelyonpatch_level_2_split_test_{x,y}.h5` |
| NCT-CRC-HE-100K, CRC-VAL-HE-7K (Kather et al.), v1 case study only | Zenodo [10.5281/zenodo.1214456](https://doi.org/10.5281/zenodo.1214456) | CC BY 4.0 | see `legacy_v1/` |

## Pretrained weights

`python scripts/fetch_backbones.py` writes all of them to `$HISTOPATH_DATA/backbone/` and checks the hashes.

| File | Model | Source | Licence |
|---|---|---|---|
| `resnet18.pth`, `resnet50.pth`, `convnext_tiny.pth`, `vit_b_16.pth` | ImageNet-1k (torchvision `IMAGENET1K_V1`; ResNet-50 `IMAGENET1K_V2`) | download.pytorch.org (hash prefixes f37072fd, 11ad3fa6, 983f1562, c867db91) | see the torchvision model documentation |
| `vit_b_16_swag.pth` | SWAG ViT-B/16, IG-3.6B then ImageNet-1k (`IMAGENET1K_SWAG_E2E_V1`) | download.pytorch.org (9ac1b537) | CC BY-NC 4.0 |
| `lunit_{bt,swav,mocov2}_rn50_ep200.torch`, `lunit_dino_vit_small_patch16_ep200.torch` | Kang et al., CVPR 2023, pre-trained on 19 M TCGA patches | github.com/lunit-io/benchmark-ssl-pathology, release `pretrained-weights` | Lunit Inc. Public License: non-commercial research only |

The Lunit normalization constants used for every Lunit model are those of the release notes
(mean 0.7032, 0.5361, 0.6610; std 0.2172, 0.2608, 0.2072).
