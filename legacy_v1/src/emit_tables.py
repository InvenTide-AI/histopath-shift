"""Emit the LaTeX table bodies for the per-arm appendix tables.

Reads results/rules/synthesis.json and prints one block of rows per cohort,
ordered by worst-domain AUROC. Keeps the manuscript's appendix tables and the
aggregated records in sync without hand transcription.
"""
from __future__ import annotations
import json

NAME = {
    "baseline": "ERM baseline",
    "bnadapt": "BN-adapt",
    "bnadapt-on-ssl": "BN-adapt on SSL",
    "bnadapt-on-erm_henorm": r"BN-adapt on H\&E jitter",
    "tent": "TENT",
    "coral": "DeepCORAL",
    "erm_henorm": r"ERM + H\&E jitter",
    "fish": "Fish",
    "groupdro": "GroupDRO",
    "irm": "IRMv1",
    "lisa": "LISA",
    "macenko": "Macenko + ERM",
    "mixstyle": "MixStyle",
    "probe_convnext_tiny": "Probe: ConvNeXt-T",
    "probe_resnet18": "Probe: ResNet18",
    "probe_resnet50": "Probe: ResNet50",
    "probe_vit_b_16": "Probe: ViT-B/16",
    "probe_vit_b_16_swag": "Probe: ViT-B/16 SWAG",
    "probe_vit_l_16_swag": "Probe: ViT-L/16 SWAG",
    "rotinv": "RotInv",
    "sam": "SAM",
    "ssl": "SSL (ours)",
    "ssl_sam": "SSL+SAM (ours)",
    "tia_style": "TIA-style",
}
COHORTS = ["camelyon17", "midog", "canine", "canine_noscale"]


def main() -> None:
    syn = json.load(open("results/rules/synthesis.json"))
    for cohort in COHORTS:
        arms = syn[cohort]["arms"]
        print("%% ---- %s (K=%d) ----" % (cohort, syn[cohort]["n_domains"]))
        for arm, v in sorted(arms.items(), key=lambda kv: -kv[1]["worst"]):
            print(r"%-26s & $%.3f$ & $%.3f$ & $%.3f$ & $%.3f$ & $%+.3f$ & %d \\"
                  % (NAME.get(arm, arm), v["worst"], v["sd"], v["mean"],
                     v["ece"], v["gain_mean"], v["n_seeds"]))
        print()


if __name__ == "__main__":
    main()
