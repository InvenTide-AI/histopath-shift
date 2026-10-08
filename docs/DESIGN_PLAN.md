# MedIA revision plan (v2 pipeline)

Source of requirements: `TMLR_Comments.md`, `Guidance_Medical_Image_Anal.md`, `IEEETMI/Response_Editor.md`,
plus the code audit of 2026-10-05 (bugs listed in §1). Everything below is a spec for the v2 pipeline.
v1 records under `$HISTOPATH_DATA/{runs,midog_runs,canine_runs,canine_noscale_runs}`
are kept untouched; v2 lives under `V2=$HISTOPATH_DATA/v2`.

## 1. Defects in v1 that v2 must fix (all verified by the audit)

| # | Defect | v2 fix |
|---|---|---|
| D1 | MixStyle module not registered → stays in train mode at eval (stochastic test predictions); inserted before (not after) 2nd MaxPool | register as submodule; insert after stage-1 and stage-2 pooling (per Zhou et al.); deterministic at eval |
| D2 | "Macenko" uses the reference stain matrix for every tile → one fixed global colour transform | true per-tile Macenko (estimate HE per tile, normalise concentrations to reference; fall back to reference HE when the tile has too few tissue pixels) |
| D3 | RotInv identical to ERM (all arms already get uniform dihedral aug); "hard negatives" not implemented | drop the RotInv arm; say so in the paper |
| D4 | IRM anneal is a no-op (λ=1 throughout) | DomainBed-style: λ_pre=1 for `anneal_steps`, then λ_irm (tuned, e.g. 10/100), optimizer reset at anneal |
| D5 | MIDOG21 Xov == Xiv (no OOD val) and id_val not image-disjoint; canine id_val not slide-disjoint | id_val patient/slide/image-disjoint on every cohort; record which selection rules are available |
| D6 | GroupDRO phantom empty group on MIDOG/canine (site labels not 0..K-1) | remap training-site labels to 0..K_train-1 everywhere |
| D7 | Fish: BN buffers interpolated, inner momentum reset, no WD; selects epoch 1 | params-only meta-interpolation, buffers copied from inner model; inner opt per Shi et al.; meta-lr tuned; per-checkpoint logging lets us report every selection rule |
| D8 | TENT on Camelyon17: test set sorted by label → single-class batches → AUROC≈0.5 artifact | shuffle the test stream with a fixed seed before any test-time adaptation |
| D9 | Paper claims compute-matched / aug-only control scripts exist — they do not | implement and run both |
| D10 | probe_resnet50 records from two scripts with different schedulers/epochs | one probe script, one schedule, all backbones |
| D11 | C17 SSL pool can contain unselected patches of id_val patients | exclude id_val patients/slides/images from the unlabelled pool |

## 2. Design: 2 × 2 (task × shift type), four cohorts, leave-one-domain-out

| Cohort | Task | Domain = | Shift type | K | Val domain |
|---|---|---|---|---|---|
| Camelyon17-WILDS | tumour patch (lymph node) | hospital | institutional (scanner+stain+patients) | 5 | yes, (t-1) mod 5 |
| Canine cutaneous SCC multi-scanner | tumour vs non-tumour | scanner (same slides, slide-disjoint folds) | acquisition-only | 5 | yes, (t-1) mod 5 |
| MIDOG 2021 (`midog21sn`) | mitotic figure vs imposter | scanner (one lab), scale-normalised to constant physical extent | acquisition-only | 3 | no (ID-val only) |
| MIDOG++ | mitotic figure vs imposter | tumour type / lab / species | institutional + biological | 7 | yes, (t-1) mod 7 |

Common protocol on every cohort (removes the v1 schedule / selection confounds):
- Fixed optimisation budget: **1,200 SGD steps**, batch 128, cosine LR; checkpoint every 120 steps (10 checkpoints).
- Every checkpoint is evaluated on id_val, ood_val (if present) and ood_test; all per-checkpoint metrics + final predictions for the selected checkpoints are saved.
- **Primary model selection: training-domain (ID) validation** (DomainBed's default; available on all four cohorts).
  Secondary: leave-one-domain-out validation (C17, canine, MIDOG++). Last checkpoint reported as a third rule in the appendix. Oracle (test-selected) only as an upper bound in the appendix.
- Hyperparameters selected **per fold** by that fold's own ID-val AUROC (seed 0), never by test. (Pooling ID-val across folds
  would leak test-domain labels, because fold t's test domain is a training domain of other folds.) Tier S is tuned in two stages:
  base LR (ERM; ViT AdamW LR) first, then method grids at each fold's selected base LR.

## 3. Arms

### Tier C (compact CNN, 0.58 M params, 64 px) — **5 seeds**, all four cohorts
ERM; SAM; SimCLR→FT ("SimCLR", not "ours"); SimCLR+SAM; **Aug-only control** (SimCLR's view augmentations used as
supervised augmentation, no contrastive loss); **Compute-matched ERM** (same number of image forward/backward passes as
SimCLR pre-training + fine-tuning); GroupDRO; IRMv1; DeepCORAL; MixStyle; Fish; LISA; Macenko (per-tile);
H&E jitter; TIA-style (per-tile Macenko + H&E jitter); test-time: BN-adapt (on ERM, SimCLR, H&E jitter), TENT (on ERM).

### Tier S (standard scale, 96 px) — **3 seeds**, all four cohorts
- ResNet-50, ImageNet init, fine-tuned end to end: ERM, SAM, GroupDRO, IRMv1, DeepCORAL, MixStyle, Fish, LISA,
  Macenko, H&E jitter, TIA-style, SimCLR (continued SimCLR pre-training of the ImageNet RN50 on the fold's unlabelled
  pool, then FT), BN-adapt (on ERM).
- Pathology foundation models (Lunit, Kang et al. CVPR 2023; pretrained on TCGA + internal, no Camelyon/MIDOG data):
  - ResNet-50 Barlow Twins (`lunit_bt_rn50_ep200.torch`): frozen probe; fine-tuned ERM; fine-tuned + H&E jitter.
  - ViT-S/16 DINO (`lunit_dino_vit_small_patch16_ep200.torch`): frozen probe; fine-tuned ERM; fine-tuned + H&E jitter.
- ImageNet ResNet-50 frozen probe (96 px input, upsampled to 224 — same rule for all probes).
- Supplementary representation ladder (frozen probes, 3 seeds, one script): resnet18, resnet50, convnext_tiny, vit_b_16, vit_b_16_swag, Lunit BT/SwAV/MoCo RN50, Lunit DINO ViT-S.

### Tuning grid (seed 0, all folds, select by mean ID-val AUROC per cohort and tier)
GroupDRO η ∈ {1e-3, 1e-2, 1e-1}; IRM λ ∈ {1, 10, 100} (anneal at 1/4 of budget); CORAL λ ∈ {0.1, 1, 10};
MixStyle p ∈ {0.5, 1.0} × α ∈ {0.1, 0.3}; Fish meta-lr ∈ {0.01, 0.05, 0.1, 0.5}; LISA α ∈ {0.5, 2} × p_sel ∈ {0.5, 1.0};
SAM ρ ∈ {0.02, 0.05, 0.1}; H&E jitter strength ∈ {0.2, 0.4}. Tier-S base LR for ERM ∈ {1e-3, 3e-3, 1e-2} (SGD, momentum 0.9, wd 1e-4), chosen once per cohort and shared by all Tier-S arms; ViT fine-tune uses AdamW with LR ∈ {1e-5, 3e-5, 1e-4}.

## 4. Statistics (v2)
- Per arm and cohort, one-way random-effects model y_{ts} = μ + b_t + e_{ts} (domain t, seed s; seeds are NOT crossed — call it what it is).
  REML estimates; exact F-based 95% CI for ICC(1); Bayesian posterior (weakly-informative half-normal/half-Cauchy priors,
  grid integration) for σ_domain, σ_run and ICC, reported with credible intervals — handles boundary (zero) estimates.
- Paired contrasts across domains: mean Δ with t-interval, Holm correction across arms; fold-dependence sensitivity
  (correlation ρ ∝ training-set overlap, K_eff) following Nadeau & Bengio (2003) / Bates et al. (2023).
- Power: K needed from the SD of the seed-averaged per-domain *difference* (not the baseline's site SD), with and without K_eff correction.
- Rank stability: arm ranks with bootstrap (over seeds, and over domains) intervals; Kendall's τ between cohort rankings
  with bootstrap CIs; test whether ranking agreement clusters by task or by shift type (2×2).
- Response shape: replace r − r₀ with Oldham's correlation / Pitman–Morgan test (appendix).
- External example of the ICC diagnostic: DomainBed published per-environment results (parse from arXiv source of
  Gulrajani & Lopez-Paz 2021, never typed from memory).

## 5. Paper (Elsevier CAS, Medical Image Analysis)
- Frame as an evaluation-methodology contribution (MedIA requires a significant methodological contribution):
  protocol + variance-decomposition diagnostics + rank-stability analysis, demonstrated on a 2×2 task × shift design, two model scales, pathology FMs.
- Claims must follow the v2 tables exactly; no clinical-deployment recommendations from compact models.
- Related work: DomainBed, WILDS / WILDS 2.0, Wiles et al. 2022, Stacke et al. 2021, pathology FMs (UNI, CONCH, Virchow,
  Phikon, CTransPath, Lunit), stain normalisation (Reinhard, Macenko, Vahadane, StainGAN, Tellez 2019), FM robustness to centre differences.
- Main text includes the NCT-CRC-HE negative result and the withdrawn single-hospital synergy claim (as a case study).
- Deliverables: `paper/media/` (main.tex, main.bib, figs/), highlights, graphical abstract, cover letter, declarations.

## 6. Amendments (2026-10-05, after implementation review)
- MIDOG 2021 primary cache is `midog21sn` (scanner magnification differs by ~11%; canine and MIDOG++ are scale-normalised too).
  The un-normalised `midog21` cache is run at tier C as a sensitivity analysis.
- Near-border MIDOG/MIDOG++ objects are cut from a mirror-padded image (`pad_*` arrays allow excluding them in a sensitivity check).
- Canine unlabelled pool = random tissue crops from non-id-val training slides (not the labelled patches).
- C17 id-val holds out patients from every training centre.
- Run records gained `status` (ok/diverged), `cache_fingerprint`, `hp_source`, `id_val_by_domain`; final tags are `{arm}__selected__s{seed}`.
- Evaluation always in fp32 (training bf16 for tier S).
