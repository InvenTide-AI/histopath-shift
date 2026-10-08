# Claim trace for `sections/related.tex`

Each row is one factual claim about a cited work, with the key and where the claim is supported in the source. All sources were read on 2026-10-05, from the arXiv e-print LaTeX or PDF, from PubMed/Europe PMC abstracts or full text, from Crossref abstracts, or from the authors' PDFs. Numbers come from the stated table or sentence and were not recalled from memory.

## 1. Site and center effects

| Claim (paraphrase of the sentence in related.tex) | Key | Source and location |
|---|---|---|
| Pneumonia CNNs trained on radiographs from 3 hospital systems; internal > external performance in 3 of 5 comparisons | zech2018variable | https://pubmed.ncbi.nlm.nih.gov/30399157/ (abstract, Background and Conclusion) |
| CNNs identified the hospital system for >99.9% of radiographs from two systems (99.95% NIH, 99.98% MSH) | zech2018variable | same abstract, Methods and Findings |
| Systematic review of 86 algorithms (83 studies): 81% (70/86) showed a decrease on external data; 24% (21/86) a decrease of ≥0.10 on the unit scale | yu2022external | https://pubmed.ncbi.nlm.nih.gov/35652114/ (abstract, Results) |
| Submitting site of TCGA slides identifiable by DL despite common color normalization and augmentation; site signatures bias survival, mutation and stage prediction; proposed splits with no site in both training and validation (quadratic programming) | howard2021site | Crossref abstract, https://doi.org/10.1038/s41467-021-24698-1 |
| Proposed a model-specific measure of the distance between domains in the learned representation | stacke2019closer | https://arxiv.org/abs/1909.11575 (abstract) |
| "Representation shift" correlates highly with the drop in performance across many types of domain shift | stacke2021measuring | https://pubmed.ncbi.nlm.nih.gov/33085623/ (abstract) |

## 2. Domain generalization benchmarks and protocols

| Claim | Key | Source and location |
|---|---|---|
| Survey of DG | zhou2023domain | https://arxiv.org/abs/2103.02503 (abstract) |
| DomainBed: 14 algorithms, 7 multi-domain datasets, 3 model-selection criteria | gulrajani2021domainbed | ICLR 2021 camera-ready (OpenReview lQdXeXDoWtI, read via web.archive.org), Sec. 4 "fourteen algorithms"; arXiv v1 (the only arXiv version) says nine, which is superseded by the cited ICLR version |
| Every configuration trained with one domain hidden for testing | gulrajani2021domainbed | Sec. 5 (Experiments), first paragraph: "We consider all configurations of a dataset where we hide one domain for testing and train on the remaining ones." |
| Entire study repeated 3 times, redrawing hyperparameters, weight initializations and dataset splits | gulrajani2021domainbed | Sec. 5, paragraph "Standard error bars" |
| A DG algorithm without a model-selection strategy is incomplete | gulrajani2021domainbed | abstract; Sec. 3 ("...remains incomplete") |
| Three rules: training-domain validation, leave-one-domain-out cross-validation, test-domain (oracle) validation | gulrajani2021domainbed | Sec. 3.1 "Three model selection methods" |
| With training-domain or oracle selection no algorithm beats ERM's average by more than one point; authors report no improvement significant (t-test, alpha=0.05); under leave-one-domain-out CORAL leads ERM by 1.8 points | gulrajani2021domainbed | ICLR version Sec. 5.1 "Claim 2" and Table 3: averages ERM/best = 66.6/67.5 (CORAL, training-domain), 63.2/65.0 (CORAL, leave-one-domain-out), 68.7/69.2 (CORAL, oracle). The paper's own wording "given any model selection criterion ... more than one point" is contradicted by its Table 3 for leave-one-domain-out, so the text no longer repeats it |
| Training-domain validation outperforms leave-one-domain-out cross-validation across multiple datasets and algorithms | gulrajani2021domainbed | Sec. 5.1, paragraph "Model selection methods matter" |
| Camelyon17-WILDS: 3 training hospitals, OOD validation hospital 4, test hospital 5, chosen because its patches were most visually distinctive | koh2021wilds; bandi2019camelyon17 | https://arxiv.org/abs/2012.07421v3 Appendix E.2.1 "Data" (p. 72) |
| ERM test (OOD) accuracy SD 6.4 across 10 seeds | koh2021wilds | arXiv v3 App. E.2.2 (p. 73) and Table 5 (ERM test 70.3 (6.4)) |
| CORAL, IRM and GroupDRO comparable to or worse than ERM | koh2021wilds | App. E.2.2 "Additional baseline methods"; Table 5 (59.5, 64.2, 68.4 vs 70.3) |
| WILDS asks for 10 seeds for Camelyon17 submissions | koh2021wilds | App. E.2.2 "Discussion" (p. 74) |
| Warning against overfitting to the few test domains (one OOD test hospital in Camelyon17) | koh2021wilds | Sec. 9 "Guidelines for method developers", subsection "Avoiding overfitting to the test distribution" (p. 29) |
| WILDS 2.0 adds unlabeled patches from the training, validation and test hospitals | sagawa2022extending | https://arxiv.org/abs/2112.05090v2 Sec. 4, Camelyon17 paragraph ("unlabeled patches from train and test hospitals"); Appendix Table 4 (1,799,247 source / 600,030 validation-OOD / 600,030 target unlabeled) |
| ERM 70.8% OOD without augmentation, 82.0% with it | sagawa2022extending | Table 2 (p. 10), Camelyon17 block (ERM (-data aug) 70.8 (7.2); ERM 82.0 (7.4)); Sec. 6.2 "Image classification" paragraph. Augmentation = RandAugment + Cutout (Sec. 6.1); footnote says it includes color jitter |
| SwAV on unlabeled target-hospital patches: 91.4% (SD 2.0); Noisy Student 86.7% | sagawa2022extending | Table 2, Camelyon17 block, column header "(Unlabeled target, avg acc)"; SwAV 91.4 (2.0), Noisy Student 86.7 (1.7) |
| CORAL, DANN, Pseudo-Label, FixMatch below augmented ERM | sagawa2022extending | Table 2 (77.9, 68.4, 67.7, 71.0 vs 82.0); Sec. 6.2 ("the other methods ... underperformed ERM") |
| Use of target-domain unlabeled data = unsupervised adaptation | sagawa2022extending | Title ("...for Unsupervised Adaptation"); Sec. 6.1 "Unlabeled data" ("we used unlabeled target data to allow methods to directly adapt to the target distribution") |
| Wiles et al.: spurious correlation, low-data drift, unseen data shift | wiles2022fine | https://arxiv.org/abs/2110.11328v2 Sec. 1 (Introduction) and Sec. 2.2 "Distribution Shifts" |
| >85K models, 19 methods; six datasets including Camelyon17 | wiles2022fine | abstract; Sec. 4 (Experiments), dataset list |
| Pre-training and augmentation often give large gains; the best methods are inconsistent across datasets and shifts | wiles2022fine | abstract; Takeaway 1 (Sec. 4.1) |
| On Camelyon17, pre-training not best under unseen data shift (augmentation or DANN best) | wiles2022fine | Sec. 4.1, Takeaway 1 |
| On Camelyon17, color jitter harms performance | wiles2022fine | Sec. 4.1, Takeaway 3 |
| Leave-one-domain-out evaluation and training-domain selection are not claimed as new; our design (variance decomposition, rank stability, 2×2 task × shift) | (own work) | revision/PLAN.md §2, §4 |

## 3. Stain normalization and augmentation

| Claim | Key | Source and location |
|---|---|---|
| Stain differences between labs degrade CNNs on unseen labs | tellez2019stain | https://arxiv.org/abs/1902.06543 (abstract) |
| Reinhard: match per-axis mean and SD in decorrelated lαβ space | reinhard2001color | Author PDF https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf, sections "Decorrelated color space" and "Statistics and color correction" ("we compute the means and standard deviations for each axis separately in lαβ space") |
| Color deconvolution separates up to three stains using stain-specific RGB absorption | ruifrok2001deconvolution | https://pubmed.ncbi.nlm.nih.gov/11531144/ (abstract, Study design and Conclusion) |
| HED augmentation is built on color deconvolution with a fixed matrix | tellez2019stain | arXiv 1902.06543, Sec. "Methods", "Stain color augmentation", paragraph "Hematoxylin-Eosin-DAB (HED)" |
| Macenko: per-slide stain vectors from OD; projection onto the plane of the two largest singular vectors; robust angle percentiles; intensities scaled to a common 99th-percentile pseudo-maximum | macenko2009 | Author PDF https://www.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf, Sec. 3 and Algorithm 1 (p. 1108–1109); Sec. 4 "Intensity variation and correction" |
| Vahadane: sparse non-negative stain density maps combined with a target's stain color basis, preserving structure | vahadane2016structure | https://pubmed.ncbi.nlm.nih.gov/27164577/ (abstract) |
| StainGAN: CycleGAN-inspired, end to end, no expert-chosen single reference slide needed (still needs a target-domain image set) | shaban2019staingan | https://arxiv.org/abs/1804.01601 (abstract) |
| Tellez 2018: H&E channels perturbed directly; with ensembling, a stain-invariant mitosis detector | tellez2018whole | https://arxiv.org/abs/1808.05896 (abstract) |
| Tellez 2019: 4 tasks and 9 labs; HSV/HED color augmentation essential; top configuration without normalization; normalization makes results less sensitive to the augmentation choice | tellez2019stain | arXiv 1902.06543 abstract; Table 1 (ranking); Sec. "Experimental results", subsections "Effects of stain color augmentation" and "Effects of stain color normalization"; Conclusion |
| Site still detectable after normalization | howard2021site | Crossref abstract (see Section 1 above) |
| Our arms (per-tile Macenko, H&E jitter, combination) | (own work) | PLAN.md §3 (Tier C/S arm list; D2 fix) |

## 4. Self-supervision and pathology foundation models

| Claim | Key | Source and location |
|---|---|---|
| SimCLR on 57 unlabeled histopathology datasets; linear classifiers beat ImageNet by >28% F1 on average | ciga2022ssl | https://arxiv.org/abs/2011.13971 (abstract) |
| Kang et al.: MoCo v2, SwAV, Barlow Twins, DINO; 19M patches from 20,994 TCGA WSIs; domain-aligned > ImageNet in linear and fine-tuning evaluation | kang2023benchmarking | https://arxiv.org/abs/2212.04690 abstract; Sec. 4 "Experiment setup" (Table 1: TCGA 20,994 WSIs, 19,000,069 patches; "All experiments, unless specified otherwise, present the results of pre-training on the TCGA dataset only"; method/architecture paragraph) |
| CTransPath: CNN + multi-scale Swin Transformer | wang2022ctranspath | https://pubmed.ncbi.nlm.nih.gov/35952419/ (abstract) |
| Phikon: iBOT ViT-Base, >40M images, 16 cancer types | filiot2023phikon | medRxiv abstract via Crossref, https://doi.org/10.1101/2023.07.21.23292757 |
| UNI: >100M images, >100,000 WSIs, 20 tissue types | chen2024uni | https://pubmed.ncbi.nlm.nih.gov/38504018/ (Nat Med abstract) |
| CONCH: >1.17M image–caption pairs | lu2024conch | https://pubmed.ncbi.nlm.nih.gov/38504017/ (Nat Med abstract) |
| Virchow: 632M-parameter ViT (ViT-H/14), DINOv2, ~1.5M WSIs (1,488,550) | vorontsov2024virchow | Nat Med full text, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11485232/ (Results, first section; Methods "Million-scale training dataset" and "Virchow architecture and training") |
| Prov-GigaPath: 1.3B tiles, 171,189 slides, Providence network of 28 cancer centres | xu2024gigapath | Crossref abstract, https://doi.org/10.1038/s41586-024-07441-w |
| FM embeddings contain hospital signatures that dominate feature-space distances and are not removed by stain normalization | komen2024batch | https://arxiv.org/abs/2411.05489 (abstract) |
| 10 public FMs all strongly represent the medical center; only one has robustness index >1, and only slightly | dejong2025unrobust | https://arxiv.org/abs/2501.18055 (abstract; Sec. "Robustness Index") |
| PathoROB: robustness deficits in all 20 FMs; more robust FMs, vision-language alignment and post-hoc robustification reduce but do not eliminate the risk of errors | komen2026robust | https://pubmed.ncbi.nlm.nih.gov/42277006/ (Nat Commun 2026 abstract) |
| We test frozen and fine-tuned Lunit encoders under the common protocol | kang2023benchmarking; (own work) | PLAN.md §3 Tier S |

## 5. Test-time adaptation

| Claim | Key | Source and location |
|---|---|---|
| AdaBN modulates BN statistics for the target domain; parameter-free | li2017adabn | https://arxiv.org/abs/1603.04779 (abstract; arXiv version of the Pattern Recognition paper) |
| BN statistics re-estimated on corrupted images improve ImageNet-C robustness for 25 models; adapted results should be reported in OOD settings | schneider2020bn | https://arxiv.org/abs/2006.16971 (abstract) |
| Prediction-time BN improves accuracy and calibration under covariate shift; mixed with pre-training; weaker under more natural shift | nado2020bn | https://arxiv.org/abs/2006.10963 (abstract) |
| TENT: test entropy minimization; estimates normalization statistics and optimizes channel-wise affine parameters online per batch | wang2021tent | https://arxiv.org/abs/2006.10726 (abstract) |
| Test stream shuffled with a fixed seed before adaptation | (own work) | PLAN.md §1, D8 |

## 6. Variance, rankings and statistical reporting

| Claim | Key | Source and location |
|---|---|---|
| Data sampling, parameter initialization and hyperparameter choice markedly affect benchmark results | bouthillier2021variance | https://arxiv.org/abs/2103.03098 (abstract) |
| Test-set scores alone are insufficient for deciding which model is best | dodge2019show | https://arxiv.org/abs/1909.03004 (abstract) |
| Challenge ranks are generally not robust to test data, ranking scheme and annotators | maierhein2018rankings | https://pubmed.ncbi.nlm.nih.gov/30523263/ (abstract) |
| challengeR: bootstrap analysis of ranking stability | wiesenfarth2021challenger | https://arxiv.org/abs/1910.05121, Sec. "Bootstrap approach" (methods for ranking stability); https://pubmed.ncbi.nlm.nih.gov/33504883/ |
| CV error bars are large (≈±10% for 100 samples); the standard error across folds strongly underestimates them | varoquaux2018cv | https://pubmed.ncbi.nlm.nih.gov/28655633/ (abstract) |
| In two Kaggle medical-imaging challenges with test sets <1,000, the public–private difference exceeded the winner vs top-10% gap | varoquaux2022failures | https://arxiv.org/abs/2103.10292v2 Sec. 4.1 "Evaluation error is often larger than algorithmic improvements" and the Kaggle figure ("For two challenges ... with test set sizes below 1000 ... larger than improving 10% on the leaderboard"). The published npj Digit Med version (doi:10.1038/s41746-022-00592-y) has the same section |
| Every variance estimator based only on CV results is biased; ignoring training-set variability can lead to false conclusions of significance | nadeau2003inference | Springer PDF https://link.springer.com/content/pdf/10.1023/A:1024068626366.pdf, abstract (p. 239) |
| CV estimates the average error of models fit to other training sets from the same population (proved for OLS linear models; shown empirically more broadly); fold correlation makes the usual variance/intervals too small | bates2023cv | https://arxiv.org/abs/2104.00673 (abstract) |
| K = 3–7 domains; fold-dependence sensitivity; bootstrap rank intervals; Kendall's τ CIs | (own work) | PLAN.md §2 (K per cohort), §4 |

Notes
- Citation keys whose printed year differs from the key: li2017adabn → 2018, bates2023cv → 2024.
- demsar2006statistical was verified (JMLR abstract: Wilcoxon and Friedman tests) but cut for length, so it is not cited in this section.
- WILDS (koh2021wilds) appendix and table numbers follow arXiv v3; the ICML proceedings appendix may number them differently. The text therefore cites the claim without a table pointer. For WILDS 2.0, Table 2 is the arXiv v2 numbering (Table 1 there is the domain-type table in Sec. 3).


## Added 2026-10-06 (MIDOG paragraph in Sec. 2.1; source: abstracts via Semantic Scholar API, DOIs in main.bib)
- aubreville2023midog (10.1016/j.media.2022.102699, arXiv 2204.03742): "training set of 200 cases, split across four scanning systems"; "test set, an additional 100 cases split across four scanning systems, including two previously unseen scanners"; "winning algorithm yielded an F1 score of 0.748 (CI95: 0.704-0.781)".
- aubreville2024midog2022 (10.1016/j.media.2024.103155, arXiv 2309.15589): "evaluated the algorithmic approaches ... provided by nine challenge participants on ten independent domains"; "domain characteristics not present in the training set (feline as new species, spindle cell shape as new morphology and a new scanner) led to small but significant decreases in performance".
- jahanifar2024mitosis (10.1016/j.media.2024.103132, arXiv 2208.12587): "two-stage mitosis detection framework"; "incorporating domain generalization methods"; "winning the two mitosis domain generalization challenge contests (MIDOG21 and MIDOG22)".
- aubreville2023midogpp (10.1038/s41597-023-02327-4): "region of interest images from 503 histological specimens of seven different tumor types"; "In a leave-one-domain-out setting, generalizability improved considerably" (relative to single-domain training).
- wilm2023canine: five scanners, same slides; consistent with the dataset as used here (44 slides x 5 scanners; sections/setup.tex). Abstract not retrievable via API; claim limited to the dataset design.
