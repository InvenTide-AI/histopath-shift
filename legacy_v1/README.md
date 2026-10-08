# Legacy v1 pipeline

This folder holds the earlier (v1) pipeline and its result tables. **It is not used for any result of the
article except the two case studies in the Discussion**, which are explicitly labeled as v1:

- the withdrawn single-hospital SimCLR × SAM synergy claim (`results/camelyon17_center2/`, `runs/camelyon17_center2/`,
  and the five-hospital grid in `records/v1_fivehospital_case_study.tar.xz`);
- the NCT-CRC-HE single-institution result (`results/nct_crc_he/`).

An audit of the v1 code found defects that the v2 pipeline (`src/v2/`) fixes: MixStyle stayed active at
evaluation; Macenko normalization applied one fixed global transform; the rotation-invariance arm was identical
to ERM; the IRM annealing schedule had no effect; MIDOG 2021 had no out-of-distribution validation split and a
non-disjoint ID-validation split; GroupDRO had an empty group; TENT ran on a label-sorted test stream. These
defects do not touch the four arms of the case studies (ERM, SAM, SimCLR, SimCLR+SAM), except that v1 SAM
updated batch-norm statistics on the perturbed pass; the article discloses this.

`README_v1_tmlr.md` is the original v1 README; its claims are superseded by the article.
`src/train_lib.py` is also imported by one v2 unit test that checks the compact CNN is unchanged.
