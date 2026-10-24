# Earlier (v1) pipeline: records for Supplementary Note S15

Only the files behind the case studies of Supplementary Note S15 are kept here. No other result of the article
uses them.

| File | Used for |
|---|---|
| `runs/camelyon17_center2/*.json` | the withdrawn single-hospital SimCLR × SAM synergy claim (one test hospital, one or two seeds per arm) |
| `results/camelyon17_center2/results_raw.csv` | the same runs as one table |
| `records/v1_fivehospital_case_study.tar.xz` | the v1 five-hospital grid (ERM, SAM, SimCLR, SimCLR+SAM × 3 seeds); unpacked by `make records`, checked by `src/v2/stats_tests/test_v1_regression.py` |
| `results/nct_crc_he/crc_results_raw.csv` | the NCT-CRC-HE single-institution result (S15.2) |
| `src/train_lib.py` | v1 model definition; `src/v2/tests/test_v2.py` checks that the compact CNN is unchanged |

An audit of the v1 code found defects that the v2 pipeline (`src/v2/`) fixes (Supplementary Note S15). They do
not touch the four arms of the case studies, except that v1 SAM updated batch-norm statistics on the perturbed
pass.
