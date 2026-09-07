# Incomplete third task — not reported in the paper

A chest-radiograph arm was started as a third task and abandoned before any arm
completed the full seed grid. `cxr_partial_results_recovered.csv` holds the runs
that finished, recovered from partial logs.

These numbers are **not** in the manuscript and should not be cited or compared
against the two completed tasks: the arms are unbalanced in seeds, the epoch
selection protocol used for the reported experiments was not applied, and no
bootstrap intervals were computed. The files are kept for provenance — so that
the scripts in `src/*_cxr.py` are not orphaned — and for nothing else.
