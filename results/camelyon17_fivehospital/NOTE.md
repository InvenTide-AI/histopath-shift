# Five-hospital sweep — aggregated tables only

`results_folds.csv` is the per-run record of the 20 runs (4 arms x 5 held-out
hospitals, one seed each) that produce Table II and Fig. 1 of the paper. The
individual run JSON files those rows were aggregated from were not retained, so
unlike the center-2 experiment there is no `runs/` directory for this sweep.

Every published quantity for this experiment is recoverable from the three CSVs
here; `src/check_release.py` verifies that. What cannot be recovered from them is
anything below the selected epoch — per-epoch curves, per-sample predictions,
and calibration bins are not available for the sweep.
