#!/bin/bash
# Unpack the run records shipped in records/ into $HISTOPATH_DATA/v2 (default: ./data/v2).
#   snap_final2_20261005/{runs,runs_tune}    the frozen snapshot behind every table and figure (id 1381bdd759f9)
#   runs_quarantine_bt_lr, runs_quarantine_macenko_prefix, runs_diag   records used only in the appendices
set -euo pipefail
cd "$(dirname "$0")/.."
( cd records && sha256sum -c SHA256SUMS --ignore-missing )
V2="${HISTOPATH_DATA:-data}/v2"
SNAP="$V2/snap_final2_20261005"
mkdir -p "$SNAP"
tar -xJf records/v2_final_runs.tar.xz -C "$SNAP"
tar -xJf records/v2_tuning_runs.tar.xz -C "$SNAP"
cp records/SNAPSHOT.json "$SNAP/"
# minimal test predictions (labels + ID-selected probabilities) used by the site acceptance test
for t in C S; do tar -xf "records/v2_test_predictions_$t.tar" -C "$SNAP"; done
for a in v2_quarantine_bt_lr v2_quarantine_macenko_prefix v2_diagnostics_macenko; do
  tar -xJf "records/$a.tar.xz" -C "$V2"
done
# v1 records behind the two case studies and the v1 regression test (src/v2/stats_tests/test_v1_regression.py)
tar -xJf legacy_v1/records/v1_fivehospital_case_study.tar.xz -C "${HISTOPATH_DATA:-data}"
echo "records unpacked under $V2 ($(find "$SNAP/runs" -name '*.json' | wc -l) final, $(find "$SNAP/runs_tune" -name '*.json' | wc -l) tuning)"
