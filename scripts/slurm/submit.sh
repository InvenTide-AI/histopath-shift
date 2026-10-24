#!/bin/bash
# Submit a plan's shard list as one SLURM array:  scripts/slurm/submit.sh SHARDS_FILE [sbatch args...]
set -euo pipefail
SHARDS=${1:?usage: submit.sh SHARDS_FILE [sbatch args...]}; shift
N=$(grep -c . "$SHARDS")
[ "$N" -gt 0 ] || { echo "no shards in $SHARDS"; exit 1; }
cd "${SLURM_SUBMIT_DIR:-.}"   # run from the repository root
sbatch --array=0-$(( N - 1 )) "$@" scripts/slurm/run_plan.sh "$SHARDS"
