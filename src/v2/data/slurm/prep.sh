#!/bin/bash
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:00:00
# usage: sbatch -J data_<cohort> -o <log> prep.sh <script.py> [args...]
# module load PyTorch/2.7.0-CONDA   # cluster-specific; or activate an environment from requirements.txt
cd "${SLURM_SUBMIT_DIR:-.}"
echo "host $(hostname) start $(date)"
python -u "$@"
echo "exit $? end $(date)"
