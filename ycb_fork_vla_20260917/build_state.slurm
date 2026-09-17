#!/bin/bash
#SBATCH --job-name=fork-build-state
#SBATCH --partition=sc-freecpu
#SBATCH --account=default
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/iris/u/jasonyan/data/ycb_fork_vla300_variation_v2_20260917/logs/build_state_%j.out
#SBATCH --error=/iris/u/jasonyan/data/ycb_fork_vla300_variation_v2_20260917/logs/build_state_%j.err
set -euo pipefail
REPO=/iris/u/jasonyan/repos/robosuite-ycb-fork-v2-20260917
ROOT=/iris/u/jasonyan/data/ycb_fork_vla300_variation_v2_20260917
PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
export PYTHONPATH="$REPO" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" "$REPO/robosuite/scripts/build_ycb_fork_training_hdf5.py" --root "$ROOT" --output "$ROOT/dataset_state_20hz.hdf5"
"$PY" "$REPO/robosuite/scripts/resample_ycb_fork_state_14hz.py" --source "$ROOT/dataset_state_20hz.hdf5" --output "$ROOT/dataset_state_14hz.hdf5" --audit "$ROOT/resample_14hz.json"
