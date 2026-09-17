#!/bin/bash
#SBATCH --job-name=fork-native256
#SBATCH --partition=sc-freecpu
#SBATCH --account=default
#SBATCH --array=0-299%40
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --output=/iris/u/jasonyan/data/ycb_fork_vla300_variation_v2_20260917/logs/render_%A_%a.out
#SBATCH --error=/iris/u/jasonyan/data/ycb_fork_vla300_variation_v2_20260917/logs/render_%A_%a.err
set -euo pipefail
REPO=/iris/u/jasonyan/repos/robosuite-ycb-fork-v2-20260917
ROOT=/iris/u/jasonyan/data/ycb_fork_vla300_variation_v2_20260917
PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
export PYTHONPATH="$REPO" MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" "$REPO/robosuite/scripts/render_ycb_fork_native_images.py" --source "$ROOT/dataset_state_14hz.hdf5" --output-dir "$ROOT/native256_shards" --index "$SLURM_ARRAY_TASK_ID" --height 256 --width 256 --review "$ROOT/native256_review"
