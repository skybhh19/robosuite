#!/bin/bash
#SBATCH --partition=sc-freecpu
#SBATCH --account=default
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=human400_v6_render
set -euo pipefail
ROOT=/iris/u/jasonyan/data/humanlike_dataset_v2_20260907
TARGET="$ROOT/production_v6/threading"
export PYTHONPATH="/iris/u/jasonyan/repos/robosuite-threading-humanlike-v4-20260907:${ROOT}"
export MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1
cd /iris/u/jasonyan/repos/robosuite-threading-humanlike-v4-20260907
/iris/u/jasonyan/miniforge3/envs/openx/bin/python "$ROOT/render.py" \
  --root "$TARGET" --index "$SLURM_ARRAY_TASK_ID"
