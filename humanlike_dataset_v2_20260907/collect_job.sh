#!/bin/bash
#SBATCH --partition=sc-freecpu
#SBATCH --account=default
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --job-name=human_v2
set -euo pipefail
ROOT=/iris/u/jasonyan/data/humanlike_dataset_v2_20260907
KIND=$1
MODE=$2
export PYTHONPATH="/iris/u/jasonyan/repos/robosuite-${KIND}-humanlike-v2-20260907:${ROOT}"
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
cd "/iris/u/jasonyan/repos/robosuite-${KIND}-humanlike-v2-20260907"
if [ "$MODE" = prepare ]; then
  "$PY" "$ROOT/collect.py" prepare --kind "$KIND" --root "$ROOT/pilot/$KIND" --seed 202609072 --pairs 60
else
  "$PY" "$ROOT/collect.py" run --kind "$KIND" --root "$ROOT/pilot/$KIND" --index "${SLURM_ARRAY_TASK_ID:-0}" --shards 60
fi
