#!/bin/bash
#SBATCH --partition=sc-freecpu
#SBATCH --account=default
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --job-name=human400_v6
set -euo pipefail
ROOT=/iris/u/jasonyan/data/humanlike_dataset_v2_20260907
TARGET="$ROOT/production_v6/threading"
MODE=$1
export PYTHONPATH="/iris/u/jasonyan/repos/robosuite-threading-humanlike-v4-20260907:${ROOT}"
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
cd /iris/u/jasonyan/repos/robosuite-threading-humanlike-v4-20260907
if [ "$MODE" = prepare ]; then
  "$PY" "$ROOT/collect.py" prepare --production --human-version human_v6 --kind threading \
    --root "$TARGET" --seed 202609079 --pairs 260
else
  "$PY" "$ROOT/collect.py" run --human-version human_v6 --kind threading \
    --root "$TARGET" --index "${SLURM_ARRAY_TASK_ID:-0}" --shards 260
fi
