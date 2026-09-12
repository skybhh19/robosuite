#!/bin/bash
#SBATCH --partition=sc-freecpu
#SBATCH --account=default
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --job-name=th_vla_v7
set -euo pipefail

ATTEMPT=${1:?attempt number required}
CODE=/iris/u/jasonyan/repos/robosuite-toolhang-vla-20260911
ROOT=/iris/u/jasonyan/data/toolhang_vla_upgrade_v3_20260912/production_moderate_v7_300
PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
export PYTHONPATH="$CODE:$CODE/humanlike_dataset_v2_20260907"
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$CODE"

"$PY" humanlike_dataset_v2_20260907/collect.py run \
  --human-version toolhang_vla_v6 --kind toolhang --root "$ROOT" \
  --index "${SLURM_ARRAY_TASK_ID}" --shards 180 --attempt "$ATTEMPT" \
  --output-dir "attempt${ATTEMPT}"
