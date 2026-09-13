#!/bin/bash
#SBATCH --partition=sc-freecpu
#SBATCH --account=default
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --job-name=th_vla_v8
set -euo pipefail

ATTEMPT=${1:?attempt number required}
CODE=/iris/u/jasonyan/repos/robosuite-toolhang-vla-20260911
ROOT=/iris/u/jasonyan/data/toolhang_vla_upgrade_v3_20260912/production_independent_v8_300
PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
export PYTHONPATH="$CODE:$CODE/humanlike_dataset_v2_20260907"
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$CODE"

REGIME=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["entries"][int(sys.argv[2])]["regime_assignment"])' "$ROOT/manifest.json" "$SLURM_ARRAY_TASK_ID")
"$PY" humanlike_dataset_v2_20260907/collect.py run \
  --human-version toolhang_vla_v6 --kind toolhang --root "$ROOT" \
  --index "${SLURM_ARRAY_TASK_ID}" --shards 360 --attempt "$ATTEMPT" \
  --retry-regime "$REGIME" --output-dir "attempt${ATTEMPT}"
