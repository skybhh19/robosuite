#!/bin/bash
#SBATCH --partition=sc-freecpu
#SBATCH --account=default
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --job-name=th_wide_v3
set -euo pipefail

ATTEMPT=${1:?attempt number required}
CODE=/iris/u/jasonyan/repos/robosuite-toolhang-v2-wide-20260919
ROOT=/iris/u/jasonyan/data/toolhang_vla_wide_randomization_v3_20260919/production_unpaired_f100_p200_300
PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
export PYTHONPATH="$CODE:$CODE/humanlike_dataset_v2_20260907"
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$CODE"

REGIME=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["entries"][int(sys.argv[2])]["regime_assignment"])' "$ROOT/manifest.json" "$SLURM_ARRAY_TASK_ID")
"$PY" humanlike_dataset_v2_20260907/collect.py run \
  --human-version toolhang_vla_v2 --kind toolhang --root "$ROOT" \
  --index "$SLURM_ARRAY_TASK_ID" --shards 600 --attempt "$ATTEMPT" \
  --retry-regime "$REGIME" --output-dir "attempt$ATTEMPT"
