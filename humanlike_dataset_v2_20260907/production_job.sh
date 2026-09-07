#!/bin/bash
#SBATCH --partition=sc-freecpu
#SBATCH --account=default
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --job-name=human400
set -euo pipefail
ROOT=/iris/u/jasonyan/data/humanlike_dataset_v2_20260907
KIND=$1
MODE=$2
PAIRS=$3
export PYTHONPATH="/iris/u/jasonyan/repos/robosuite-${KIND}-humanlike-v4-20260907:${ROOT}"
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
cd "/iris/u/jasonyan/repos/robosuite-${KIND}-humanlike-v4-20260907"
if [ "$MODE" = prepare ]; then
  "$PY" "$ROOT/collect.py" prepare --production --human-version human_v4 --kind "$KIND" \
    --root "$ROOT/production/$KIND" --seed 202609073 --pairs "$PAIRS"
elif [ "$MODE" = retry_full ] || [ "$MODE" = retry_partial ]; then
  REGIME=${MODE#retry_}
  "$PY" "$ROOT/collect.py" run --human-version human_v4 --kind "$KIND" \
    --root "$ROOT/production/$KIND" --index "${SLURM_ARRAY_TASK_ID:-0}" --shards "$PAIRS" \
    --attempt 1 --output-dir retry1 --retry-regime "$REGIME" \
    --failed-from "$ROOT/production/$KIND/trials"
elif [ "$MODE" = retry_audit ]; then
  "$PY" "$ROOT/audit_retry_completion.py" --root "$ROOT/production/$KIND" --version human_v4
elif [ "$MODE" = repair_chunks ]; then
  "$PY" "$ROOT/flatten_action_chunks.py" --root "$ROOT/production/$KIND"
  "$PY" "$ROOT/fix_version_metadata.py" --root "$ROOT/production/$KIND"
  "$PY" "$ROOT/validate_final.py" --root "$ROOT/production/$KIND"
else
  "$PY" "$ROOT/collect.py" run --human-version human_v4 --kind "$KIND" \
    --root "$ROOT/production/$KIND" --index "${SLURM_ARRAY_TASK_ID:-0}" --shards "$PAIRS"
fi
