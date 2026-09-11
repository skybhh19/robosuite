#!/bin/bash
#SBATCH --partition=sc-freecpu
#SBATCH --account=default
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --job-name=th_vla_v2
set -euo pipefail

VARIANT=$1
MODE=$2
PAIRS=${3:-40}
DATA_ROOT=/iris/u/jasonyan/data/toolhang_vla_upgrade_v2_20260912
CODE_ROOT=/iris/u/jasonyan/repos/robosuite-toolhang-vla-20260911
PIPELINE_ROOT=$CODE_ROOT/humanlike_dataset_v2_20260907
PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
TARGET=$DATA_ROOT/$VARIANT

case "$VARIANT" in
  moderate_v2)
    X=(-0.055 0.055); Y=(-0.040 0.040); YAW=(-30.0 35.0) ;;
  d08_like_v2)
    X=(-0.070 0.070); Y=(-0.050 0.050); YAW=(-15.0 45.0) ;;
  production_observability_v2)
    # Filled after the paired pilot comparison.
    X=(-0.055 0.055); Y=(-0.040 0.040); YAW=(-30.0 35.0) ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac

export PYTHONPATH="$CODE_ROOT:$PIPELINE_ROOT"
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$CODE_ROOT"

if [ "$MODE" = prepare ]; then
  "$PY" "$PIPELINE_ROOT/collect.py" prepare --production \
    --human-version toolhang_vla_v2 --kind toolhang --root "$TARGET" \
    --seed 202609121 --pairs "$PAIRS" \
    --fixture-x-range-m "${X[@]}" --fixture-y-range-m "${Y[@]}" \
    --fixture-yaw-range-deg "${YAW[@]}" --tool-grip-friction 2.0
elif [ "$MODE" = run ]; then
  "$PY" "$PIPELINE_ROOT/collect.py" run \
    --human-version toolhang_vla_v2 --kind toolhang --root "$TARGET" \
    --index "${SLURM_ARRAY_TASK_ID:-0}" --shards "$PAIRS"
elif [ "$MODE" = summary ]; then
  ARGS=(--root "$TARGET" --require-visibility-labels)
  if [ "$VARIANT" = production_observability_v2 ]; then ARGS+=(--target-pairs 150); fi
  "$PY" "$PIPELINE_ROOT/summarize.py" "${ARGS[@]}"
elif [ "$MODE" = render ]; then
  export MUJOCO_GL=osmesa
  "$PY" "$PIPELINE_ROOT/render.py" --root "$TARGET" \
    --index "${SLURM_ARRAY_TASK_ID:-0}"
elif [ "$MODE" = merge ]; then
  "$PY" "$PIPELINE_ROOT/merge_images.py" --root "$TARGET"
  "$PY" "$PIPELINE_ROOT/finalize_metadata.py" --root "$TARGET"
  "$PY" "$PIPELINE_ROOT/integrity_audit.py" --root "$TARGET"
elif [ "$MODE" = resample ]; then
  "$PY" "$PIPELINE_ROOT/resample_human_rate.py" --root "$TARGET"
  "$PY" "$PIPELINE_ROOT/fix_version_metadata.py" --root "$TARGET"
  "$PY" "$PIPELINE_ROOT/validate_final.py" --root "$TARGET" \
    --expected-episodes 300 --expected-full 150 --expected-partial 150
else
  echo "unknown mode: $MODE" >&2; exit 2
fi
