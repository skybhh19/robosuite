#!/bin/bash
#SBATCH --partition=sc-freecpu
#SBATCH --account=default
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --job-name=th_vla_pilot
set -euo pipefail

VARIANT=$1
MODE=$2
PAIRS=${3:-40}
DATA_ROOT=/iris/u/jasonyan/data/toolhang_vla_upgrade_20260911
CODE_ROOT=/iris/u/jasonyan/repos/robosuite-toolhang-vla-20260911
PIPELINE_ROOT=$CODE_ROOT/humanlike_dataset_v2_20260907
PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
TARGET=$DATA_ROOT/$VARIANT
CENTER_ARGS=()

case "$VARIANT" in
  fixed_f2)
    X=(-0.0 0.0); Y=(-0.0 0.0); YAW=(-0.0 0.0); FRICTION=2.0 ;;
  random_f2)
    X=(-0.035 0.035); Y=(-0.025 0.025); YAW=(-20.0 20.0); FRICTION=2.0 ;;
  production_random_f2)
    X=(-0.035 0.035); Y=(-0.025 0.025); YAW=(-20.0 20.0); FRICTION=2.0 ;;
  random_f4)
    X=(-0.035 0.035); Y=(-0.025 0.025); YAW=(-20.0 20.0); FRICTION=4.0 ;;
  random_center_f4)
    X=(-0.035 0.035); Y=(-0.025 0.025); YAW=(-20.0 20.0); FRICTION=2.0
    CENTER_ARGS=(--tool-center-grip-half-length 0.020 --tool-center-grip-friction 4.0) ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac

export PYTHONPATH="$CODE_ROOT:$PIPELINE_ROOT"
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$CODE_ROOT"

if [ "$MODE" = prepare ]; then
  "$PY" "$PIPELINE_ROOT/collect.py" prepare --production \
    --human-version toolhang_vla_v1 --kind toolhang --root "$TARGET" \
    --seed 202609111 --pairs "$PAIRS" \
    --fixture-x-range-m "${X[@]}" --fixture-y-range-m "${Y[@]}" \
    --fixture-yaw-range-deg "${YAW[@]}" --tool-grip-friction "$FRICTION" \
    "${CENTER_ARGS[@]}"
elif [ "$MODE" = run ]; then
  "$PY" "$PIPELINE_ROOT/collect.py" run \
    --human-version toolhang_vla_v1 --kind toolhang --root "$TARGET" \
    --index "${SLURM_ARRAY_TASK_ID:-0}" --shards "$PAIRS"
elif [ "$MODE" = retry_full ] || [ "$MODE" = retry_partial ]; then
  REGIME=${MODE#retry_}
  "$PY" "$PIPELINE_ROOT/collect.py" run \
    --human-version toolhang_vla_v1 --kind toolhang --root "$TARGET" \
    --index "${SLURM_ARRAY_TASK_ID:-0}" --shards "$PAIRS" \
    --attempt 1 --output-dir retry1 --retry-regime "$REGIME" \
    --failed-from "$TARGET/trials"
elif [ "$MODE" = retry_audit ]; then
  "$PY" "$PIPELINE_ROOT/audit_retry_completion.py" --root "$TARGET" \
    --version toolhang_vla_v1
elif [ "$MODE" = render ]; then
  export MUJOCO_GL=osmesa
  "$PY" "$PIPELINE_ROOT/render.py" --root "$TARGET" \
    --index "${SLURM_ARRAY_TASK_ID:-0}"
elif [ "$MODE" = summary ]; then
  "$PY" "$PIPELINE_ROOT/summarize.py" --root "$TARGET"
else
  echo "unknown mode: $MODE" >&2
  exit 2
fi
