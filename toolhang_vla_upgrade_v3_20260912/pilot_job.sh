#!/bin/bash
#SBATCH --partition=sc-freecpu
#SBATCH --account=default
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --job-name=th_vla_v3
set -euo pipefail

MODE=$1
PAIRS=${2:-40}
VARIANT=${3:-threading_style_ring_small_pilot}
DATA_ROOT=/iris/u/jasonyan/data/toolhang_vla_upgrade_v3_20260912/$VARIANT
CODE_ROOT=/iris/u/jasonyan/repos/robosuite-toolhang-vla-20260911
PIPELINE_ROOT=$CODE_ROOT/humanlike_dataset_v2_20260907
PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
export PYTHONPATH="$CODE_ROOT:$PIPELINE_ROOT"
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$CODE_ROOT"

if [ "$MODE" = prepare ]; then
  "$PY" "$PIPELINE_ROOT/collect.py" prepare --production \
    --human-version toolhang_vla_v3 --kind toolhang --root "$DATA_ROOT" \
    --seed 202609121 --pairs "$PAIRS" \
    --fixture-x-range-m -0.055 0.055 --fixture-y-range-m -0.040 0.040 \
    --fixture-yaw-range-deg -30.0 35.0 --tool-grip-friction 2.0
elif [ "$MODE" = run ]; then
  "$PY" "$PIPELINE_ROOT/collect.py" run \
    --human-version toolhang_vla_v3 --kind toolhang --root "$DATA_ROOT" \
    --index "${SLURM_ARRAY_TASK_ID:-0}" --shards "$PAIRS"
elif [ "$MODE" = summary ]; then
  "$PY" "$PIPELINE_ROOT/summarize.py" --root "$DATA_ROOT" --require-visibility-labels
else
  echo "unknown mode: $MODE" >&2
  exit 2
fi
