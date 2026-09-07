#!/bin/bash
#SBATCH --partition=sc-freecpu
#SBATCH --account=default
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=threading_v7
set -euo pipefail
ROOT=/iris/u/jasonyan/data/humanlike_dataset_v2_20260907
PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
MODE=$1
VERSION=${2:-human_v7}
TAG=${VERSION#human_}
COLLECTION_SEED=${3:-202609081}
TARGET="$ROOT/production_${TAG}/threading"
export PYTHONPATH="/iris/u/jasonyan/repos/robosuite-threading-humanlike-v4-20260907:${ROOT}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1
cd /iris/u/jasonyan/repos/robosuite-threading-humanlike-v4-20260907
if [ "$MODE" = prepare ]; then
  export MUJOCO_GL=egl
  "$PY" "$ROOT/collect.py" prepare --production --human-version "$VERSION" --kind threading \
    --root "$TARGET" --seed "$COLLECTION_SEED" --pairs 260
elif [ "$MODE" = run ]; then
  export MUJOCO_GL=egl
  "$PY" "$ROOT/collect.py" run --human-version "$VERSION" --kind threading \
    --root "$TARGET" --index "$SLURM_ARRAY_TASK_ID" --shards 260
elif [ "$MODE" = summary ]; then
  "$PY" "$ROOT/summarize.py" --root "$TARGET" --target-pairs 200
  "$PY" "$ROOT/temporal_audit.py" --dataset "threading_${TAG}:$TARGET/dataset_state.hdf5:0" \
    --dataset "real_wrench:/iliad/u/jasonyan/data/real_wrench_0828_0830_pomdp_v4_jointpos_minmax_triview_20260903/hdf5/wrench_0828_0830_triview_jointpos_raw.hdf5:1" \
    --dataset "real_holder:/iliad/u/jasonyan/data/real_tool_holder_0831_pomdp_v4_jointpos_minmax_triview_20260903/hdf5/tool_holder_0831_triview_jointpos_raw.hdf5:1" \
    --output "$TARGET/temporal_audit.json"
  "$PY" "$ROOT/audit_correlation.py" --path "$TARGET/dataset_state.hdf5" --absolute \
    --output "$TARGET/correlation_audit.json"
elif [ "$MODE" = retry ]; then
  export MUJOCO_GL=egl
  "$PY" "$ROOT/collect.py" run --human-version "$VERSION" --kind threading \
    --root "$TARGET" --index "$SLURM_ARRAY_TASK_ID" --shards 260 \
    --attempt 1 --output-dir retry1 --retry-regime full --failed-from "$TARGET/trials"
elif [ "$MODE" = retry_audit ]; then
  "$PY" "$ROOT/audit_retry_completion.py" --root "$TARGET" --version "$VERSION"
elif [ "$MODE" = render ]; then
  export MUJOCO_GL=osmesa
  "$PY" "$ROOT/render.py" --root "$TARGET" --index "$SLURM_ARRAY_TASK_ID"
elif [ "$MODE" = merge ]; then
  "$PY" "$ROOT/merge_images.py" --root "$TARGET"
  "$PY" "$ROOT/finalize_metadata.py" --root "$TARGET"
  "$PY" "$ROOT/integrity_audit.py" --root "$TARGET"
elif [ "$MODE" = resample ]; then
  "$PY" "$ROOT/resample_human_rate.py" --root "$TARGET"
  "$PY" "$ROOT/temporal_audit.py" --dataset "threading_${TAG}_14hz:$TARGET/dataset_image84_14hz.hdf5:0" \
    --dataset "real_wrench:/iliad/u/jasonyan/data/real_wrench_0828_0830_pomdp_v4_jointpos_minmax_triview_20260903/hdf5/wrench_0828_0830_triview_jointpos_raw.hdf5:1" \
    --dataset "real_holder:/iliad/u/jasonyan/data/real_tool_holder_0831_pomdp_v4_jointpos_minmax_triview_20260903/hdf5/tool_holder_0831_triview_jointpos_raw.hdf5:1" \
    --output "$TARGET/human_rate_temporal_audit.json"
  "$PY" "$ROOT/audit_correlation.py" --path "$TARGET/dataset_image84_14hz.hdf5" --absolute \
    --output "$TARGET/human_rate_correlation_audit.json"
  "$PY" "$ROOT/fix_version_metadata.py" --root "$TARGET"
  "$PY" "$ROOT/validate_final.py" --root "$TARGET"
elif [ "$MODE" = repair_chunks ]; then
  "$PY" "$ROOT/flatten_action_chunks.py" --root "$TARGET"
  "$PY" "$ROOT/fix_version_metadata.py" --root "$TARGET"
  "$PY" "$ROOT/validate_final.py" --root "$TARGET"
else
  echo "unknown mode: $MODE" >&2
  exit 2
fi
