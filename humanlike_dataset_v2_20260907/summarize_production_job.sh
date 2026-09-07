#!/bin/bash
#SBATCH --partition=sc-freecpu
#SBATCH --account=default
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=human400_summary
set -euo pipefail
ROOT=/iris/u/jasonyan/data/humanlike_dataset_v2_20260907
KIND=$1
PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
TARGET="$ROOT/production/$KIND"
"$PY" "$ROOT/summarize.py" --root "$TARGET" --target-pairs 200
if [ "$KIND" = toolhang ]; then
  "$PY" "$ROOT/trim_validation_tail.py" --root "$TARGET"
fi
"$PY" "$ROOT/temporal_audit.py" \
  --dataset "${KIND}_v4:$TARGET/dataset_state.hdf5:0" \
  --dataset "real_wrench:/iliad/u/jasonyan/data/real_wrench_0828_0830_pomdp_v4_jointpos_minmax_triview_20260903/hdf5/wrench_0828_0830_triview_jointpos_raw.hdf5:1" \
  --dataset "real_holder:/iliad/u/jasonyan/data/real_tool_holder_0831_pomdp_v4_jointpos_minmax_triview_20260903/hdf5/tool_holder_0831_triview_jointpos_raw.hdf5:1" \
  --output "$TARGET/temporal_audit.json"
"$PY" "$ROOT/audit_correlation.py" --path "$TARGET/dataset_state.hdf5" --absolute \
  --output "$TARGET/correlation_audit.json"
