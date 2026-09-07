#!/bin/bash
#SBATCH --partition=sc-freecpu
#SBATCH --account=default
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --job-name=human400_v6_14hz
set -euo pipefail
ROOT=/iris/u/jasonyan/data/humanlike_dataset_v2_20260907
TARGET="$ROOT/production_v6/threading"
PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
"$PY" "$ROOT/resample_human_rate.py" --root "$TARGET"
"$PY" "$ROOT/temporal_audit.py" \
  --dataset "threading_v6_14hz:$TARGET/dataset_image84_14hz.hdf5:0" \
  --dataset "real_wrench:/iliad/u/jasonyan/data/real_wrench_0828_0830_pomdp_v4_jointpos_minmax_triview_20260903/hdf5/wrench_0828_0830_triview_jointpos_raw.hdf5:1" \
  --dataset "real_holder:/iliad/u/jasonyan/data/real_tool_holder_0831_pomdp_v4_jointpos_minmax_triview_20260903/hdf5/tool_holder_0831_triview_jointpos_raw.hdf5:1" \
  --output "$TARGET/human_rate_temporal_audit.json"
"$PY" "$ROOT/audit_correlation.py" --path "$TARGET/dataset_image84_14hz.hdf5" --absolute \
  --output "$TARGET/human_rate_correlation_audit.json"
"$PY" "$ROOT/fix_version_metadata.py" --root "$TARGET"
"$PY" "$ROOT/validate_final.py" --root "$TARGET"
