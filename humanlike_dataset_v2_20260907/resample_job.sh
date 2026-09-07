#!/bin/bash
#SBATCH --partition=sc-freecpu
#SBATCH --account=default
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --job-name=human14hz
set -euo pipefail
ROOT=/iris/u/jasonyan/data/humanlike_dataset_v2_20260907
/iris/u/jasonyan/miniforge3/envs/openx/bin/python "$ROOT/resample_human_rate.py" \
  --root "$ROOT/production/$1"
PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
"$PY" "$ROOT/temporal_audit.py" \
  --dataset "$1_14hz:$ROOT/production/$1/dataset_image84_14hz.hdf5:0" \
  --dataset "real_wrench:/iliad/u/jasonyan/data/real_wrench_0828_0830_pomdp_v4_jointpos_minmax_triview_20260903/hdf5/wrench_0828_0830_triview_jointpos_raw.hdf5:1" \
  --dataset "real_holder:/iliad/u/jasonyan/data/real_tool_holder_0831_pomdp_v4_jointpos_minmax_triview_20260903/hdf5/tool_holder_0831_triview_jointpos_raw.hdf5:1" \
  --output "$ROOT/production/$1/human_rate_temporal_audit.json"
"$PY" "$ROOT/audit_correlation.py" \
  --path "$ROOT/production/$1/dataset_image84_14hz.hdf5" --absolute \
  --output "$ROOT/production/$1/human_rate_correlation_audit.json"
"$PY" "$ROOT/fix_version_metadata.py" --root "$ROOT/production/$1"
"$PY" "$ROOT/validate_final.py" --root "$ROOT/production/$1"
