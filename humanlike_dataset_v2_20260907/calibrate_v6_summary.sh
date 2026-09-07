#!/bin/bash
#SBATCH --partition=sc-freecpu
#SBATCH --account=default
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=human_v6_summary
set -euo pipefail
ROOT=/iris/u/jasonyan/data/humanlike_dataset_v2_20260907
PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
CAL="$ROOT/calibration_v6/threading"
"$PY" "$ROOT/summarize.py" --root "$CAL"
"$PY" "$ROOT/temporal_audit.py" \
  --dataset "threading_v6:$CAL/dataset_state.hdf5:0" \
  --dataset "real_wrench:/iliad/u/jasonyan/data/real_wrench_0828_0830_pomdp_v4_jointpos_minmax_triview_20260903/hdf5/wrench_0828_0830_triview_jointpos_raw.hdf5:1" \
  --dataset "real_holder:/iliad/u/jasonyan/data/real_tool_holder_0831_pomdp_v4_jointpos_minmax_triview_20260903/hdf5/tool_holder_0831_triview_jointpos_raw.hdf5:1" \
  --output "$CAL/temporal_audit.json"
