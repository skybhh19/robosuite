#!/bin/bash
#SBATCH --partition=sc-freecpu
#SBATCH --account=default
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=human_audit
set -euo pipefail
ROOT=/iris/u/jasonyan/data/humanlike_dataset_v2_20260907
PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
"$PY" "$ROOT/temporal_audit.py" \
  --dataset "thread_v1:/iris/u/jasonyan/data/humanlike_pilot_20260907/threading/dataset_state.hdf5:0" \
  --dataset "tool_v1:/iris/u/jasonyan/data/humanlike_pilot_20260907/toolhang/dataset_state.hdf5:0" \
  --dataset "real_wrench:/iliad/u/jasonyan/data/real_wrench_0828_0830_pomdp_v4_jointpos_minmax_triview_20260903/hdf5/wrench_0828_0830_triview_jointpos_raw.hdf5:1" \
  --dataset "real_holder:/iliad/u/jasonyan/data/real_tool_holder_0831_pomdp_v4_jointpos_minmax_triview_20260903/hdf5/tool_holder_0831_triview_jointpos_raw.hdf5:1" \
  --output "$ROOT/calibration/temporal_metrics.json"
