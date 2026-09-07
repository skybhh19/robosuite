#!/bin/bash
#SBATCH --partition=sc-freecpu
#SBATCH --account=default
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=old_new_temporal
set -euo pipefail
ROOT=/iris/u/jasonyan/data/humanlike_dataset_v2_20260907
PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
"$PY" "$ROOT/temporal_audit.py" \
  --dataset "old_thread:/iris/u/tiangao/projects/robosuite/output/threading_d06_hard_joint_position_strict_clean_40x10_seed20260901/dataset/demo.hdf5:0" \
  --dataset "new_thread_20hz:$ROOT/production_v8/threading/dataset_state.hdf5:0" \
  --dataset "new_thread_14hz:$ROOT/production_v8/threading/dataset_image84_14hz.hdf5:0" \
  --dataset "old_tool:/iris/u/jasonyan/data/tool_hang_robust_joint_dataset400_20260904/dataset/tool_hang_robust_joint_400_image84.hdf5:0" \
  --dataset "new_tool_20hz:$ROOT/production/toolhang/dataset_state.hdf5:0" \
  --dataset "new_tool_14hz:$ROOT/production/toolhang/dataset_image84_14hz.hdf5:0" \
  --dataset "real_wrench:/iliad/u/jasonyan/data/real_wrench_0828_0830_pomdp_v4_jointpos_minmax_triview_20260903/hdf5/wrench_0828_0830_triview_jointpos_raw.hdf5:1" \
  --dataset "real_holder:/iliad/u/jasonyan/data/real_tool_holder_0831_pomdp_v4_jointpos_minmax_triview_20260903/hdf5/tool_holder_0831_triview_jointpos_raw.hdf5:1" \
  --output "$ROOT/evidence/old_new_temporal.json"
