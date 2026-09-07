#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --job-name=human400_bc
set -euo pipefail
ROOT=/iris/u/jasonyan/data/humanlike_dataset_v2_20260907
MODE=$1
INDEX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$INDEX" -lt 6 ]; then
  RS=/iris/u/jasonyan/repos/robosuite-threading-humanlike-v4-20260907
else
  RS=/iris/u/jasonyan/repos/robosuite-toolhang-humanlike-v4-20260907
fi
export PYTHONPATH="${RS}:/iris/u/jasonyan/repos/demonstration-information/robomimic:${ROOT}"
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
"$PY" "$ROOT/bc_workflow.py" "$MODE" --index "$INDEX"
