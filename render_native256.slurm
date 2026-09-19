#!/bin/bash
#SBATCH --partition=sc-freecpu
#SBATCH --account=default
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --job-name=th_native256
#SBATCH --output=/iris/u/jasonyan/data/slurm-%A_%a-native256.out
set -euo pipefail
SOURCE=$1
SHARDS=$2
REVIEW=$3
CODE=/iris/u/jasonyan/repos/robosuite-toolhang-v2-wide-20260919
PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
export PYTHONPATH=$CODE
export MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $CODE
$PY robosuite/scripts/render_tool_hang_native_images.py --source "$SOURCE" --output-dir "$SHARDS" --index "${SLURM_ARRAY_TASK_ID}" --height 256 --width 256 --review "$REVIEW"
