#!/bin/bash
#SBATCH --partition=sc-freecpu
#SBATCH --account=default
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --job-name=th256_final
#SBATCH --output=/iris/u/jasonyan/data/slurm-%j-native256-finalize.out
set -euo pipefail
SOURCE=$1
SHARDS=$2
OUTPUT=$3
REPORT=$4
EPISODES=$5
FULL=$6
PARTIAL=$7
CODE=/iris/u/jasonyan/repos/robosuite-toolhang-v2-wide-20260919
PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
export PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $CODE
$PY robosuite/scripts/merge_tool_hang_native_images.py --source "$SOURCE" --shards "$SHARDS" --output "$OUTPUT" --height 256 --width 256
$PY robosuite/scripts/validate_tool_hang_native_images.py --source "$SOURCE" --output "$OUTPUT" --expected-episodes "$EPISODES" --expected-full "$FULL" --expected-partial "$PARTIAL" --height 256 --width 256 --report "$REPORT"
