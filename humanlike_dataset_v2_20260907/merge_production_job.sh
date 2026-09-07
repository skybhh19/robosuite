#!/bin/bash
#SBATCH --partition=sc-freecpu
#SBATCH --account=default
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --job-name=human400_merge
set -euo pipefail
ROOT=/iris/u/jasonyan/data/humanlike_dataset_v2_20260907
KIND=$1
PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
"$PY" "$ROOT/merge_images.py" --root "$ROOT/production/$KIND"
"$PY" "$ROOT/finalize_metadata.py" --root "$ROOT/production/$KIND"
"$PY" "$ROOT/integrity_audit.py" --root "$ROOT/production/$KIND"
