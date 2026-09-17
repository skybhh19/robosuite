#!/bin/bash
#SBATCH --job-name=fork-finalize256
#SBATCH --partition=sc-freecpu
#SBATCH --account=default
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/iris/u/jasonyan/data/ycb_fork_vla300_variation_v2_20260917/logs/finalize_%j.out
#SBATCH --error=/iris/u/jasonyan/data/ycb_fork_vla300_variation_v2_20260917/logs/finalize_%j.err
set -euo pipefail
REPO=/iris/u/jasonyan/repos/robosuite-ycb-fork-v2-20260917
ROOT=/iris/u/jasonyan/data/ycb_fork_vla300_variation_v2_20260917
PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
export PYTHONPATH="$REPO" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" "$REPO/robosuite/scripts/merge_ycb_fork_native_images.py" --source "$ROOT/dataset_state_14hz.hdf5" --shards "$ROOT/native256_shards" --output "$ROOT/dataset_image256_14hz.hdf5"
"$PY" "$REPO/robosuite/scripts/validate_ycb_fork_image_hdf5.py" --dataset "$ROOT/dataset_image256_14hz.hdf5" --report "$ROOT/native256_validation.json"
sha256sum "$ROOT/dataset_image256_14hz.hdf5" > "$ROOT/SHA256SUMS"
