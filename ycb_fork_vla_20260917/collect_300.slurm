#!/bin/bash
#SBATCH --job-name=fork-v2-collect
#SBATCH --partition=iliad
#SBATCH --account=iliad
#SBATCH --exclude=iliad-hgx-1
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/iris/u/jasonyan/data/ycb_fork_vla300_variation_v2_20260917/logs/collect_%A_%a.out
#SBATCH --error=/iris/u/jasonyan/data/ycb_fork_vla300_variation_v2_20260917/logs/collect_%A_%a.err

set -euo pipefail

REPO=/iris/u/jasonyan/repos/robosuite-ycb-fork-v2-20260917
ROOT=/iris/u/jasonyan/data/ycb_fork_vla300_variation_v2_20260917

if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
    REGIME=full_visible
    SEED=202609171
else
    REGIME=partial_hidden
    SEED=202609172
fi

mkdir -p "${ROOT}/logs" "${ROOT}/raw/${REGIME}" "${ROOT}/videos/${REGIME}"

PY=/iris/u/jasonyan/miniforge3/envs/openx/bin/python
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${REPO}"
"${PY}" robosuite/scripts/collect_ycb_fork_joint.py \
    --directory "${ROOT}/raw/${REGIME}" \
    --video-dir "${ROOT}/videos/${REGIME}" \
    --regime "${REGIME}" \
    --num-demos-per-regime 150 \
    --max-attempts 5000 \
    --video-count 10 \
    --video-fps 20 \
    --size 320 \
    --horizon 3000 \
    --seed "${SEED}"
