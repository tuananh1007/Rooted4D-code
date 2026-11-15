#!/bin/bash
# E32: does a handful of known correspondences rescue image-domain fitting?
#
# Tests the prediction section 19 makes and section 21 sharpened: the image
# objective is blind to rotation about the annular axis because a surface
# likelihood carries no vertex identity, so supplying a few identified points
# should collapse the rotation error and two should already do most of the work.
#
#SBATCH --job-name=rooted4d-landmarks
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
NVME_ROOT="/work/nvme/bhio/${USER}/Rooted4D"
HDD_ROOT="/work/hdd/bhio/${USER}/Rooted4D"
RUN_ID="${SLURM_JOB_ID:-manual}"
RESULTS="${NVME_ROOT}/results/${RUN_ID}"
cd "${NVME_ROOT}/code"
mkdir -p logs "${RESULTS}"
rm -rf data && ln -sT "${HDD_ROOT}/data" data
[ -L data ] || { echo "ERROR: data is not a symlink" >&2; exit 1; }

# The venv is excluded from the sync, so it stays at the code-tree root
# rather than moving under Codes/ with the package.
PY="${NVME_ROOT}/code/.venv/bin/python"
export PYTHONPATH="${NVME_ROOT}/code/Codes" R4D_OUT="${RESULTS}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "run_id ${RUN_ID}  host $(hostname)  started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
[ -f PROVENANCE.txt ] && { cp -f PROVENANCE.txt "${RESULTS}/"; grep -E '^(commit|dirty)=' PROVENANCE.txt; }

R4D_BENCHMARK="data/derived/semisynthetic-real-v3" R4D_MODES=32 R4D_SAMPLES=30 \
  $PY Codes/scripts/experiments/e32_image_landmarks.py > "${RESULTS}/e32_image_landmarks.log" 2>&1 || { tail -40 "${RESULTS}/e32_image_landmarks.log"; exit 1; }
tail -25 "${RESULTS}/e32_image_landmarks.log"

echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
