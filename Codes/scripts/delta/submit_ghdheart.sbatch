#!/bin/bash
#SBATCH --job-name=rooted4d-ghd
#SBATCH --account=bhio-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
RESULTS="${NVME_ROOT}/results/${SLURM_JOB_ID:-manual}"
cd "${NVME_ROOT}/code"; mkdir -p logs "${RESULTS}"

rm -rf data && ln -sT "${HDD_DATA}" data
[ -L data ] || { echo "data is not a symlink, refusing to run"; exit 1; }

# GHDHeart needs its own environment: torch 2.1.1+cu118 and pytorch3d, which
# the project venv does not carry and should not.
GHDVENV="${NVME_ROOT}/external/.venv-ghd"
export GHD_ROOT="${NVME_ROOT}/external/GHDHeart"
export PYTHONPATH="${NVME_ROOT}/code/Codes:${GHD_ROOT}:${PYTHONPATH:-}"
export R4D_SPLITS="${NVME_ROOT}/code/results_seed/acdc_patient_splits.json"
export R4D_OUT="${RESULTS}"

for f in "${GHDVENV}/bin/python" "${R4D_SPLITS}" "${GHD_ROOT}/canonical_shapes/wholeheart_canonical/3.obj"; do
  [ -e "$f" ] || { echo "missing required input: $f"; exit 1; }
done

[ -f PROVENANCE.txt ] && { cp -f PROVENANCE.txt "${RESULTS}/"; grep -E '^(commit|dirty)=' PROVENANCE.txt; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"${GHDVENV}/bin/python" Codes/scripts/experiments/e37_ghdheart_chamber.py \
  > "${RESULTS}/e37.log" 2>&1 || { echo "e37 failed; tail:"; tail -40 "${RESULTS}/e37.log"; exit 1; }
tail -25 "${RESULTS}/e37.log"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
