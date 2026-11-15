#!/bin/bash
#SBATCH --job-name=rooted4d-ghdsweep
#SBATCH --account=bhio-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=10:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail

# One array task per num_basis. A single serial sweep would need roughly twelve
# hours of fitting; four concurrent tasks each do a quarter of it.
BASES=(16 36 64 100)
BASIS=${BASES[$SLURM_ARRAY_TASK_ID]}

NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
RESULTS="${NVME_ROOT}/results/${SLURM_ARRAY_JOB_ID}_sweep"
cd "${NVME_ROOT}/code"; mkdir -p logs "${RESULTS}"

# Concurrent array tasks must not race on this. "rm -rf data && ln -sT" leaves a
# window in which the link is absent, and a sibling task reading through it gets
# a FileNotFoundError partway through a run. "ln -sfT" replaces a symlink
# atomically, so there is no window; it refuses if data is a real directory,
# which is the safe way to fail.
# Several jobs share this working directory, so the data symlink must not be
# rewritten on every run: replacing it, even with an identical target, opens a
# window in which a concurrent job reading through it gets a FileNotFoundError
# on a file that exists. Only create it when it is absent or points elsewhere.
if [ -e data ] && [ ! -L data ]; then
  echo "data exists and is not a symlink, refusing to run"; exit 1
fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then
  ln -sfT "${HDD_DATA}" data
fi
[ -L data ] || { echo "data is not a symlink, refusing to run"; exit 1; }

GHDVENV="${NVME_ROOT}/external/.venv-ghd"
export GHD_ROOT="${NVME_ROOT}/external/GHDHeart"
export PYTHONPATH="${NVME_ROOT}/code/Codes:${GHD_ROOT}:${PYTHONPATH:-}"
export R4D_SPLITS="${NVME_ROOT}/code/results_seed/acdc_patient_splits.json"
export R4D_OUT="${RESULTS}"
export R4D_MODE=validate
export R4D_BASIS="${BASIS}"
export R4D_OUTNAME="E37_val_B${BASIS}.json"
# The control is a property of the frame mapping, not of num_basis, so only the
# first array task pays for it.
if [ "${SLURM_ARRAY_TASK_ID}" != "0" ]; then export R4D_SKIP_CONTROL=1; fi
export TQDM_DISABLE=1

for f in "${GHDVENV}/bin/python" "${R4D_SPLITS}" "${GHD_ROOT}/canonical_shapes/wholeheart_canonical/3.obj"; do
  [ -e "$f" ] || { echo "missing required input: $f"; exit 1; }
done

[ -f PROVENANCE.txt ] && cp -f PROVENANCE.txt "${RESULTS}/"
nvidia-smi --query-gpu=name --format=csv,noheader || true
echo "array task ${SLURM_ARRAY_TASK_ID}: num_basis=${BASIS}"

"${GHDVENV}/bin/python" Codes/scripts/experiments/e37_ghdheart_chamber.py \
  > "${RESULTS}/e37_B${BASIS}.log" 2>&1 \
  || { echo "failed; tail:"; tail -30 "${RESULTS}/e37_B${BASIS}.log"; exit 1; }
tail -6 "${RESULTS}/e37_B${BASIS}.log"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
