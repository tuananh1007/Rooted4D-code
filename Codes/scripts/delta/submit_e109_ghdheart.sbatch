#!/bin/bash
# E109's GHDHeart part (e109_qualitative_96.py): GHDHeart refitted on the chamber
# donors' labelled frames at its validated setting (E37: 64 basis functions,
# 2000 iterations, learning rate 0.01). The picks are recomputed here, by the
# same rule as the CPU job, so the two jobs need not wait for each other.
#SBATCH --job-name=rooted4d-e109-ghd
#SBATCH --account=bhio-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=48g
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
NVME="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
cd "${NVME}/code"; mkdir -p logs
if [ -e data ] && [ ! -L data ]; then echo "data is not a symlink, refusing"; exit 1; fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then ln -sfT "${HDD_DATA}" data; fi
grep -E '^(commit|dirty)=' PROVENANCE.txt
if grep -q '^dirty=yes' PROVENANCE.txt; then echo "refusing: dirty tree"; exit 1; fi

RESULTS="${NVME}/results/${SLURM_JOB_ID}_qual96"; mkdir -p "${RESULTS}"
cp -f PROVENANCE.txt "${RESULTS}/"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TQDM_DISABLE=1
export R4D_ALLOW_TEST=1 R4D_OUT="${RESULTS}"
export R4D_E100_ROWS="${NVME}/results/e100/v4n96o/test/bd364644c4605d9d"
export R4D_ACDC="${HDD_DATA}/raw/acdc"
export GHD_ROOT="${NVME}/external/GHDHeart"
export R4D_SPLITS="${NVME}/code/results_seed/acdc_patient_splits.json"
export R4D_ITER=2000 R4D_LR=0.01
SCRIPT=Codes/scripts/experiments/e109_qualitative_96.py

echo "### picks"
PYTHONPATH="${NVME}/code/Codes" R4D_PART=picks "${NVME}/code/.venv/bin/python" -u "${SCRIPT}" \
  > "${RESULTS}/picks.log" 2>&1 || { tail -30 "${RESULTS}/picks.log"; exit 1; }
cat "${RESULTS}/picks.log"

echo "### ghdheart"
nvidia-smi --query-gpu=name --format=csv,noheader || true
PYTHONPATH="${NVME}/code/Codes:${GHD_ROOT}" R4D_PART=ghdheart "${NVME}/external/.venv-ghd/bin/python" -u "${SCRIPT}" \
  > "${RESULTS}/ghdheart.log" 2>&1 || { tail -30 "${RESULTS}/ghdheart.log"; exit 1; }
tail -12 "${RESULTS}/ghdheart.log"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
