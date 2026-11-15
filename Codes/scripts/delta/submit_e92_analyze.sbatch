#!/bin/bash
# E92 (Docs/E92_INTERVENTIONS_PREREGISTRATION_2026-09-28.md): the analysis. It
# reads the intervention instances' test samples, which exist only here, and
# every level's rows from the E59 results root.
#SBATCH --job-name=rooted4d-e92-analyze
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
cd "${NVME_ROOT}/code"; mkdir -p logs
if [ -e data ] && [ ! -L data ]; then echo "data is not a symlink, refusing"; exit 1; fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then ln -sfT "${HDD_DATA}" data; fi
grep -E '^(commit|dirty)=' PROVENANCE.txt
if grep -q '^dirty=yes' PROVENANCE.txt; then echo "refusing: dirty tree"; exit 1; fi
source .venv/bin/activate
export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u Codes/scripts/experiments/e92_interventions.py analyze "${NVME_ROOT}/results/e59"
mkdir -p "${NVME_ROOT}/results/e92"
cp -f results/benchmark/E92/E92_interventions.json "${NVME_ROOT}/results/e92/"
cp -f PROVENANCE.txt "${NVME_ROOT}/results/e92/PROVENANCE_${SLURM_JOB_ID}.txt"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
