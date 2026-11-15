#!/bin/bash
# E103 (Docs/E103_E107_REVIEWER_ANALYSES_PREREGISTRATION_2026-10-03.md): late
# fusion with cross-modal transfer on MITEA, E99's same-scan pairing, test.
# Reads E99's geometry and rows on Delta; writes numbers only.
#SBATCH --job-name=rooted4d-e103
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=02:00:00
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
S=Codes/scripts/experiments/e103_mitea_transfer.py
R4D_MODE=fit R4D_ALLOW_TEST=1 python -u "$S"
R4D_MODE=analyze python -u "$S"
mkdir -p "${NVME_ROOT}/results/e103"
cp -f results/benchmark/E103/*.json "${NVME_ROOT}/results/e103/"
cp -f PROVENANCE.txt "${NVME_ROOT}/results/e103/PROVENANCE_${SLURM_JOB_ID}.txt"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
