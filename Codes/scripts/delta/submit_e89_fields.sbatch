#!/bin/bash
# E89 part 2: the measured error fields and the measured-error test instance.
#SBATCH --job-name=rooted4d-e89-fields
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
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
S=Codes/scripts/experiments/e89_measured_error.py
R4D_MODE=fields python -u "$S"
R4D_MODE=instance python -u "$S"
mkdir -p "${NVME_ROOT}/results/e89"
cp -f results/benchmark/E89/E89_fields.npz results/benchmark/E89/E89_fields.json data/derived/semisynthetic-real-v4-measured/ring_disagreement.json "${NVME_ROOT}/results/e89/"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
