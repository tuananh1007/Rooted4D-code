#!/bin/bash
# E90 (Docs/E90_RELATIONAL_MEASUREMENTS_PREREGISTRATION_2026-09-28.md): the
# relational and control measurements on stored fits. SciPy is needed for the
# convex-hull test, so this runs in the nnU-Net environment with the project on
# PYTHONPATH, as E86's component analysis did.
#   sbatch --export=ALL,R4D_E90_SPLIT=val,R4D_E90_ROWS=<rows dir> submit_e90_measure.sbatch
# R4D_E90_ROWS may also be a chain root: its TEST_DONE names the test rows (E100).
#SBATCH --job-name=rooted4d-e90-measure
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
: "${R4D_E90_SPLIT:?set R4D_E90_SPLIT to val or test}"
: "${R4D_E90_ROWS:?set R4D_E90_ROWS to the stored-fit rows directory}"
if [ -f "${R4D_E90_ROWS}/TEST_DONE" ]; then R4D_E90_ROWS="${R4D_E90_ROWS}/$(cat "${R4D_E90_ROWS}/TEST_DONE")"; fi
if [ -e data ] && [ ! -L data ]; then echo "data is not a symlink, refusing"; exit 1; fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then ln -sfT "${HDD_DATA}" data; fi
grep -E '^(commit|dirty)=' PROVENANCE.txt
if grep -q '^dirty=yes' PROVENANCE.txt; then echo "refusing: dirty tree"; exit 1; fi
export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# R4D_E90_INSTANCE=oriented measures E94's test fits (E90's amendment); the default is the original.
INSTANCE="${R4D_E90_INSTANCE:-original}"
SUFFIX=""; if [ "${INSTANCE}" != "original" ]; then SUFFIX="_${INSTANCE}"; fi
"${NVME_ROOT}/external/.venv-nnunet/bin/python" -u Codes/scripts/experiments/e90_relational.py measure "${R4D_E90_SPLIT}" "${R4D_E90_ROWS}" "${INSTANCE}"
mkdir -p "${NVME_ROOT}/results/e90"
cp -f "results/benchmark/E90/E90_${R4D_E90_SPLIT}${SUFFIX}.json" "${NVME_ROOT}/results/e90/"
cp -f PROVENANCE.txt "${NVME_ROOT}/results/e90/PROVENANCE_${SLURM_JOB_ID}.txt"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
