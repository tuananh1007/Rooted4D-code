#!/bin/bash
#SBATCH --job-name=rooted4d-ladder
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
cd "${NVME_ROOT}/code"; mkdir -p logs
RESULTS="${NVME_ROOT}/results/${SLURM_JOB_ID}_ladder"; mkdir -p "${RESULTS}"

if [ -e data ] && [ ! -L data ]; then echo "data is not a symlink, refusing"; exit 1; fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then ln -sfT "${HDD_DATA}" data; fi

source .venv/bin/activate
export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export R4D_THREADS="${SLURM_CPUS_PER_TASK}"
export R4D_OUT="${RESULTS}"
export R4D_BENCHMARK="data/derived/semisynthetic-real-v3"
export CUDA_VISIBLE_DEVICES=""

[ -f PROVENANCE.txt ] && { cp -f PROVENANCE.txt "${RESULTS}/"; grep -E '^(commit|dirty)=' PROVENANCE.txt; }

echo "### full suite first: a repaired baseline that breaks the suite is not a repair"
python -m unittest discover -s Codes/tests -q > "${RESULTS}/tests.log" 2>&1 || {
  echo "tests failed"; tail -20 "${RESULTS}/tests.log"; exit 1; }
tail -3 "${RESULTS}/tests.log"

echo "### E51: ladder rerun with the repaired late fusion"
python Codes/scripts/experiments/e51_ladder_rerun.py > "${RESULTS}/e51.log" 2>&1 || {
  echo "e51 failed"; tail -30 "${RESULTS}/e51.log"; exit 1; }
tail -14 "${RESULTS}/e51.log"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
