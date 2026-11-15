#!/bin/bash
#SBATCH --job-name=rooted4d-thprep
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
T="${NVME_ROOT}/external/TetHeart"
VENV="${NVME_ROOT}/external/.venv-tetheart"
cd "${NVME_ROOT}/code"; mkdir -p logs
RESULTS="${NVME_ROOT}/results/${SLURM_JOB_ID}_thprep"; mkdir -p "${RESULTS}"

# ACDC ships 100 training and 50 testing patients and our split lock covers all
# 150, but handle_acdc.py reads only ACDC/training/. tetheart_src stages every
# patient under that one path as symlinks, with the citation file sorting first
# because their code drops the first glob entry. Their script is unmodified
# apart from DATA_ROOT, which is a placeholder in the distribution.
SRC="${HDD_DATA}/tetheart_src"
[ -d "${SRC}/ACDC/training" ] || { echo "staging directory missing: ${SRC}"; exit 1; }
echo "staged patients: $(ls -d ${SRC}/ACDC/training/patient* | wc -l)"

TETDATA="${HDD_DATA}/tetheart_data"
mkdir -p "${TETDATA}"

cd "$T"
# DATA_ROOT is distributed as "/path/to/your/datasets"; point it at the staging
# tree. Recorded rather than silent, and reverted afterwards so the checkout
# stays clean.
cp -f tools/preprocess/handle_acdc.py "${RESULTS}/handle_acdc.py.orig"
sed -i "s|^DATA_ROOT = .*|DATA_ROOT = \"${SRC}\"|" tools/preprocess/handle_acdc.py
grep -n "^DATA_ROOT" tools/preprocess/handle_acdc.py

echo "### handle_acdc: convert to the MSD layout"
"${VENV}/bin/python" tools/preprocess/handle_acdc.py --task acdc \
  > "${RESULTS}/handle_acdc.log" 2>&1 || { echo "handle_acdc failed"; tail -25 "${RESULTS}/handle_acdc.log"; \
  cp -f "${RESULTS}/handle_acdc.py.orig" tools/preprocess/handle_acdc.py; exit 1; }
tail -5 "${RESULTS}/handle_acdc.log"

echo "### preprocess task 914"
"${VENV}/bin/python" tools/preprocess/preprocess.py --task 914 --n_jobs 8 \
  --results "${TETDATA}" \
  > "${RESULTS}/preprocess_914.log" 2>&1 || { echo "preprocess failed"; tail -25 "${RESULTS}/preprocess_914.log"; \
  cp -f "${RESULTS}/handle_acdc.py.orig" tools/preprocess/handle_acdc.py; exit 1; }
tail -5 "${RESULTS}/preprocess_914.log"

cp -f "${RESULTS}/handle_acdc.py.orig" tools/preprocess/handle_acdc.py
echo "restored handle_acdc.py"

echo "=== produced:"
find "${TETDATA}" -maxdepth 1 -type d | head
echo "  914_3d patient folders: $(ls -d ${TETDATA}/914_3d/ACDC_patient* 2>/dev/null | wc -l)"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
