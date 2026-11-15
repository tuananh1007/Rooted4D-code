#!/bin/bash
#SBATCH --job-name=rooted4d-thgt
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
cd "${NVME_ROOT}/code"; mkdir -p logs
RESULTS="${NVME_ROOT}/results/${SLURM_JOB_ID}_thgt"; mkdir -p "${RESULTS}"

export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export R4D_ACDC="${HDD_DATA}/raw/acdc"
export R4D_TETHEART_GT="${HDD_DATA}/tetheart_gt"
export R4D_TETHEART_DATA="${HDD_DATA}/tetheart_data"
export R4D_SPLITS="${NVME_ROOT}/code/results_seed/acdc_patient_splits.json"
export R4D_OUT="${RESULTS}"
V="${NVME_ROOT}/external/.venv-4dmr/bin/python"

echo "### E46: right-ventricle meshes, so the loader's three-per-frame indexing holds"
"${V}" Codes/scripts/experiments/e46_add_rv_meshes.py > "${RESULTS}/e46.log" 2>&1 || {
  echo "e46 failed"; tail -25 "${RESULTS}/e46.log"; exit 1; }
sed -n "/^{/,/^}/p" "${RESULTS}/e46.log" | tail -18

echo "### E47: stage complete patients into the preprocessed folders"
"${V}" Codes/scripts/experiments/e47_stage_tetheart_gt.py > "${RESULTS}/e47.log" 2>&1 || {
  echo "e47 failed"; tail -25 "${RESULTS}/e47.log"; exit 1; }
cat "${RESULTS}/e47.log"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
