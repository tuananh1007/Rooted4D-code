#!/bin/bash
#SBATCH --job-name=rooted4d-tetframe
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
cd "${NVME_ROOT}/code"; mkdir -p logs
RESULTS="${NVME_ROOT}/results/${SLURM_JOB_ID}_tetframe"; mkdir -p "${RESULTS}"
export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export R4D_ACDC="${HDD_DATA}/raw/acdc"
export R4D_TETHEART_GT="${HDD_DATA}/tetheart_gt"
export R4D_TETHEART_DATA="${HDD_DATA}/tetheart_data"
export R4D_TET_FRAME_GT="${HDD_DATA}/tetheart_gt_tetframe"
export R4D_OUT="${RESULTS}"
V="${NVME_ROOT}/external/.venv-4dmr/bin/python"

echo "### E50: convert ground truth into TetHeart's 64^3 frame"
"${V}" Codes/scripts/experiments/e50_gt_to_tet_frame.py > "${RESULTS}/e50.log" 2>&1 || {
  echo "e50 failed"; tail -25 "${RESULTS}/e50.log"; exit 1; }
sed -n "/^{/,/^}/p" "${RESULTS}/e50.log" | tail -30

echo "### E47: re-stage into the preprocessed folders, from the converted meshes"
export R4D_TETHEART_GT="${HDD_DATA}/tetheart_gt_tetframe"
export R4D_SPLITS="${NVME_ROOT}/code/results_seed/acdc_patient_splits.json"
"${V}" Codes/scripts/experiments/e47_stage_tetheart_gt.py > "${RESULTS}/e47.log" 2>&1 || {
  echo "e47 failed"; tail -20 "${RESULTS}/e47.log"; exit 1; }
sed -n "/^{/,/^}/p" "${RESULTS}/e47.log"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
