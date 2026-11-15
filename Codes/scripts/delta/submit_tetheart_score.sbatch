#!/bin/bash
#SBATCH --job-name=rooted4d-thscore
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
VENV="${NVME_ROOT}/external/.venv-tetheart"
cd "${NVME_ROOT}/code"; mkdir -p logs
SRC="${R4D_PRED_ROOT}"
RESULTS="${NVME_ROOT}/results/${SLURM_JOB_ID}_thscore"; mkdir -p "${RESULTS}"
export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export R4D_ACDC="${HDD_DATA}/raw/acdc"
export R4D_OUT="${RESULTS}"
for d in "${SRC}"/preds_*; do
  [ -d "$d" ] || continue
  tag=$(basename "$d" | sed 's/^preds_//')
  echo "### ${tag}: $(ls ${d}/*_pred.obj 2>/dev/null | wc -l) meshes"
  R4D_PRED_DIR="$d" R4D_OUTNAME="E49_${tag}.json" \
    "${VENV}/bin/python" Codes/scripts/experiments/e49_tetheart_eval.py \
    > "${RESULTS}/e49_${tag}.log" 2>&1 || { echo "  failed"; tail -6 "${RESULTS}/e49_${tag}.log"; continue; }
  sed -n '/^{/,/^}/p' "${RESULTS}/e49_${tag}.log" | tail -22
done
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
