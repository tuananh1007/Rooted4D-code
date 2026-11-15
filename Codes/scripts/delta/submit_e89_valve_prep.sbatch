#!/bin/bash
# E89 part 1 (Docs/E89_VALVE_SEGMENTER_PREREGISTRATION_2026-09-28.md): confirm the
# valve donor order, convert the benchmark's valve partition to nnU-Net raw
# (Dataset503_MVSegBench), plan and preprocess, and write fold 0.
#SBATCH --job-name=rooted4d-e89-prep
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
cd "${NVME_ROOT}/code"; mkdir -p logs
if [ -e data ] && [ ! -L data ]; then echo "data exists and is not a symlink, refusing to run"; exit 1; fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then ln -sfT "${HDD_DATA}" data; fi
grep -E '^(commit|dirty)=' PROVENANCE.txt
if grep -q '^dirty=yes' PROVENANCE.txt; then echo "refusing: synced tree was dirty"; exit 1; fi

export nnUNet_raw="${HDD_DATA}/nnunet/raw"
export nnUNet_preprocessed="${HDD_DATA}/nnunet/preprocessed"
export nnUNet_results="${NVME_ROOT}/nnunet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
VENV="${NVME_ROOT}/external/.venv-nnunet"
export R4D_MVSEG="data/raw/mvseg2023"
S=Codes/scripts/experiments/e89_valve_segmenter.py

echo "### donor order"
( source .venv/bin/activate; export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
  R4D_MODE=check python "$S" )
echo "### convert"
R4D_MODE=convert "${VENV}/bin/python" "$S"
echo "### plan and preprocess"
"${VENV}/bin/nnUNetv2_plan_and_preprocess" -d 503 --verify_dataset_integrity -np 8 -c 3d_fullres
echo "### fold 0 = the benchmark's valve partition"
R4D_MODE=splits "${VENV}/bin/python" "$S"
echo "### plans"
head -80 "${nnUNet_preprocessed}/Dataset503_MVSegBench/nnUNetPlans.json"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
