#!/bin/bash
# E99 part 1 (Docs/E99_MITEA_SEGMENTER_PREREGISTRATION_2026-09-29.md): convert
# MITEA under the committed subject split to nnU-Net raw (Dataset504_MITEA), plan
# and preprocess, and write fold 0.
#SBATCH --job-name=rooted4d-e99-prep
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=02:00:00
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
export R4D_MITEA="data/raw/mitea"
S=Codes/scripts/experiments/e99_mitea_segmenter.py

echo "### convert"
R4D_MODE=convert "${VENV}/bin/python" "$S"
echo "### plan and preprocess"
"${VENV}/bin/nnUNetv2_plan_and_preprocess" -d 504 --verify_dataset_integrity -np 8 -c 3d_fullres
echo "### fold 0 = the committed subject split"
R4D_MODE=splits "${VENV}/bin/python" "$S"
echo "### plans"
head -60 "${nnUNet_preprocessed}/Dataset504_MITEA/nnUNetPlans.json"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
