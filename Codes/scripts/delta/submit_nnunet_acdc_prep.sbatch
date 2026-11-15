#!/bin/bash
#SBATCH --job-name=rooted4d-e84-prep
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

# E84 (Docs/E84_NNUNET_CHAMBER_PREREGISTRATION_2026-09-28.md): ACDC into nnU-Net's
# raw format under the project split, plan and preprocess 3d_fullres, and write
# fold 0 as the project split (train patients / validation patients).

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
export R4D_ACDC="data/raw/acdc"
export R4D_SPLITS="${NVME_ROOT}/code/results_seed/acdc_patient_splits.json"
S=Codes/scripts/experiments/e84_nnunet_chamber.py

if [ -d "${nnUNet_raw}/Dataset502_ACDC" ]; then echo "Dataset502_ACDC already exists; refusing to overwrite"; exit 1; fi
echo "### convert"
R4D_MODE=convert "${VENV}/bin/python" "$S"
echo "### plan and preprocess"
"${VENV}/bin/nnUNetv2_plan_and_preprocess" -d 502 --verify_dataset_integrity -np 8 -c 3d_fullres
echo "### fold 0 = the project split"
R4D_MODE=splits "${VENV}/bin/python" "$S"
echo "### plans"
head -80 "${nnUNet_preprocessed}/Dataset502_ACDC/nnUNetPlans.json"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
