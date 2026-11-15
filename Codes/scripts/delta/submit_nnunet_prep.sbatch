#!/bin/bash
#SBATCH --job-name=rooted4d-nnprep
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

NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
cd "${NVME_ROOT}/code"; mkdir -p logs

# Several jobs share this working directory, so the data symlink must not be
# rewritten on every run: replacing it, even with an identical target, opens a
# window in which a concurrent job reading through it gets a FileNotFoundError
# on a file that exists. Only create it when it is absent or points elsewhere.
if [ -e data ] && [ ! -L data ]; then
  echo "data exists and is not a symlink, refusing to run"; exit 1
fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then
  ln -sfT "${HDD_DATA}" data
fi
[ -L data ] || { echo "data symlink missing"; exit 1; }

# nnU-Net's preprocessed tensors are large and are intermediate, so they live on
# the HDD tier with the data rather than on NVMe with the code.
export nnUNet_raw="${HDD_DATA}/nnunet/raw"
export nnUNet_preprocessed="${HDD_DATA}/nnunet/preprocessed"
export nnUNet_results="${NVME_ROOT}/nnunet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

VENV="${NVME_ROOT}/external/.venv-nnunet"
export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export R4D_MVSEG="data/raw/mvseg2023"

echo "### convert"
"${VENV}/bin/python" Codes/scripts/experiments/e41_nnunet_convert.py

echo "### plan and preprocess"
"${VENV}/bin/nnUNetv2_plan_and_preprocess" -d 501 --verify_dataset_integrity -np 8 -c 3d_fullres

echo "### plans"
cat "${nnUNet_preprocessed}/Dataset501_MVSeg/nnUNetPlans.json" | head -60
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
