#!/bin/bash
#SBATCH --job-name=rooted4d-nneval
#SBATCH --account=bhio-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
cd "${NVME_ROOT}/code"; mkdir -p logs
RESULTS="${NVME_ROOT}/results/${SLURM_JOB_ID}_nneval"; mkdir -p "${RESULTS}"

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

export nnUNet_raw="${HDD_DATA}/nnunet/raw"
export nnUNet_preprocessed="${HDD_DATA}/nnunet/preprocessed"
export nnUNet_results="${NVME_ROOT}/nnunet_results"
VENV="${NVME_ROOT}/external/.venv-nnunet"
PRED="${HDD_DATA}/nnunet/predictions_test"
mkdir -p "${PRED}"

echo "### inference on the 40 challenge test cases, fold 0"
"${VENV}/bin/nnUNetv2_predict" \
  -i "${nnUNet_raw}/Dataset501_MVSeg/imagesTs" \
  -o "${PRED}" -d 501 -c 3d_fullres -f 0 --disable_tta \
  > "${RESULTS}/predict.log" 2>&1 || { echo "predict failed"; tail -30 "${RESULTS}/predict.log"; exit 1; }
echo "predicted: $(ls ${PRED}/*.nii.gz 2>/dev/null | wc -l) cases"

echo "### downstream: our fitting behind nnU-Net vs behind reference labels"
export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export R4D_MVSEG_TEST="${HDD_DATA}/raw/mvseg2023/test"
export R4D_NNUNET_PRED="${PRED}"
export R4D_OUT="${RESULTS}"
"${VENV}/bin/python" Codes/scripts/experiments/e42_nnunet_valve.py \
  > "${RESULTS}/e42.log" 2>&1 || { echo "e42 failed"; tail -30 "${RESULTS}/e42.log"; exit 1; }
tail -30 "${RESULTS}/e42.log"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
