#!/bin/bash
#SBATCH --job-name=rooted4d-nnval
#SBATCH --account=bhio-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

# Stage C (E79): the frozen fold-0 nnU-Net on the 30 MVSeg2023 validation cases,
# with E42's settings. The model trained on the 105 training cases only; E41 kept
# the validation cases out of nnU-Net's raw dataset entirely, so they are staged
# here in a separate folder rather than added to Dataset501.

NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
cd "${NVME_ROOT}/code"; mkdir -p logs
grep -q "^dirty=no" PROVENANCE.txt || { echo "refusing: synced tree was dirty"; exit 1; }
head -2 PROVENANCE.txt

export nnUNet_raw="${HDD_DATA}/nnunet/raw"
export nnUNet_preprocessed="${HDD_DATA}/nnunet/preprocessed"
export nnUNet_results="${NVME_ROOT}/nnunet_results"
VENV="${NVME_ROOT}/external/.venv-nnunet"
IMAGES="${HDD_DATA}/nnunet/images_val"
PRED="${HDD_DATA}/nnunet/predictions_val"
mkdir -p "${IMAGES}" "${PRED}"

for image in "${HDD_DATA}"/raw/mvseg2023/val/*-US.nii.gz; do
  name=$(basename "${image}" -US.nii.gz)
  cp -f "${image}" "${IMAGES}/${name}_0000.nii.gz"
done
echo "staged: $(ls ${IMAGES}/*_0000.nii.gz | wc -l) validation images"

"${VENV}/bin/nnUNetv2_predict" -i "${IMAGES}" -o "${PRED}" -d 501 -c 3d_fullres -f 0 --disable_tta \
  > "logs/nnval_predict_${SLURM_JOB_ID}.log" 2>&1 || { echo "predict failed"; tail -30 "logs/nnval_predict_${SLURM_JOB_ID}.log"; exit 1; }
echo "predicted: $(ls ${PRED}/val_*.nii.gz 2>/dev/null | wc -l) cases"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
