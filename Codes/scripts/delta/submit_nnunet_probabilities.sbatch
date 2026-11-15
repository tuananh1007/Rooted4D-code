#!/bin/bash
#SBATCH --job-name=rooted4d-nnprob
#SBATCH --account=bhio-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bhio/avu2/Rooted4D/code/logs/%x_%j.out
#SBATCH --error=/work/nvme/bhio/avu2/Rooted4D/code/logs/%x_%j.err
set -euo pipefail

# E81 (qualitative figures): the frozen fold-0 nnU-Net's per-voxel probabilities
# for a few MVSeg2023 test cases, with E42's settings plus --save_probabilities.
# The labels it writes are E42's predictions again; nothing here is scored.
# Cases are passed as R4D_CASES=test_027+test_026+... (sbatch --export splits on commas).

NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
export nnUNet_raw="${HDD_DATA}/nnunet/raw"
export nnUNet_preprocessed="${HDD_DATA}/nnunet/preprocessed"
export nnUNet_results="${NVME_ROOT}/nnunet_results"
VENV="${NVME_ROOT}/external/.venv-nnunet"
IMAGES="${HDD_DATA}/nnunet/images_qualitative"
OUT="${HDD_DATA}/nnunet/probabilities_qualitative"
rm -rf "${IMAGES}"; mkdir -p "${IMAGES}" "${OUT}"  # earlier cases' outputs in OUT are kept
IFS='+' read -ra CASES <<< "${R4D_CASES:?set R4D_CASES}"
for case in "${CASES[@]}"; do
  cp -f "${nnUNet_raw}/Dataset501_MVSeg/imagesTs/${case}_0000.nii.gz" "${IMAGES}/"
done
"${VENV}/bin/nnUNetv2_predict" -i "${IMAGES}" -o "${OUT}" -d 501 -c 3d_fullres -f 0 --disable_tta \
  --save_probabilities > "${NVME_ROOT}/code/logs/nnprob_${SLURM_JOB_ID}.log" 2>&1 \
  || { echo "predict failed"; tail -30 "${NVME_ROOT}/code/logs/nnprob_${SLURM_JOB_ID}.log"; exit 1; }
ls -la "${OUT}"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
