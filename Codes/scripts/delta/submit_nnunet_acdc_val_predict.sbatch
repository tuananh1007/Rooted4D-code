#!/bin/bash
#SBATCH --job-name=rooted4d-e86-valpred
#SBATCH --account=bhio-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

# E86 (Docs/E86_CAVITY_COMPONENTS_PREREGISTRATION_2026-09-28.md): the E84 nnU-Net's
# labels for the 60 frames of the 30 validation patients, with E84's test settings
# (-f 0 --disable_tta). nnU-Net saw these frames only as its own validation fold.

NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
cd "${NVME_ROOT}/code"; mkdir -p logs
export nnUNet_raw="${HDD_DATA}/nnunet/raw"
export nnUNet_preprocessed="${HDD_DATA}/nnunet/preprocessed"
export nnUNet_results="${NVME_ROOT}/nnunet_results"
VENV="${NVME_ROOT}/external/.venv-nnunet"
IMAGES="${HDD_DATA}/nnunet/images_acdc_val"
OUT="${HDD_DATA}/nnunet/predictions_acdc_val"
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then echo "$OUT is not empty; refusing to overwrite"; exit 1; fi
rm -rf "$IMAGES"; mkdir -p "$IMAGES" "$OUT"
"${VENV}/bin/python" - <<EOF
import json, shutil
from pathlib import Path
splits = json.loads(Path("${NVME_ROOT}/code/results_seed/acdc_patient_splits.json").read_text())
fold = json.loads(Path("${nnUNet_preprocessed}/Dataset502_ACDC/splits_final.json").read_text())[0]
raw = Path("${nnUNet_raw}/Dataset502_ACDC/imagesTr")
for case in fold["val"]:
    assert splits[case.split("_")[0]] == "val", case
    shutil.copy(raw / f"{case}_0000.nii.gz", Path("${IMAGES}") / f"{case}_0000.nii.gz")
print("copied", len(fold["val"]), "validation frames")
EOF
"${VENV}/bin/nnUNetv2_predict" -i "$IMAGES" -o "$OUT" -d 502 -c 3d_fullres -f 0 --disable_tta \
  > "logs/e86_valpred_${SLURM_JOB_ID}.log" 2>&1 \
  || { echo "predict failed"; tail -30 "logs/e86_valpred_${SLURM_JOB_ID}.log"; exit 1; }
ls "$OUT" | grep -c "nii.gz"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
