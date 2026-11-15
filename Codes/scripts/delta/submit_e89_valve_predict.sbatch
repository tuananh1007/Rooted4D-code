#!/bin/bash
# E89 part 2 (Docs/E89_MEASURED_ERROR_PREREGISTRATION_2026-09-28.md): the single
# prediction of the 35 test-partition valve donors with part 1's fold-0 model,
# with E42's settings (-f 0 --disable_tta).
#SBATCH --job-name=rooted4d-e89-predict
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
if [ "${R4D_ALLOW_TEST:-0}" != "1" ]; then echo "refusing to predict test donors without R4D_ALLOW_TEST=1"; exit 1; fi
NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
cd "${NVME_ROOT}/code"; mkdir -p logs
grep -E '^(commit|dirty)=' PROVENANCE.txt
export nnUNet_raw="${HDD_DATA}/nnunet/raw"
export nnUNet_preprocessed="${HDD_DATA}/nnunet/preprocessed"
export nnUNet_results="${NVME_ROOT}/nnunet_results"
VENV="${NVME_ROOT}/external/.venv-nnunet"
OUT="${HDD_DATA}/nnunet/predictions_mvsegbench_test"
MODEL="${nnUNet_results}/Dataset503_MVSegBench/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
[ -f "$MODEL" ] || { echo "no final checkpoint at $MODEL; training has not finished"; exit 1; }
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then echo "$OUT is not empty; test is predicted once"; exit 1; fi
mkdir -p "$OUT"
"${VENV}/bin/nnUNetv2_predict" -i "${nnUNet_raw}/Dataset503_MVSegBench/imagesTs" -o "$OUT" -d 503 -c 3d_fullres \
  -f 0 --disable_tta > "logs/e89_predict_${SLURM_JOB_ID}.log" 2>&1 \
  || { echo "predict failed"; tail -30 "logs/e89_predict_${SLURM_JOB_ID}.log"; exit 1; }
ls "$OUT" | grep -c "nii.gz"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
