#!/bin/bash
#SBATCH --job-name=rooted4d-nntrain
#SBATCH --account=bhio-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=2-00:00:00
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

export nnUNet_raw="${HDD_DATA}/nnunet/raw"
export nnUNet_preprocessed="${HDD_DATA}/nnunet/preprocessed"
export nnUNet_results="${NVME_ROOT}/nnunet_results"
export nnUNet_n_proc_DA="${SLURM_CPUS_PER_TASK}"
VENV="${NVME_ROOT}/external/.venv-nnunet"

# nnU-Net enables torch.compile by default, which needs triton, which in turn
# imports setuptools at runtime. uv creates minimal venvs without setuptools, so
# the first attempt died 51 seconds in with "Cannot find a working triton
# installation" -- a misleading message, since triton was installed and its
# missing dependency was setuptools. Assert it here rather than discover it
# again after a queue wait.
"${VENV}/bin/python" -c "import triton, setuptools" || {
  echo "triton/setuptools missing; either fix the venv or set nnUNet_compile=f"; exit 1; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# nnU-Net's own default schedule, 1000 epochs, so the comparison cannot be
# dismissed as under-training. Fold 0 only: five folds is five times the compute
# for a baseline row, and nnU-Net's single fold is normally within a point of the
# ensemble. "-c" resumes from the latest checkpoint if this job is requeued.
FOLD="${R4D_FOLD:-0}"
# The dataset defaults to the valve front end (E41/E42); E84 sets 502 and Dataset502_ACDC.
DATASET_ID="${R4D_DATASET_ID:-501}"
DATASET_NAME="${R4D_DATASET_NAME:-Dataset501_MVSeg}"
RESUME=""
if [ -f "${nnUNet_results}/${DATASET_NAME}/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_${FOLD}/checkpoint_latest.pth" ]; then
  RESUME="-c"
  echo "resuming from existing checkpoint"
fi

echo "training ${DATASET_NAME} fold ${FOLD}, 3d_fullres, default trainer"
"${VENV}/bin/nnUNetv2_train" "${DATASET_ID}" 3d_fullres "${FOLD}" ${RESUME} --npz
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
