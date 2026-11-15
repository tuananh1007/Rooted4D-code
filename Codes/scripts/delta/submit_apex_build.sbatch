#!/bin/bash
#SBATCH --job-name=rooted4d-apex
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

# apex only needs nvcc and a C++ toolchain to build, not a device, so this runs
# on CPU and leaves the GPU allocation for training. TORCH_CUDA_ARCH_LIST is
# restricted to 8.0 because A100 is what we train on; compiling every
# architecture is the slow default their INSTALL.md warns about.
NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
EXT="${NVME_ROOT}/external"
VENV="${EXT}/.venv-tetheart"
cd "${NVME_ROOT}/code"; mkdir -p logs
RESULTS="${NVME_ROOT}/results/${SLURM_JOB_ID}_apex"; mkdir -p "${RESULTS}"

# The module system leaves nvcc 13.2 on PATH regardless of which cudatoolkit is
# loaded, and apex refuses a CUDA major-version mismatch against torch. torch is
# cu121, so point CUDA_HOME at the 12.9 toolkit explicitly: same major version,
# which is what apex checks.
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="${CUDA_HOME}/bin:${PATH}"
[ -x "${CUDA_HOME}/bin/nvcc" ] || { echo "no nvcc at ${CUDA_HOME}"; exit 1; }
echo "nvcc: $(which nvcc)"; nvcc --version | tail -2

# uv creates venvs without pip, and apex's build is driven through pip.
"${VENV}/bin/python" -m pip --version >/dev/null 2>&1 || \
  uv pip install --python "${VENV}/bin/python" pip

# torch is cu121 and the toolkit here is 12.9. apex compares major versions, so
# this is a minor mismatch it tolerates with a warning; recorded because a
# silent mismatch would be a plausible cause of later numerical oddities.
"${VENV}/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=8
cd "${EXT}/apex"
"${VENV}/bin/python" -m pip install -v --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" \
  --config-settings "--build-option=--cuda_ext" \
  . > "${RESULTS}/apex_build.log" 2>&1 || {
    echo "apex build FAILED; last 40 lines:"; tail -40 "${RESULTS}/apex_build.log"
    echo "FusedAdam will be unavailable; the training config must switch to torch AdamW."
    exit 1; }

echo "=== verify"
"${VENV}/bin/python" -c "
import apex
from apex.optimizers import FusedAdam
print('apex ok, FusedAdam importable')
" 2>&1 | tail -3
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
