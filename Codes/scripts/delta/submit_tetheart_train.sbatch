#!/bin/bash
#SBATCH --job-name=rooted4d-thtrain
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
T="${NVME_ROOT}/external/TetHeart"
VENV="${NVME_ROOT}/external/.venv-tetheart"
cd "${NVME_ROOT}/code"; mkdir -p logs
RESULTS="${NVME_ROOT}/results/${SLURM_JOB_ID}_thtrain"; mkdir -p "${RESULTS}"

CONFIG="${T}/configs/Unified/tet/static/rooted4d_acdc_static.py"
WORKDIR="${R4D_WORKDIR:-${NVME_ROOT}/tetheart_work/rooted4d_acdc_static}"
ITERS="${R4D_ITERS:-}"
mkdir -p "${WORKDIR}"

[ -f "${CONFIG}" ] || { echo "config missing: ${CONFIG}"; exit 1; }
"${VENV}/bin/python" -c "import apex, tetheart; assert getattr(apex,'__apex_shim__',False)" || {
  echo "apex shim not installed in the TetHeart venv"; exit 1; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "config:  ${CONFIG}"
echo "workdir: ${WORKDIR}"
echo "NOTE: the config names FusedAdam; here that is the shim and resolves to torch AdamW."

# Their evaluation hook runs every 1000 iterations from iteration 1000 and
# selects a checkpoint by validation Chamfer distance. The first evaluation
# killed the run: at 1000 iterations the marching-tetrahedra output is still
# empty, and sample_points_from_meshes raises "Meshes are empty." Training
# itself was healthy, with the loss falling from 6.40 to 0.37.
#
# Delaying the first evaluation would only narrow the window, since a single
# empty mesh at any later evaluation would kill a ten-hour run. Instead
# validation is disabled here and checkpoints are kept every 10000 iterations,
# and the checkpoint is selected afterwards on our own validation patients using
# the metric this paper reports -- the same way GHDHeart's capacity was chosen.
# Note that lr_config is None in this config, so the learning rate is constant
# and the final checkpoint is not automatically the best one; selection matters.
declare -a EXTRA=(--no-validate)
declare -a OPTIONS=(checkpoint_config.interval=10000 checkpoint_config.max_keep_ckpts=20)
if [ -n "${ITERS}" ]; then
  OPTIONS+=("runner.max_iters=${ITERS}")
  echo "smoke test: max_iters=${ITERS}"
fi
EXTRA+=(--cfg-options "${OPTIONS[@]}")

cd "$T"
# Under `set -u` an empty array expanded as "${EXTRA[@]:-}" yields one empty
# string argument, which argparse rejects as unrecognised. The smoke test passed
# only because the array was non-empty there. Branch instead of expanding.
if [ ${#EXTRA[@]} -gt 0 ]; then
  "${VENV}/bin/python" tools/train.py "${CONFIG}" \
    --work-dir "${WORKDIR}" --seed 0 "${EXTRA[@]}" \
    > "${RESULTS}/train.log" 2>&1 || {
      echo "training failed; last 40 lines:"; tail -40 "${RESULTS}/train.log"; exit 1; }
else
  "${VENV}/bin/python" tools/train.py "${CONFIG}" \
    --work-dir "${WORKDIR}" --seed 0 \
    > "${RESULTS}/train.log" 2>&1 || {
      echo "training failed; last 40 lines:"; tail -40 "${RESULTS}/train.log"; exit 1; }
fi

echo "=== last lines"; tail -20 "${RESULTS}/train.log"
echo "=== checkpoints"; ls -la "${WORKDIR}"/*.pth 2>/dev/null | awk '{print $5, $9}'
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
