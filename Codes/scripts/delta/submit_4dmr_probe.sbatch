#!/bin/bash
#SBATCH --job-name=rooted4d-4dmrprobe
#SBATCH --account=bhio-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=48g
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
D="${NVME_ROOT}/external/4D-Myocardium-Reconstruction-with-Decoupled-Motion-and-Shape-Model"
VENV="${NVME_ROOT}/external/.venv-4dmr"
cd "${NVME_ROOT}/code"; mkdir -p logs
RESULTS="${NVME_ROOT}/results/${SLURM_JOB_ID}_4dmr"; mkdir -p "${RESULTS}"

# The generated case configuration belongs in configs/, which is where the
# pipeline reads it from; e43 writes it beside the inputs so it travels with them.
cp -f "${D}/inputs/cases.json" "${D}/configs/cases.json"

for f in "${VENV}/bin/python" "${D}/run_pipeline.py" \
         "${D}/examples/acdc/4DMM/endo/ModelParameters/latest.pth" \
         "${D}/examples/acdc/4DMM/epi/ModelParameters/latest.pth"; do
  [ -e "$f" ] || { echo "missing required input: $f"; exit 1; }
done

nvidia-smi --query-gpu=name --format=csv,noheader || true
CASE="${R4D_4DMR_CASE:-patient007}"
echo "timing one case: ${CASE}"
cd "$D"
START=$(date +%s)
"${VENV}/bin/python" run_pipeline.py --case "${CASE}" > "${RESULTS}/4dmr_${CASE}.log" 2>&1 || {
  echo "pipeline failed; tail:"; tail -40 "${RESULTS}/4dmr_${CASE}.log"; exit 1; }
END=$(date +%s)
echo "elapsed: $((END-START)) s for one case"
tail -20 "${RESULTS}/4dmr_${CASE}.log"
echo "=== outputs:"
find "${D}/outputs/${CASE}" -type f 2>/dev/null | head -12
cp -r "${D}/outputs/${CASE}" "${RESULTS}/" 2>/dev/null || true
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
