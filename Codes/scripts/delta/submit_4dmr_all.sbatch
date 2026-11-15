#!/bin/bash
#SBATCH --job-name=rooted4d-4dmrall
#SBATCH --account=bhio-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
D="${NVME_ROOT}/external/4D-Myocardium-Reconstruction-with-Decoupled-Motion-and-Shape-Model"
VENV="${NVME_ROOT}/external/.venv-4dmr"
cd "${NVME_ROOT}/code"; mkdir -p logs
RESULTS="${NVME_ROOT}/results/${SLURM_JOB_ID}_4dmrall"; mkdir -p "${RESULTS}"

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

# Ground truth is generated for every ACDC patient, because TetHeart trains on
# the split lock's training patients and is scored on its test patients.
export R4D_ACDC="${HDD_DATA}/raw/acdc"
export R4D_4DMR_INPUTS="${D}/inputs"
export R4D_SPLITS="${NVME_ROOT}/code/results_seed/acdc_patient_splits.json"
export R4D_SPLIT_NAMES="train,val,test"
export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"

echo "### stage every ACDC patient as a 4DMR case"
"${VENV}/bin/python" Codes/scripts/experiments/e43_acdc_to_4dmr.py
cp -f "${D}/inputs/cases.json" "${D}/configs/cases.json"

nvidia-smi --query-gpu=name --format=csv,noheader || true
echo "### 4DMR inference over all staged cases"
cd "$D"
START=$(date +%s)
"${VENV}/bin/python" run_pipeline.py > "${RESULTS}/4dmr_all.log" 2>&1 || {
  echo "pipeline reported failure; tail:"; tail -40 "${RESULTS}/4dmr_all.log"; }
END=$(date +%s)
echo "elapsed: $((END-START)) s"

# The report lists per-case status, which is what the converter's gate reads.
python - <<'PY'
import json, glob, os
reports = sorted(glob.glob("outputs/*/report.json"))
print("cases with a report: %d" % len(reports))
bad = []
for r in reports:
    try:
        d = json.load(open(r))
    except Exception:
        bad.append(r); continue
print("unreadable reports: %d" % len(bad))
vtk = sorted(glob.glob("outputs/*/vtk/frame_*.vtk"))
print("vtk meshes written: %d" % len(vtk))
PY
cp -f "${RESULTS}/4dmr_all.log" "${RESULTS}/" 2>/dev/null || true
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
