#!/bin/bash
#SBATCH --job-name=rooted4d-4dmrfix
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
D="${NVME_ROOT}/external/4D-Myocardium-Reconstruction-with-Decoupled-Motion-and-Shape-Model"
VENV="${NVME_ROOT}/external/.venv-4dmr"
cd "${NVME_ROOT}/code"; mkdir -p logs
RESULTS="${NVME_ROOT}/results/${SLURM_JOB_ID}_4dmrfix"; mkdir -p "${RESULTS}"

if [ -e data ] && [ ! -L data ]; then echo "data is not a symlink, refusing"; exit 1; fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then ln -sfT "${HDD_DATA}" data; fi

# The patients whose postprocessing raised a GeometryError on the full run.
FAILED=$(grep -oE "\[case\] failed patient[0-9]+" \
  "${NVME_ROOT}/results/21977471_4dmrall/4dmr_all.log" | grep -oE "patient[0-9]+" | sort -u | tr '\n' ' ')
echo "failed cases: ${FAILED}"
echo "count: $(echo ${FAILED} | wc -w)"

# Stage 1: re-run inference for those cases with artifacts retained, so the raw
# surfaces survive for cleaning. Postprocess is disabled here because it is what
# fails; it runs again in stage 3 against the cleaned meshes.
"${VENV}/bin/python" - <<PY
import json, pathlib
cfg = pathlib.Path("${D}/configs/cases.json")
d = json.loads(cfg.read_text()); d.setdefault("cases", {})
for c in "${FAILED}".split():
    d["cases"][c] = {"retain_artifacts": True,
                     "stages": {"preprocess": True, "inference": True,
                                "postprocess": False, "video": False}}
cfg.write_text(json.dumps(d, indent=2) + "\n")
print("configured", len(d["cases"]), "cases for inference-only with retention")
PY

cd "$D"
for c in ${FAILED}; do
  "${VENV}/bin/python" run_pipeline.py --case "${c}" > "${RESULTS}/infer_${c}.log" 2>&1 || \
    echo "  inference failed for ${c}"
done
echo "raw surfaces present: $(find outputs -path '*artifacts/raw*' -name '*.obj' | wc -l)"

# Stage 2: remove debris components below 4DMR's own negligible-artifact ratio.
cd "${NVME_ROOT}/code"
export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export R4D_4DMR_DIR="${D}"
export R4D_CLEAN_CASES="${FAILED}"
export R4D_OUT="${RESULTS}"
"${VENV}/bin/python" Codes/scripts/experiments/e45_clean_4dmr_debris.py \
  > "${RESULTS}/e45.log" 2>&1 || { echo "e45 failed"; tail -20 "${RESULTS}/e45.log"; exit 1; }
tail -25 "${RESULTS}/e45.log"

# Stage 3: postprocess only, now reading the cleaned raw surfaces.
"${VENV}/bin/python" - <<PY
import json, pathlib
cfg = pathlib.Path("${D}/configs/cases.json")
d = json.loads(cfg.read_text())
for c in "${FAILED}".split():
    d["cases"][c] = {"retain_artifacts": True,
                     "stages": {"preprocess": False, "inference": False,
                                "postprocess": True, "video": False}}
cfg.write_text(json.dumps(d, indent=2) + "\n")
PY
cd "$D"
recovered=0
for c in ${FAILED}; do
  if "${VENV}/bin/python" run_pipeline.py --case "${c}" > "${RESULTS}/post_${c}.log" 2>&1; then
    recovered=$((recovered+1))
  else
    echo "  still failing: ${c}: $(grep -oE '"error": "[^"]{0,70}' ${RESULTS}/post_${c}.log | head -1)"
  fi
done
echo "recovered ${recovered} of $(echo ${FAILED} | wc -w) cases"
echo "total vtk meshes now: $(find outputs -name 'frame_*.vtk' | wc -l)"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
