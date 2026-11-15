#!/bin/bash
#SBATCH --job-name=rooted4d-4dmrdbg
#SBATCH --account=bhio-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=48g
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
D="${NVME_ROOT}/external/4D-Myocardium-Reconstruction-with-Decoupled-Motion-and-Shape-Model"
VENV="${NVME_ROOT}/external/.venv-4dmr"
cd "${NVME_ROOT}/code"; mkdir -p logs
RESULTS="${NVME_ROOT}/results/${SLURM_JOB_ID}_4dmrdbg"; mkdir -p "${RESULTS}"

# Three representative failures: two "did not reconnect" with different component
# counts, and one "no boundary loop". Artifacts are retained so the geometry can
# be inspected instead of guessed at.
CASES="${R4D_DEBUG_CASES:-patient014 patient025 patient037}"

"${VENV}/bin/python" - <<PY
import json, pathlib
cfg = pathlib.Path("${D}/configs/cases.json")
d = json.loads(cfg.read_text())
d.setdefault("cases", {})
for c in "${CASES}".split():
    d["cases"][c] = {"retain_artifacts": True,
                     "stages": {"preprocess": True, "inference": True,
                                "postprocess": True, "video": False}}
cfg.write_text(json.dumps(d, indent=2) + "\n")
print("marked for artifact retention:", "${CASES}")
PY

cd "$D"
for c in ${CASES}; do
  echo "### ${c}"
  "${VENV}/bin/python" run_pipeline.py --case "${c}" > "${RESULTS}/${c}.log" 2>&1 || true
  tail -3 "${RESULTS}/${c}.log"
  echo "  artifacts: $(find outputs/${c} -type f 2>/dev/null | wc -l) files"
done
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
