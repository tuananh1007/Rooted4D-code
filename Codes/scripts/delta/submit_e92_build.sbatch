#!/bin/bash
# E92 (Docs/E92_INTERVENTIONS_PREREGISTRATION_2026-09-28.md): the nine
# intervention instances of the realistic-session benchmark. E100 builds its own
# from the rebuilt primary's configs (its amendment):
#   sbatch --export=ALL,R4D_E92_CONFIG_DIR=Codes/configs/e100/e92,R4D_E92_PREFIX=data/derived/e100-v4n96o-e92,R4D_E92_ORIGINAL=data/derived/e100-v4n96o,R4D_E92_ORIENTED=1 submit_e92_build.sbatch
#SBATCH --job-name=rooted4d-e92-build
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96g
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
cd "${NVME_ROOT}/code"; mkdir -p logs
if [ -e data ] && [ ! -L data ]; then echo "data is not a symlink, refusing"; exit 1; fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then ln -sfT "${HDD_DATA}" data; fi
source .venv/bin/activate
export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CONFIG_DIR="${R4D_E92_CONFIG_DIR:-Codes/configs/e92}"
PREFIX="${R4D_E92_PREFIX:-data/derived/semisynthetic-real-v4-e92}"
ORIGINAL="${R4D_E92_ORIGINAL:-data/derived/semisynthetic-real-v4}"
grep -E '^(commit|dirty)=' PROVENANCE.txt
if grep -q '^dirty=yes' PROVENANCE.txt; then echo "refusing: dirty tree"; exit 1; fi
build() {
  local config="$1" name out rc=0
  name="$(basename "${config}" .yaml)"; name="${name#semisynthetic_real_donors_v4_}"
  out="${PREFIX}-${name//_/-}"
  if [ -f "${out}/audit.json" ]; then echo "${out} exists"; return 0; fi
  if python -m rooted4d.synthetic.dataset_cli generate --config "${config}" --output "${out}" \
      > "logs/e92_build_${name}_${SLURM_JOB_ID}.json" 2>&1; then rc=0; else rc=$?; fi
  echo "${out}: exit ${rc}"
  return "${rc}"
}
pids=()
for config in "${CONFIG_DIR}"/semisynthetic_real_donors_v4_*.yaml; do build "${config}" & pids+=($!); done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
echo "### test samples per instance (same sample ids as the original?)"
for out in "${PREFIX}"-*; do
  same=$(diff <(ls "${out}/samples/test") <(ls "${ORIGINAL}/samples/test") > /dev/null && echo yes || echo no)
  echo "${out}: $(ls "${out}/samples/test" | wc -l) test samples, same ids: ${same}"
done
if [ "${R4D_E92_ORIENTED:-0}" = "1" ]; then
  echo "### E100: every sample's free edge on the ventricular side"
  python - "${PREFIX}" <<'PY' || status=1
import sys
from pathlib import Path
import numpy as np
from rooted4d.experiments.fitted_geometry import ring_groups
failed = False
for out in sorted(Path(sys.argv[1]).parent.glob(Path(sys.argv[1]).name + "-*")):
    heights = []
    for path in sorted((out / "samples" / "test").glob("*.npz")):
        x = np.load(path)["template_vertices"]
        g = ring_groups((len(x) - 1) // 3)
        ring = x[g["annulus"]].mean(0); apex = x[g["apex"][0]]
        axis = (apex - ring) / np.linalg.norm(apex - ring)
        heights.append(float(((x[g["free"]] - ring) @ axis).mean()))
    heights = np.array(heights)
    failed |= bool((heights <= 0).any())
    print(f"{out}: free-edge height {heights.mean():+.1f} mm [{heights.min():+.1f}, {heights.max():+.1f}]")
if failed:
    raise SystemExit("a non-positive free-edge height")
PY
fi
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ) status ${status}"
exit "${status}"
