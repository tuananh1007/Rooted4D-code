#!/bin/bash
#SBATCH --job-name=rooted4d-threscore
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
VENV="${NVME_ROOT}/external/.venv-tetheart"
RES="${NVME_ROOT}/results/22102244_thfinal"
cd "${NVME_ROOT}/code"; mkdir -p logs
export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export R4D_ACDC="${HDD_DATA}/raw/acdc"
export R4D_TET_FRAME_REPORT="${NVME_ROOT}/results/22045910_tetframe/E50_tet_frame.json"
export R4D_OUT="${RES}"
# Eight workers, not thirty-two. A single scoring process peaks near 8.4 GB --
# point_to_mesh_mm holds several 512 x 60000 x 3 temporaries at once -- so 32 of
# them reached exactly the 96 GB limit and the step was OOM-killed, which in turn
# wedged the pool waiting on results from children the kernel had taken. Eight
# workers sit near 67 GB and leave headroom for a larger-than-average mesh.
# One BLAS thread per worker: processes each spawning their own thread pool
# oversubscribes the node and runs slower than serial.
export R4D_WORKERS=8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Checkpoint selection is redone here: E49 selected on a metric whose class check
# was invalid, so the selection it made is not one we can inherit.
for iters in 40000 60000 80000 100000; do
  d="${RES}/preds_val_iter_${iters}"
  [ -d "$d" ] || { echo "missing ${d}"; continue; }
  echo "--- val iter_${iters}  $(date -u +%H:%M:%S)"
  R4D_PRED_DIR="$d" R4D_OUTNAME="E49d_val_iter_${iters}.json" \
    "${VENV}/bin/python" Codes/scripts/experiments/e49d_tetheart_rescore.py \
    > "${RES}/e49d_val_iter_${iters}.log" 2>&1 || echo "  FAILED"
  "${VENV}/bin/python" - "${RES}/E49d_val_iter_${iters}.json" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
if p.exists():
    s = json.loads(p.read_text())["summary"]
    print("    scored %d/%d  clipped %.4f  raw %.4f" % (
        s["scored"], s["groups"],
        s["symmetric_mm_clipped_with_endcaps"]["median"],
        s["symmetric_mm_raw_with_endcaps"]["median"]))
else:
    print("    no output")
PY
done

echo "### select on validation, by the clipped end-cap-inclusive median"
BEST=$("${VENV}/bin/python" - <<'PY'
import glob, json
best = None
for path in sorted(glob.glob("/work/nvme/bhio/avu2/Rooted4D/results/22102244_thfinal/E49d_val_iter_*.json")):
    s = json.load(open(path))["summary"].get("symmetric_mm_clipped_with_endcaps")
    if s and (best is None or s["median"] < best[0]):
        best = (s["median"], path.split("E49d_val_")[1].replace(".json", ""))
print(best[1] if best else "")
PY
)
echo "selected: ${BEST}"

echo "--- test ${BEST}  $(date -u +%H:%M:%S)"
if [ -d "${RES}/preds_test_${BEST}" ]; then
  R4D_PRED_DIR="${RES}/preds_test_${BEST}" R4D_OUTNAME="E49d_test_${BEST}.json" \
    "${VENV}/bin/python" Codes/scripts/experiments/e49d_tetheart_rescore.py \
    > "${RES}/e49d_test_${BEST}.log" 2>&1 || echo "  FAILED"
  sed -n '/^{/,$p' "${RES}/e49d_test_${BEST}.log" | tail -70
else
  echo "no test predictions for ${BEST}; the GPU test pass would have to be rerun"
fi
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
