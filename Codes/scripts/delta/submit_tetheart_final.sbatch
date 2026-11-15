#!/bin/bash
#SBATCH --job-name=rooted4d-thfinal
#SBATCH --account=bhio-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
T="${NVME_ROOT}/external/TetHeart"
VENV="${NVME_ROOT}/external/.venv-tetheart"
WORK="${NVME_ROOT}/tetheart_work/rooted4d_acdc_static"
cd "${NVME_ROOT}/code"; mkdir -p logs
RESULTS="${NVME_ROOT}/results/${SLURM_JOB_ID}_thfinal"; mkdir -p "${RESULTS}"

if [ -e data ] && [ ! -L data ]; then echo "data is not a symlink, refusing"; exit 1; fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then ln -sfT "${HDD_DATA}" data; fi

export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export R4D_ACDC="${HDD_DATA}/raw/acdc"
export R4D_OUT="${RESULTS}"
# E49 needs the per-frame transform E50 recorded, to return predictions from the
# preprocessed 64^3 frame to millimetres. Without it it refuses to score.
export R4D_TET_FRAME_REPORT="${NVME_ROOT}/results/22045910_tetframe/E50_tet_frame.json"
[ -f "${R4D_TET_FRAME_REPORT}" ] || { echo "missing E50 transform report"; exit 1; }

# MRIDataset's test_mode takes 1/nfolds of a root, so two folds at nfolds=2
# cover a root exactly once.
score_root () {
  local root="$1" tag="$2" ckpt="$3"
  local stem; stem="$(basename "${ckpt}" .pth)"
  local cfg="${T}/configs/Unified/tet/static/rooted4d_eval_${tag}.py"
  "${VENV}/bin/python" - "$root" "$cfg" <<'PY'
import re, sys, pathlib
root, dest = sys.argv[1], sys.argv[2]
src = pathlib.Path("/work/nvme/bhio/avu2/Rooted4D/external/TetHeart/configs/Unified/tet/static/rooted4d_acdc_static.py")
text = re.sub(r"^data_root\s*=.*$", "data_root = %r" % root,
              src.read_text(), flags=re.M).replace("nfolds=5", "nfolds=2")
pathlib.Path(dest).write_text(text)
PY
  local preds="${RESULTS}/preds_${tag}_${stem}"
  mkdir -p "${preds}"
  for fold in 0 1; do
    (cd "$T" && "${VENV}/bin/python" tools/test.py "${cfg}" "${ckpt}" \
       --show --show-dir "${preds}" --cfg-options "fold=${fold}" \
       > "${RESULTS}/test_${tag}_${stem}_f${fold}.log" 2>&1) \
      || echo "    test.py failed: ${tag} ${stem} fold ${fold}"
  done
  local n; n=$(ls "${preds}"/*_pred.obj 2>/dev/null | wc -l)
  local nonempty; nonempty=$(find "${preds}" -name '*_pred.obj' -size +0 2>/dev/null | wc -l)
  echo "    ${n} meshes written, ${nonempty} non-empty"
  R4D_PRED_DIR="${preds}" R4D_OUTNAME="E49_${tag}_${stem}.json" \
    "${VENV}/bin/python" Codes/scripts/experiments/e49_tetheart_eval.py \
    > "${RESULTS}/e49_${tag}_${stem}.log" 2>&1 \
    || { echo "    e49 failed"; tail -8 "${RESULTS}/e49_${tag}_${stem}.log"; return 0; }
  sed -n '/^{/,/^}/p' "${RESULTS}/e49_${tag}_${stem}.log" | tail -30
}

echo "### checkpoint selection on our validation split (never trained on)"
for iters in 40000 60000 80000 100000; do
  ck="${WORK}/iter_${iters}.pth"
  [ -f "$ck" ] || { echo "  missing ${ck}"; continue; }
  echo "--- iter_${iters}"
  score_root "${HDD_DATA}/tetheart_914_3d_val" "val" "$ck"
done

echo "### pick the best validation checkpoint, then touch test once"
BEST=$("${VENV}/bin/python" - <<PY
import glob, json
best = None
for path in glob.glob("${RESULTS}/E49_val_iter_*.json"):
    d = json.load(open(path))
    s = d["summary"].get("symmetric_mm_no_endcaps")
    if not s:
        continue
    if best is None or s["median"] < best[0]:
        best = (s["median"], path.split("E49_val_")[1].replace(".json", ""))
print(best[1] if best else "")
PY
)
if [ -z "${BEST}" ]; then echo "no validation checkpoint produced a score"; exit 1; fi
echo "selected: ${BEST}"
echo "--- test at ${BEST}"
score_root "${HDD_DATA}/tetheart_914_3d_test" "test" "${WORK}/${BEST}.pth"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
