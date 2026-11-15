#!/bin/bash
#SBATCH --job-name=rooted4d-theval
#SBATCH --account=bhio-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
T="${NVME_ROOT}/external/TetHeart"
VENV="${NVME_ROOT}/external/.venv-tetheart"
WORK="${NVME_ROOT}/tetheart_work/rooted4d_acdc_static"
cd "${NVME_ROOT}/code"; mkdir -p logs
RESULTS="${NVME_ROOT}/results/${SLURM_JOB_ID}_theval"; mkdir -p "${RESULTS}"
export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export R4D_ACDC="${HDD_DATA}/raw/acdc"
export R4D_OUT="${RESULTS}"

# MRIDataset's test_mode takes only 1/nfolds of the folders in a root, so two
# folds at nfolds=2 cover a root exactly once.
run_split () {
  local root="$1" tag="$2" ckpt="$3"
  local cfg="${T}/configs/Unified/tet/static/rooted4d_eval_${tag}.py"
  "${VENV}/bin/python" - "$root" "$cfg" <<'PY'
import re, sys, pathlib
root, dest = sys.argv[1], sys.argv[2]
src = pathlib.Path("/work/nvme/bhio/avu2/Rooted4D/external/TetHeart/configs/Unified/tet/static/rooted4d_acdc_static.py")
text = src.read_text()
text = re.sub(r"^data_root\s*=.*$", "data_root = %r" % root, text, flags=re.M)
text = text.replace("nfolds=5", "nfolds=2")
pathlib.Path(dest).write_text(text)
print("wrote", dest, "->", root)
PY
  local preds="${RESULTS}/preds_${tag}_$(basename ${ckpt} .pth)"
  mkdir -p "${preds}"
  for fold in 0 1; do
    (cd "$T" && "${VENV}/bin/python" tools/test.py "${cfg}" "${ckpt}" \
      --show --show-dir "${preds}" \
      --cfg-options "fold=${fold}" \
      > "${RESULTS}/test_${tag}_$(basename ${ckpt} .pth)_f${fold}.log" 2>&1) || \
      echo "  test.py failed for ${tag} fold ${fold}"
  done
  echo "  ${tag} $(basename ${ckpt}): $(ls ${preds}/*_pred.obj 2>/dev/null | wc -l) predicted meshes"
  R4D_PRED_DIR="${preds}" R4D_OUTNAME="E49_${tag}_$(basename ${ckpt} .pth).json" \
    "${VENV}/bin/python" Codes/scripts/experiments/e49_tetheart_eval.py \
    > "${RESULTS}/e49_${tag}_$(basename ${ckpt} .pth).log" 2>&1 || \
    echo "  e49 failed for ${tag} $(basename ${ckpt})"
  tail -14 "${RESULTS}/e49_${tag}_$(basename ${ckpt} .pth).log"
}

echo "### checkpoint selection on our validation patients"
for iters in 40000 60000 80000 100000; do
  ck="${WORK}/iter_${iters}.pth"
  [ -f "$ck" ] || { echo "  missing ${ck}"; continue; }
  echo "--- iter_${iters}"
  run_split "${HDD_DATA}/tetheart_914_3d_val" "val" "$ck"
done
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
