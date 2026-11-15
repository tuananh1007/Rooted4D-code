#!/bin/bash
# Generate TetHeart test-split predictions for one checkpoint.
#
# Only preds_test_iter_60000 exists, because the original run selected iter_60000
# under a metric whose class check was later found invalid. If the corrected
# selection prefers a different checkpoint, the test meshes for it do not exist
# and have to be produced before anything can be scored. All ten checkpoints
# (10k..100k) are on disk, so this is a test pass, not a retrain.
#
# Usage:  R4D_CKPT=iter_80000 sbatch submit_tetheart_test_preds.sbatch
#SBATCH --job-name=rooted4d-thtestpreds
#SBATCH --account=bhio-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

CKPT="${R4D_CKPT:?set R4D_CKPT, e.g. iter_80000}"
NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
T="${NVME_ROOT}/external/TetHeart"
VENV="${NVME_ROOT}/external/.venv-tetheart"
WORK="${NVME_ROOT}/tetheart_work/rooted4d_acdc_static"
RES="${NVME_ROOT}/results/22102244_thfinal"
cd "${NVME_ROOT}/code"; mkdir -p logs

CKPT_PATH="${WORK}/${CKPT}.pth"
[ -f "${CKPT_PATH}" ] || { echo "no such checkpoint: ${CKPT_PATH}"; exit 1; }

PREDS="${RES}/preds_test_${CKPT}"
if [ -d "${PREDS}" ] && [ "$(ls "${PREDS}"/*_pred.obj 2>/dev/null | wc -l)" -gt 0 ]; then
  echo "refusing to overwrite existing predictions in ${PREDS}"; exit 1
fi
mkdir -p "${PREDS}"

# Never rewrite a shared symlink another job may be using: create it only when
# it is absent or points somewhere else.
if [ -e data ] && [ ! -L data ]; then echo "data is not a symlink, refusing"; exit 1; fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then ln -sfT "${HDD_DATA}" data; fi

# MRIDataset's test_mode takes 1/nfolds of a root, so two folds at nfolds=2
# cover the test root exactly once.
CFG="${T}/configs/Unified/tet/static/rooted4d_eval_test_${CKPT}.py"
"${VENV}/bin/python" - "${HDD_DATA}/tetheart_914_3d_test" "${CFG}" <<'PY'
import re, sys, pathlib
root, dest = sys.argv[1], sys.argv[2]
src = pathlib.Path("/work/nvme/bhio/avu2/Rooted4D/external/TetHeart/configs/Unified/tet/static/rooted4d_acdc_static.py")
text = re.sub(r"^data_root\s*=.*$", "data_root = %r" % root, src.read_text(), flags=re.M)
pathlib.Path(dest).write_text(text.replace("nfolds=5", "nfolds=2"))
PY

for fold in 0 1; do
  echo "--- fold ${fold} $(date -u +%H:%M:%S)"
  (cd "$T" && "${VENV}/bin/python" tools/test.py "${CFG}" "${CKPT_PATH}" \
     --show --show-dir "${PREDS}" --cfg-options "fold=${fold}" \
     > "${RES}/testpreds_${CKPT}_f${fold}.log" 2>&1) \
    || echo "    test.py failed: ${CKPT} fold ${fold}"
done

n=$(ls "${PREDS}"/*_pred.obj 2>/dev/null | wc -l)
nonempty=$(find "${PREDS}" -name '*_pred.obj' -size +0 2>/dev/null | wc -l)
echo "${n} meshes written, ${nonempty} non-empty (expect 132 for 44 groups x 3 classes)"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
