#!/bin/bash
# E109: qualitative samples at 96 ring nodes, and the published methods on the
# same hearts (e109_qualitative_96.py). This job runs every CPU part; GHDHeart,
# which needs a GPU, is submit_e109_ghdheart.sbatch. Nothing here is scored for
# the paper; each part checks what it draws against the stored rows.
# R4D_PARTS picks the parts (default: all five). A run that fits in an hour, for
# the interactive partition:
#   sbatch -p cpu-interactive -t 00:59:00 --export=ALL,R4D_PARTS="fusion modes chamber tetheart" ...
#SBATCH --job-name=rooted4d-e109
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
NVME="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
cd "${NVME}/code"; mkdir -p logs
if [ -e data ] && [ ! -L data ]; then echo "data is not a symlink, refusing"; exit 1; fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then ln -sfT "${HDD_DATA}" data; fi
grep -E '^(commit|dirty)=' PROVENANCE.txt
if grep -q '^dirty=yes' PROVENANCE.txt; then echo "refusing: dirty tree"; exit 1; fi

RESULTS="${NVME}/results/${SLURM_JOB_ID}_qual96"; mkdir -p "${RESULTS}"
cp -f PROVENANCE.txt "${RESULTS}/"
export PYTHONPATH="${NVME}/code/Codes:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" R4D_THREADS=2
export R4D_ALLOW_TEST=1 R4D_OUT="${RESULTS}"
export R4D_E100_ROWS="${NVME}/results/e100/v4n96o/test/bd364644c4605d9d"
COMPONENTS="${NVME}/results/e100/v3n96o/components"
export R4D_E69="${COMPONENTS}/E69_v3n96o_test" R4D_E70_ROWS="${COMPONENTS}/E70_v3n96o_test" R4D_E74_ROWS="${COMPONENTS}/E74_v3n96o_test"
export R4D_ACDC="${HDD_DATA}/raw/acdc"
export R4D_SEGMENTER="${NVME}/results/21900728/E31_segmenter.pt"
export R4D_NNUNET_ACDC="${HDD_DATA}/nnunet/predictions_acdc_test"
export R4D_PRED_DIR="${NVME}/results/22102244_thfinal/preds_test_iter_60000"
export R4D_TET_FRAME_REPORT="${NVME}/results/22045910_tetframe/E50_tet_frame.json"
for f in "${R4D_E100_ROWS}/spec.json" "${R4D_E69}" "${R4D_E70_ROWS}" "${R4D_E74_ROWS}" "${R4D_SEGMENTER}" \
         "${R4D_NNUNET_ACDC}" "${R4D_PRED_DIR}" "${R4D_TET_FRAME_REPORT}"; do
  [ -e "$f" ] || { echo "missing required input: $f"; exit 1; }
done
SCRIPT=Codes/scripts/experiments/e109_qualitative_96.py
PROJECT="${NVME}/code/.venv/bin/python"
TETHEART="${NVME}/external/.venv-tetheart/bin/python"

echo "### picks"
R4D_PART=picks "${PROJECT}" -u "${SCRIPT}" > "${RESULTS}/picks.log" 2>&1 || { tail -30 "${RESULTS}/picks.log"; exit 1; }
cat "${RESULTS}/picks.log"

PARTS="${R4D_PARTS:-fusion modes chamber tetheart image}"
echo "### ${PARTS}, concurrently"
declare -A PIDS
for part in ${PARTS}; do
  if [ "${part}" = tetheart ]; then
    R4D_PART=tetheart R4D_WORKERS=4 "${TETHEART}" -u "${SCRIPT}" > "${RESULTS}/tetheart.log" 2>&1 &
  else
    R4D_PART="${part}" "${PROJECT}" -u "${SCRIPT}" > "${RESULTS}/${part}.log" 2>&1 &
  fi
  PIDS[${part}]=$!
done
FAILED=0
for part in "${!PIDS[@]}"; do
  if wait "${PIDS[$part]}"; then echo "${part}: done"; else echo "${part}: FAILED"; tail -20 "${RESULTS}/${part}.log"; FAILED=1; fi
done
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "${FAILED}"
