#!/bin/bash
# E115 (Docs/E115_RECENT_FUSION_COMPARATORS_PREREGISTRATION_2026-10-22.md).
# One job per stage, on the interactive partition:
#   sbatch --export=ALL,R4D_STAGE=val  submit_e115.sbatch
#     the BCPD port against the authors' binary, then both primaries at full
#     evidence with every configuration, the late-fusion code check, selection.
#   sbatch --export=ALL,R4D_STAGE=test submit_e115.sbatch
#     the selected configurations, once (refused unless the validation checks
#     passed): both primaries at full evidence and the realistic-session
#     primary at four thinned levels, with the code check against E100's and
#     E113's late-fusion test rows.
#SBATCH --job-name=rooted4d-e115
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200g
#SBATCH --time=00:59:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
: "${R4D_STAGE:?set R4D_STAGE to val or test}"
NVME="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
cd "${NVME}/code"; mkdir -p logs
if [ -e data ] && [ ! -L data ]; then echo "data is not a symlink, refusing"; exit 1; fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then ln -sfT "${HDD_DATA}" data; fi
source .venv/bin/activate
export PYTHONPATH="${NVME}/code/Codes:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export R4D_WORKERS="${R4D_WORKERS:-62}"
grep -E '^(commit|dirty)=' PROVENANCE.txt
if grep -q '^dirty=yes' PROVENANCE.txt; then echo "refusing: dirty tree"; exit 1; fi

export R4D_RESULTS="${NVME}/results"
export R4D_E115_OUT="${NVME}/results/e115"
export R4D_BCPD_BIN="${NVME}/external/bcpd/bcpd"
for f in "${R4D_RESULTS}/e100/v4n96o/TEST_DONE" "${R4D_RESULTS}/e100/v3n96o/TEST_DONE" \
         "${R4D_RESULTS}/e113/test/index.json" "${R4D_BCPD_BIN}"; do
  [ -e "$f" ] || { echo "missing required input: $f"; exit 1; }
done
mkdir -p "${R4D_E115_OUT}" results/benchmark/E115
cp -f PROVENANCE.txt "${R4D_E115_OUT}/PROVENANCE_${R4D_STAGE}_${SLURM_JOB_ID}.txt"
E="Codes/scripts/experiments"

echo "### unit tests of the comparators"
(cd Codes && python -m unittest tests.test_recent_fusion tests.test_published_fusion -q 2>&1 | tail -2)

case "${R4D_STAGE}" in
  val)  echo "### BCPD port against the authors' binary"
        python -u "${E}/e115_bcpd_reference.py" results/benchmark/E115/E115_bcpd_reference.json | tail -1
        python -u "${E}/e115_recent_fusion.py" run val
        python -u "${E}/e115_recent_fusion.py" analyze val ;;
  test) python -u "${E}/e115_recent_fusion.py" run test
        python -u "${E}/e115_recent_fusion.py" analyze test
        for w in main supp paired alignment thinned attachment; do
          echo "### latex ${w}"; python -u "${E}/e115_recent_fusion.py" latex test "${w}"; done ;;
  *)    echo "unknown stage ${R4D_STAGE}"; exit 1 ;;
esac
cp -f PROVENANCE.txt "results/benchmark/E115/PROVENANCE_${R4D_STAGE}_${SLURM_JOB_ID}.txt"
echo "finished ${R4D_STAGE} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
