#!/bin/bash
# E116 (Docs/E116_RESIDUAL_FIELD_PREREGISTRATION_2026-10-23.md).
# One job per stage, on the interactive partition:
#   sbatch --export=ALL,R4D_STAGE=val  submit_e116.sbatch
#     both primaries at full evidence with every setting, the check of the
#     arm's own Rooted4D fit against E100's validation rows, selection.
#   sbatch --export=ALL,R4D_STAGE=test submit_e116.sbatch
#     the selected settings, once (refused unless the validation checks
#     passed): both primaries at full evidence and the realistic-session
#     primary at four thinned levels, with the check against E100's and
#     E113's Rooted4D test rows; comparators from E100, E113 and E115.
#SBATCH --job-name=rooted4d-e116
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96g
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
export R4D_WORKERS="${R4D_WORKERS:-30}"
grep -E '^(commit|dirty)=' PROVENANCE.txt
if grep -q '^dirty=yes' PROVENANCE.txt; then echo "refusing: dirty tree"; exit 1; fi

export R4D_RESULTS="${NVME}/results"
export R4D_E116_OUT="${NVME}/results/e116"
for f in "${R4D_RESULTS}/e100/v4n96o/TEST_DONE" "${R4D_RESULTS}/e100/v3n96o/TEST_DONE" \
         "${R4D_RESULTS}/e113/test/index.json" "${R4D_RESULTS}/e115/test/index.json"; do
  [ -e "$f" ] || { echo "missing required input: $f"; exit 1; }
done
mkdir -p "${R4D_E116_OUT}" results/benchmark/E116
cp -f PROVENANCE.txt "${R4D_E116_OUT}/PROVENANCE_${R4D_STAGE}_${SLURM_JOB_ID}.txt"
E="Codes/scripts/experiments"

echo "### unit tests of the residual field"
(cd Codes && python -m unittest tests.test_residual_field -q 2>&1 | tail -2)

case "${R4D_STAGE}" in
  val)  python -u "${E}/e116_residual_field.py" run val
        python -u "${E}/e116_residual_field.py" analyze val ;;
  test) python -u "${E}/e116_residual_field.py" run test
        python -u "${E}/e116_residual_field.py" analyze test
        for w in main paired alignment thinned; do
          echo "### latex ${w}"; python -u "${E}/e116_residual_field.py" latex test "${w}"; done ;;
  *)    echo "unknown stage ${R4D_STAGE}"; exit 1 ;;
esac
cp -f PROVENANCE.txt "results/benchmark/E116/PROVENANCE_${R4D_STAGE}_${SLURM_JOB_ID}.txt"
echo "finished ${R4D_STAGE} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
