#!/bin/bash
# E110 to E113 (Docs/E110_E113_MARGIN_AND_COHORT_PREREGISTRATION_2026-10-21.md).
# One job per stage, on the interactive partition:
#   sbatch --export=ALL,R4D_STAGE=val  submit_e11x.sbatch
#     E113 on validation (seven thinning levels, two arms), its level-0 check
#     against E100's session-weight rows, E111's validation code check on
#     the unthinned rows, then E110 and E112, which read stored test rows only.
#   sbatch --export=ALL,R4D_STAGE=test submit_e11x.sbatch
#     E113's addendum level (cmr3) on validation first, then E113 on test,
#     once (refused unless the validation report passed), and E111 on E100's
#     stored test fits.
#SBATCH --job-name=rooted4d-e11x
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

export R4D_E100_ROWS="${NVME}/results/e100/v4n96o/test/bd364644c4605d9d"
export R4D_E113_OUT="${NVME}/results/e113"
export R4D_E113_VAL_REFERENCE="${NVME}/results/e100/v4n96o/session/ba112fcc6265af99"
for f in "${R4D_E100_ROWS}/spec.json" "${R4D_E113_VAL_REFERENCE}/spec.json" "${NVME}/results/e100/v4n96o/TEST_DONE"; do
  [ -e "$f" ] || { echo "missing required input: $f"; exit 1; }
done
mkdir -p "${R4D_E113_OUT}"; cp -f PROVENANCE.txt "${R4D_E113_OUT}/PROVENANCE_${R4D_STAGE}_${SLURM_JOB_ID}.txt"
E="Codes/scripts/experiments"

echo "### unit tests of the thinning"
python -m unittest tests.test_thinning -q 2>&1 | tail -2

if [ "${R4D_STAGE}" = val ]; then
  echo "### E113 validation: run, check, analyze"
  python -u "${E}/e113_sparsity.py" run val
  python -u "${E}/e113_sparsity.py" analyze val
  echo "### E111 validation code check (E113's unthinned validation rows)"
  FULL="${R4D_E113_OUT}/val/$(python -c "import json;print(json.load(open('${R4D_E113_OUT}/val/index.json'))['full'])")"
  python -u "${E}/e111_relational_cycle.py" measure val "${FULL}"
  echo "### E110: non-inferiority from the stored test rows"
  python -u "${E}/e110_noninferiority.py" "${NVME}/results"
  echo "### E112: donor mismatch (exploratory), stored test rows"
  python -u "${E}/e112_donor_mismatch.py" "${NVME}/results"
elif [ "${R4D_STAGE}" = test ]; then
  echo "### E113 addendum level on validation (registered levels are complete and not re-run), then analyze"
  python -u "${E}/e113_sparsity.py" run val
  python -u "${E}/e113_sparsity.py" analyze val
  echo "### E113 test: run once, check, analyze"
  python -u "${E}/e113_sparsity.py" run test
  python -u "${E}/e113_sparsity.py" analyze test
  echo "### E111 on E100's stored test fits"
  python -u "${E}/e111_relational_cycle.py" measure test "${R4D_E100_ROWS}"
else
  echo "unknown stage ${R4D_STAGE}"; exit 1
fi
for d in E110 E111 E112 E113; do
  [ -d "results/benchmark/${d}" ] && cp -f PROVENANCE.txt "results/benchmark/${d}/PROVENANCE_${R4D_STAGE}_${SLURM_JOB_ID}.txt"
done
echo "finished ${R4D_STAGE} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
