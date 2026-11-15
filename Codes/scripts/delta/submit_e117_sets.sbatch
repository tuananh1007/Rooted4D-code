#!/bin/bash
# E117, part A (Docs/E117_FIELD_REPLICATION_AND_MITEA_PREREGISTRATION_2026-10-24.md).
# One job per stage:
#   sbatch --export=ALL,R4D_STAGE=val  submit_e117_sets.sbatch
#     the 14 sets' validation splits with every setting, the check of the
#     arm's own joint fit against each chain's validation rows, selection.
#   sbatch --export=ALL,R4D_STAGE=test submit_e117_sets.sbatch
#     the selected and the frozen settings, once (refused unless the
#     validation checks passed), with the check against each chain's test
#     rows; comparators from the chains' test rows.
#SBATCH --job-name=rooted4d-e117-sets
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128g
#SBATCH --time=04:00:00
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
export R4D_E117_OUT="${NVME}/results/e117"
for key in v4n48o v3o v3n48o v4n96os1 v4n96os2 v4n96os3 v4n96os4 v4n96os5 \
           v3n96os1 v3n96os2 v3n96os3 v3n96os4 v3n96os5; do
  [ -e "${R4D_RESULTS}/e100/${key}/TEST_DONE" ] || { echo "missing chain: ${key}"; exit 1; }
done
[ -e "${R4D_RESULTS}/e94/v4oriented/TEST_DONE" ] || { echo "missing chain: v4oriented"; exit 1; }
mkdir -p "${R4D_E117_OUT}" results/benchmark/E117
cp -f PROVENANCE.txt "${R4D_E117_OUT}/PROVENANCE_sets_${R4D_STAGE}_${SLURM_JOB_ID}.txt"
E="Codes/scripts/experiments"

echo "### unit tests"
(cd Codes && python -m unittest tests.test_residual_field tests.test_e117 -q 2>&1 | tail -2)

case "${R4D_STAGE}" in
  val)  python -u "${E}/e117_field_replication.py" run val
        python -u "${E}/e117_field_replication.py" analyze val ;;
  test) python -u "${E}/e117_field_replication.py" run test
        python -u "${E}/e117_field_replication.py" analyze test
        echo "### latex"; python -u "${E}/e117_field_replication.py" latex ;;
  *)    echo "unknown stage ${R4D_STAGE}"; exit 1 ;;
esac
cp -f PROVENANCE.txt "results/benchmark/E117/PROVENANCE_sets_${R4D_STAGE}_${SLURM_JOB_ID}.txt"
echo "finished ${R4D_STAGE} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
