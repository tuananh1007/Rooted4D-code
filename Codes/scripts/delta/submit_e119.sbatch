#!/bin/bash
# E119 (Docs/E119_LATE_FUSION_FIELD_PREREGISTRATION_2026-11-13.md).
# One job per stage; R4D_KEYS (space-separated set keys) splits the runs over jobs:
#   sbatch --export=ALL,R4D_STAGE=smoke submit_e119.sbatch
#     two validation samples of v4n96o and a few settings, in a separate output
#     root with its own reports; the analysis never reads it.
#   sbatch --export=ALL,R4D_STAGE=val,R4D_KEYS="v4n96o v3n96o ..." submit_e119.sbatch
#     every setting on those sets' validation splits.
#   sbatch --export=ALL,R4D_STAGE=val-analyze submit_e119.sbatch
#     the code checks against the chains' unimodal validation rows, selection.
#   sbatch --export=ALL,R4D_STAGE=test,R4D_KEYS="..." submit_e119.sbatch
#     the selected settings and Rooted4D's, once (refused unless the validation
#     checks passed).
#   sbatch --export=ALL,R4D_STAGE=test-analyze submit_e119.sbatch
#     the checks against the chains' late-fusion test rows, the paired
#     statistics, the predictions and the table rows.
#SBATCH --job-name=rooted4d-e119
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128g
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
: "${R4D_STAGE:?set R4D_STAGE to smoke, val, val-analyze, test or test-analyze}"
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
if [ "${R4D_STAGE}" != smoke ] && grep -q '^dirty=yes' PROVENANCE.txt; then echo "refusing: dirty tree"; exit 1; fi

export R4D_RESULTS="${NVME}/results"
export R4D_E119_OUT="${NVME}/results/e119"
for key in v4n96o v3n96o v4n48o v3o v3n48o v4n96os1 v4n96os2 v4n96os3 v4n96os4 v4n96os5 \
           v3n96os1 v3n96os2 v3n96os3 v3n96os4 v3n96os5; do
  [ -e "${R4D_RESULTS}/e100/${key}/TEST_DONE" ] || { echo "missing chain: ${key}"; exit 1; }
done
[ -e "${R4D_RESULTS}/e94/v4oriented/TEST_DONE" ] || { echo "missing chain: v4oriented"; exit 1; }
for f in "${R4D_RESULTS}/e116/test/index.json" "${R4D_RESULTS}/e117/test/index.json" \
         results/benchmark/E116/E116_val.json results/benchmark/E117/E117_sets_val.json; do
  [ -e "$f" ] || { echo "missing required input: $f"; exit 1; }
done
E="Codes/scripts/experiments/e119_late_fusion_field.py"

echo "### unit tests"
(cd Codes && python -m unittest tests.test_late_fusion_field tests.test_residual_field -q 2>&1 | tail -2)

case "${R4D_STAGE}" in
  smoke)        export R4D_E119_OUT="${NVME}/results/e119_smoke" R4D_E119_REPORTS="${NVME}/results/e119_smoke/reports"
                export R4D_SMOKE_SHARD=0/30
                export R4D_CONFIGS='[["session",0,0,10],["shared",0,0,10],["session",4,0.5,5],["shared",4,0.5,5],["session",64,64,80]]'
                mkdir -p "${R4D_E119_OUT}" "${R4D_E119_REPORTS}"
                python -u "$E" run val v4n96o
                python -u "$E" analyze val || true ;;
  val)          : "${R4D_KEYS:?set R4D_KEYS}"
                mkdir -p "${R4D_E119_OUT}"
                cp -f PROVENANCE.txt "${R4D_E119_OUT}/PROVENANCE_${R4D_STAGE}_${SLURM_JOB_ID}.txt"
                python -u "$E" run val ${R4D_KEYS} ;;
  val-analyze)  mkdir -p results/benchmark/E119
                python -u "$E" analyze val ;;
  test)         : "${R4D_KEYS:?set R4D_KEYS}"
                cp -f PROVENANCE.txt "${R4D_E119_OUT}/PROVENANCE_${R4D_STAGE}_${SLURM_JOB_ID}.txt"
                python -u "$E" run test ${R4D_KEYS} ;;
  test-analyze) python -u "$E" analyze test
                echo "### latex"; python -u "$E" latex ;;
  *)            echo "unknown stage ${R4D_STAGE}"; exit 1 ;;
esac
if [ "${R4D_STAGE}" != smoke ]; then
  mkdir -p results/benchmark/E119
  cp -f PROVENANCE.txt "results/benchmark/E119/PROVENANCE_${R4D_STAGE}_${SLURM_JOB_ID}.txt"
fi
echo "finished ${R4D_STAGE} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
