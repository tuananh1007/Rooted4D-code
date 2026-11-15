#!/bin/bash
# E117, part B (Docs/E117_FIELD_REPLICATION_AND_MITEA_PREREGISTRATION_2026-10-24.md).
# One job per stage, on the interactive partition:
#   sbatch --export=ALL,R4D_STAGE=smoke submit_e117_mitea.sbatch
#     code check on two training subjects, labels standing in for the echo
#     predictions, a few settings; no validation or test subject is read.
#   sbatch --export=ALL,R4D_STAGE=val   submit_e117_mitea.sbatch
#     both conventions' geometry (training and validation subjects), every
#     setting on the validation subjects, the check against E99's rows,
#     selection.
#   sbatch --export=ALL,R4D_STAGE=test  submit_e117_mitea.sbatch
#     the test subjects' geometry and the frozen and selected settings, once
#     (refused unless the validation checks passed), with the check against
#     E99's and E103's rows.
#SBATCH --job-name=rooted4d-e117-mitea
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64g
#SBATCH --time=00:59:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
: "${R4D_STAGE:?set R4D_STAGE to smoke, val or test}"
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
if [ "${R4D_STAGE}" != smoke ] && grep -q '^dirty=yes' PROVENANCE.txt; then echo "refusing: dirty tree"; exit 1; fi
for f in data/derived/mitea-e99/geometry_train.npz data/derived/mitea-e99/geometry_val.npz \
         data/derived/mitea-e99/geometry_test.npz results/benchmark/E99/E99_fits_val.json \
         results/benchmark/E99/E99_fits_test.json results/benchmark/E103/E103_rows.json; do
  [ -e "$f" ] || { echo "missing required input: $f"; exit 1; }
done
mkdir -p results/benchmark/E117
E="Codes/scripts/experiments/e117_mitea_field.py"

echo "### unit tests"
(cd Codes && python -m unittest tests.test_residual_field tests.test_e117 -q 2>&1 | tail -2)

case "${R4D_STAGE}" in
  smoke) R4D_MODE=geometry R4D_SPLIT=train python -u "$E"
         R4D_MODE=fit R4D_SPLIT=train R4D_LABELS_AS_PREDICTIONS=1 R4D_LIMIT=2 \
           R4D_CONFIGS='[["shared",0,0,10],["shared",4,0.5,5],["session",4,0.5,5],["shared",64,64,80]]' \
           python -u "$E" ;;
  val)   R4D_MODE=geometry R4D_SPLIT=train python -u "$E"
         R4D_MODE=geometry R4D_SPLIT=val python -u "$E"
         R4D_MODE=fit R4D_SPLIT=val python -u "$E"
         R4D_MODE=analyze R4D_SPLIT=val python -u "$E" ;;
  test)  export R4D_ALLOW_TEST=1
         R4D_MODE=geometry R4D_SPLIT=test python -u "$E"
         R4D_MODE=fit R4D_SPLIT=test python -u "$E"
         R4D_MODE=analyze R4D_SPLIT=test python -u "$E"
         echo "### latex"; python -u "$E" latex ;;
  *)     echo "unknown stage ${R4D_STAGE}"; exit 1 ;;
esac
cp -f PROVENANCE.txt "results/benchmark/E117/PROVENANCE_mitea_${R4D_STAGE}_${SLURM_JOB_ID}.txt"
echo "finished ${R4D_STAGE} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
