#!/bin/bash
# E118: E109's qualitative fusion samples for the methods added after it
# (e118_qualitative_recent.py). Nothing here is scored for the paper; every
# drawn fit is checked against its stored row (E100, E115, E116).
#   sbatch submit_e118_qualitative.sbatch
#SBATCH --job-name=rooted4d-e118
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:59:00
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

RESULTS="${NVME}/results/${SLURM_JOB_ID}_qual_recent"; mkdir -p "${RESULTS}"
cp -f PROVENANCE.txt "${RESULTS}/"
export PYTHONPATH="${NVME}/code/Codes:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export R4D_ALLOW_TEST=1 R4D_RESULTS="${NVME}/results" R4D_OUT="${RESULTS}" R4D_WORKERS=4
for f in results/benchmark/E115/E115_val.json Codes/configs/rooted4d_main.json \
         "${NVME}/results/e100/v4n96o/TEST_DONE" "${NVME}/results/e115/test/index.json" \
         "${NVME}/results/e116/test/index.json"; do
  [ -e "$f" ] || { echo "missing required input: $f"; exit 1; }
done
"${NVME}/code/.venv/bin/python" -u Codes/scripts/experiments/e118_qualitative_recent.py \
  > "${RESULTS}/e118.log" 2>&1 || { tail -40 "${RESULTS}/e118.log"; exit 1; }
tail -3 "${RESULTS}/e118.log"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
