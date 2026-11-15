#!/bin/bash
# E88 (Docs/E88_REPLICATION_PREREGISTRATION_2026-09-28.md): one replication
# instance through selection and its single test evaluation, unattended.
#   sbatch --export=ALL,R4D_BENCH=v4,R4D_SPLIT=1 submit_e88_chain.sbatch
#SBATCH --job-name=rooted4d-e88-chain
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128g
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
cd "${NVME_ROOT}/code"; mkdir -p logs
: "${R4D_BENCH:?set R4D_BENCH to v4 or v3}"
: "${R4D_SPLIT:?set R4D_SPLIT to 1..5}"
if [ -e data ] && [ ! -L data ]; then echo "data is not a symlink, refusing"; exit 1; fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then ln -sfT "${HDD_DATA}" data; fi
source .venv/bin/activate
export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export R4D_WORKERS="${R4D_WORKERS:-60}"
export R4D_E88_OUT="${NVME_ROOT}/results/e88"
mkdir -p "${R4D_E88_OUT}"
grep -E '^(commit|dirty)=' PROVENANCE.txt
if grep -q '^dirty=yes' PROVENANCE.txt; then echo "refusing: dirty tree"; exit 1; fi
echo "### full suite first"
python -m unittest discover -s Codes/tests -q > "logs/e88_tests_${SLURM_JOB_ID}.log" 2>&1 || {
  echo "tests failed"; tail -20 "logs/e88_tests_${SLURM_JOB_ID}.log"; exit 1; }
tail -2 "logs/e88_tests_${SLURM_JOB_ID}.log"
echo "### chain ${R4D_BENCH} split ${R4D_SPLIT} with ${R4D_WORKERS} workers"
python -u Codes/scripts/experiments/e88_replication.py chain "${R4D_BENCH}" "${R4D_SPLIT}"
cp -f PROVENANCE.txt "${R4D_E88_OUT}/${R4D_BENCH}s${R4D_SPLIT}/PROVENANCE_${SLURM_JOB_ID}.txt"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
