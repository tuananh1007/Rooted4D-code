#!/bin/bash
# A named protocol chain (E93, E94) with E88's machinery: selection on
# validation, the session weight, and one test evaluation, unattended.
#   sbatch --export=ALL,R4D_CHAIN_SCRIPT=Codes/scripts/experiments/e94_oriented.py,R4D_CHAIN_OUT=results/e94 submit_chain.sbatch
#SBATCH --job-name=rooted4d-chain
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
: "${R4D_CHAIN_SCRIPT:?set R4D_CHAIN_SCRIPT to the experiment script with a chain command}"
: "${R4D_CHAIN_OUT:?set R4D_CHAIN_OUT to a results directory under the NVMe root}"
if [ -e data ] && [ ! -L data ]; then echo "data is not a symlink, refusing"; exit 1; fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then ln -sfT "${HDD_DATA}" data; fi
source .venv/bin/activate
export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export R4D_WORKERS="${R4D_WORKERS:-60}"
export R4D_E88_OUT="${NVME_ROOT}/${R4D_CHAIN_OUT}"
mkdir -p "${R4D_E88_OUT}"
grep -E '^(commit|dirty)=' PROVENANCE.txt
if grep -q '^dirty=yes' PROVENANCE.txt; then echo "refusing: dirty tree"; exit 1; fi
echo "### full suite first"
python -m unittest discover -s Codes/tests -q > "logs/chain_tests_${SLURM_JOB_ID}.log" 2>&1 || {
  echo "tests failed"; tail -20 "logs/chain_tests_${SLURM_JOB_ID}.log"; exit 1; }
tail -2 "logs/chain_tests_${SLURM_JOB_ID}.log"
echo "### ${R4D_CHAIN_SCRIPT} chain with ${R4D_WORKERS} workers"
python -u "${R4D_CHAIN_SCRIPT}" chain
cp -f PROVENANCE.txt "${R4D_E88_OUT}/PROVENANCE_${SLURM_JOB_ID}.txt"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
