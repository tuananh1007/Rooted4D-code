#!/bin/bash
# Run one experiment script on a CPU node, after the full test suite. Usage:
#   sbatch --export=ALL,R4D_SCRIPT=Codes/scripts/experiments/<script>.py[,R4D_ALLOW_TEST=1,...] submit_script.sbatch
#SBATCH --job-name=rooted4d-script
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
cd "${NVME_ROOT}/code"; mkdir -p logs
: "${R4D_SCRIPT:?set R4D_SCRIPT to a script path relative to the code root}"
if [ -e data ] && [ ! -L data ]; then echo "data is not a symlink, refusing"; exit 1; fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then ln -sfT "${HDD_DATA}" data; fi
source .venv/bin/activate
export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export R4D_WORKERS="${R4D_WORKERS:-${SLURM_CPUS_PER_TASK}}"
export R4D_OUT="${R4D_OUT:-${NVME_ROOT}/results/uniformity}"
mkdir -p "${R4D_OUT}"
[ -f PROVENANCE.txt ] && grep -E '^(commit|dirty)=' PROVENANCE.txt
if grep -q '^dirty=yes' PROVENANCE.txt 2>/dev/null; then
  echo "refusing: synced tree was dirty, results would match no commit"; exit 1; fi
if [ "${R4D_SMOKE:-0}" = "1" ]; then echo "refusing: smoke mode is for local code-path checks only"; exit 1; fi
echo "### full suite first"
python -m unittest discover -s Codes/tests -q > "logs/script_tests_${SLURM_JOB_ID}.log" 2>&1 || {
  echo "tests failed"; tail -20 "logs/script_tests_${SLURM_JOB_ID}.log"; exit 1; }
tail -2 "logs/script_tests_${SLURM_JOB_ID}.log"
echo "### ${R4D_SCRIPT} with ${R4D_WORKERS} workers"
python -u "${R4D_SCRIPT}"
cp -f PROVENANCE.txt "${R4D_OUT}/PROVENANCE_${SLURM_JOB_ID}.txt"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
