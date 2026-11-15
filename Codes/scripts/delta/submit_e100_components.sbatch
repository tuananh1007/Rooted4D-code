#!/bin/bash
# E100 (Docs/E100_ORIENTED_REBUILD_PREREGISTRATION_2026-09-29.md): component
# studies on one rebuilt instance, each with its own code (e100_components.py).
# Resumable; a finished study is skipped. Needs the instance's chain test spec.
#   sbatch --export=ALL,R4D_KEY=v3n96o,R4D_STUDIES="gate e65 e66" submit_e100_components.sbatch
#   sbatch --export=ALL,R4D_KEY=v4n96o,R4D_COMMAND=e95 submit_e100_components.sbatch   (another command)
#SBATCH --job-name=rooted4d-e100-comp
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64g
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
: "${R4D_KEY:?set R4D_KEY}"
if [ -z "${R4D_COMMAND:-}" ]; then : "${R4D_STUDIES:?set R4D_STUDIES to a space-separated list}"; fi
NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
cd "${NVME_ROOT}/code"; mkdir -p logs
if [ -e data ] && [ ! -L data ]; then echo "data is not a symlink, refusing"; exit 1; fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then ln -sfT "${HDD_DATA}" data; fi
source .venv/bin/activate
export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export R4D_WORKERS="${R4D_WORKERS:-${SLURM_CPUS_PER_TASK}}"
# E94's instance keeps its own chain root; every other instance is E100's.
if [ "${R4D_KEY}" = v4oriented ]; then CHAIN=e94; else CHAIN=e100; fi
export R4D_E88_OUT="${NVME_ROOT}/results/${CHAIN}"
grep -E '^(commit|dirty)=' PROVENANCE.txt
if grep -q '^dirty=yes' PROVENANCE.txt; then echo "refusing: dirty tree"; exit 1; fi
if [ "${R4D_SMOKE:-0}" = "1" ]; then echo "refusing: smoke mode is for local code-path checks only"; exit 1; fi
echo "### full suite first"
python -m unittest discover -s Codes/tests -q > "logs/e100comp_tests_${SLURM_JOB_ID}.log" 2>&1 || {
  echo "tests failed"; tail -20 "logs/e100comp_tests_${SLURM_JOB_ID}.log"; exit 1; }
tail -2 "logs/e100comp_tests_${SLURM_JOB_ID}.log"
if [ -n "${R4D_COMMAND:-}" ]; then
  echo "### ${R4D_COMMAND}"
  # shellcheck disable=SC2086
  python -u Codes/scripts/experiments/e100_components.py ${R4D_COMMAND}
else
  echo "### ${R4D_KEY}: ${R4D_STUDIES} with ${R4D_WORKERS} workers"
  # shellcheck disable=SC2086
  python -u Codes/scripts/experiments/e100_components.py run "${R4D_KEY}" ${R4D_STUDIES}
fi
mkdir -p "${R4D_E88_OUT}/${R4D_KEY}"
cp -f PROVENANCE.txt "${R4D_E88_OUT}/${R4D_KEY}/PROVENANCE_components_${SLURM_JOB_ID}.txt"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
