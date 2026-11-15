#!/bin/bash
# Re-run Rooted4D's main configuration (Codes/configs/rooted4d_main.json) on the
# test sets it reports and compare sample by sample with its reference rows
# (results/benchmark/E116/reference_rows/). Selects nothing.
#   sbatch submit_reproduce_rooted4d_main.sbatch
#SBATCH --job-name=rooted4d-reproduce
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
for f in results/benchmark/E116/reference_rows/v4n96o_full.jsonl results/benchmark/E116/reference_rows/v3n96o_full.jsonl; do
  [ -e "$f" ] || { echo "missing reference rows: $f"; exit 1; }
done
OUT="${NVME}/results/reproduce_rooted4d_main/${SLURM_JOB_ID}"
mkdir -p "${OUT}" results/benchmark/E116
cp -f PROVENANCE.txt "${OUT}/PROVENANCE.txt"
{ echo "## node"; hostname; uname -srm; lscpu | grep -E "Model name|^CPU\(s\)"
  echo "## python"; python --version; python -c "import numpy; print('numpy', numpy.__version__)"
  echo "## pip freeze"; pip freeze 2>/dev/null; } > "${OUT}/ENVIRONMENT.txt"
S="Codes/scripts/experiments/reproduce_rooted4d_main.py"

echo "### unit tests of the residual field and the frozen file"
(cd Codes && python -m unittest tests.test_residual_field -q 2>&1 | tail -2)
echo "### data fingerprints"
python -u "${S}" check-data
echo "### re-run"
python -u "${S}" run "${OUT}"
echo "### compare"
python -u "${S}" compare "${OUT}" "results/benchmark/E116/REPRODUCTION_${SLURM_JOB_ID}.json"
cp -f "${OUT}/ENVIRONMENT.txt" "results/benchmark/E116/REPRODUCTION_${SLURM_JOB_ID}_ENVIRONMENT.txt"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
