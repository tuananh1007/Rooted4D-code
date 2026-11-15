#!/bin/bash
# E120's long studies (E91, E96, E30, E32) split into 16-core units that each
# finish within one cpu-interactive job (59 min, up to four nodes, one job per
# user). The units are the script's arguments; see e120_unit.sh for their form.
# Each runs as its own step, and its log is logs/e120u_<job>_<unit>.log.
#   sbatch -p cpu-interactive -t 00:59:00 -n <units> -c 16 --mem-per-cpu=1000m \
#     submit_e120_units.sbatch e91:v3n96o:image_search:0/4 e30:-:-:0/4 ...
# A unit cut off by the time limit is resubmitted: E91/E96 resume per sample.
#SBATCH --job-name=rooted4d-e120u
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu-interactive
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1000m
#SBATCH --time=00:59:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
[ "$#" -gt 0 ] || { echo "no units given"; exit 1; }
NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
E120="${NVME_ROOT}/results/e120"
cd "${NVME_ROOT}/code"; mkdir -p logs "${E120}"
if [ -e data ] && [ ! -L data ]; then echo "data is not a symlink, refusing"; exit 1; fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then ln -sfT "${HDD_DATA}" data; fi
source .venv/bin/activate
export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
grep -E '^(commit|dirty)=' PROVENANCE.txt
if grep -q '^dirty=yes' PROVENANCE.txt; then echo "refusing: dirty tree"; exit 1; fi
cp -f PROVENANCE.txt "${E120}/PROVENANCE_units_${SLURM_JOB_ID}.txt"
echo "### ${SLURM_JOB_NUM_NODES} nodes (${SLURM_JOB_NODELIST}), $# units: $*"

echo "### the gauge tests"
python -m unittest discover -s Codes/tests -p "test_image_gauge.py" > "logs/e120_gauge_${SLURM_JOB_ID}.log" 2>&1 || {
  echo "gauge tests failed"; tail -20 "logs/e120_gauge_${SLURM_JOB_ID}.log"; exit 1; }
tail -1 "logs/e120_gauge_${SLURM_JOB_ID}.log"

declare -A UNIT
for unit in "$@"; do
  srun --exact -N1 -n1 -c "${SLURM_CPUS_PER_TASK}" --mem-per-cpu="${SLURM_MEM_PER_CPU}M" \
    bash Codes/scripts/delta/e120_unit.sh "${unit}" > "logs/e120u_${SLURM_JOB_ID}_${unit//[:\/+]/_}.log" 2>&1 &
  UNIT[$!]="${unit}"
done
failed=0
for pid in "${!UNIT[@]}"; do
  if wait "${pid}"; then echo "done   ${UNIT[$pid]}"; else echo "FAILED ${UNIT[$pid]} (exit $?)"; failed=$((failed + 1)); fi
done
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ): $(( $# - failed )) of $# units completed"
