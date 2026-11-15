#!/bin/bash
# E102 (Docs/E102_RICHER_INSTANCE_PREREGISTRATION_2026-10-03.md): the method
# ordering on a richer procedural instance (plan v2, W2.6).
#   sbatch --export=ALL,R4D_STAGE=prepare  --cpus-per-task=8  --time=04:00:00 submit_e102.sbatch
#   sbatch --export=ALL,R4D_STAGE=protocol --cpus-per-task=32 --time=12:00:00 --dependency=afterok:<prepare> submit_e102.sbatch
# prepare: the donor pool at six chamber rings and three leaflet rows (if absent),
#          the instance (if absent), its audit and the build checks.
# protocol: the full test suite, E88's chain (resumable), the cross-validation,
#           and the analysis against E100's v3o.
#SBATCH --job-name=rooted4d-e102
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
: "${R4D_STAGE:?set R4D_STAGE to prepare or protocol}"
NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
cd "${NVME_ROOT}/code"; mkdir -p logs
if [ -e data ] && [ ! -L data ]; then echo "data is not a symlink, refusing"; exit 1; fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then ln -sfT "${HDD_DATA}" data; fi
source .venv/bin/activate
export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export R4D_WORKERS="${R4D_WORKERS:-${SLURM_CPUS_PER_TASK}}"
export R4D_E88_OUT="${NVME_ROOT}/results/e102" R4D_E100_OUT="${NVME_ROOT}/results/e100"
grep -E '^(commit|dirty)=' PROVENANCE.txt
if grep -q '^dirty=yes' PROVENANCE.txt; then echo "refusing: dirty tree"; exit 1; fi
SCRIPT=Codes/scripts/experiments/e102_rich_instance.py
POOL=data/derived/donors_v3_rich24
OUT=data/derived/e102-v3or
case "${R4D_STAGE}" in
  prepare)
    if [ ! -f "${POOL}/donors.json" ]; then
      echo "### donor pool"
      python Codes/scripts/build_donor_pool.py --output "${POOL}" --n-annulus 24 --chamber-levels 6 --leaflet-rows 3
    fi
    if [ ! -f "${OUT}/audit.json" ]; then
      echo "### instance"
      python -m rooted4d.synthetic.dataset_cli generate --config Codes/configs/e102/v3or.yaml --output "${OUT}"
    fi
    python -m rooted4d.synthetic.dataset_cli audit --root "${OUT}" | tail -6
    echo "### checks"
    python "${SCRIPT}" check
    ;;
  protocol)
    echo "### full suite first"
    python -m unittest discover -s Codes/tests -q > "logs/e102_tests_${SLURM_JOB_ID}.log" 2>&1 || {
      echo "tests failed"; tail -20 "logs/e102_tests_${SLURM_JOB_ID}.log"; exit 1; }
    tail -2 "logs/e102_tests_${SLURM_JOB_ID}.log"
    echo "### chain with ${R4D_WORKERS} workers"
    if [ ! -f "${R4D_E88_OUT}/v3or/TEST_DONE" ]; then python -u "${SCRIPT}" chain; fi
    echo "### cross-validation"
    python -u "${SCRIPT}" gate
    echo "### analysis"
    python -u "${SCRIPT}" analyze
    mkdir -p "${R4D_E88_OUT}/v3or"
    cp -f PROVENANCE.txt "${R4D_E88_OUT}/v3or/PROVENANCE_${SLURM_JOB_ID}.txt"
    ;;
  *) echo "unknown stage ${R4D_STAGE}"; exit 1 ;;
esac
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
