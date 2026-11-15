#!/bin/bash
# The current Rooted4D experiment set, on NCSA Delta's CPU partition.
#
# Supersedes submit_experiments.sbatch, whose header still says there is no
# torch code in this repository. There is now: rooted4d.differentiable replaces
# the numerical Jacobian that limited the image-domain work. It is a CPU build
# and this job charges the CPU account, because none of it needs a GPU and the
# *-delta-gpu balance is scarce.
#
# Storage, following the project policy:
#   NVMe (/work/nvme) -> code, venv, result JSON      (I/O-hot, small)
#   HDD  (/work/hdd)  -> provider data and benchmarks (large, regenerable)
# The code tree gets a data/ symlink onto the HDD tier so the relative paths the
# experiments use resolve without every script learning about the cluster.
#
#SBATCH --job-name=rooted4d-current
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

NVME_ROOT="/work/nvme/bhio/${USER}/Rooted4D"
HDD_ROOT="/work/hdd/bhio/${USER}/Rooted4D"
RUN_ID="${SLURM_JOB_ID:-manual-$(date +%Y%m%dT%H%M%S)}"
RESULTS="${NVME_ROOT}/results/${RUN_ID}"

cd "${NVME_ROOT}/code"
mkdir -p logs "${RESULTS}"

# Provider data lives on the bulk tier; the experiments address it as ./data.
# -T is not optional here: if data/ already exists as a real directory, plain
# `ln -sfn` silently creates the link *inside* it and every path then resolves
# to whatever stale copy is there. That happened once, and the job would have
# read a partial dataset without complaining.
rm -rf data
ln -sT "${HDD_ROOT}/data" data
[ -L data ] || { echo "ERROR: data is not a symlink" >&2; exit 1; }

# The venv is excluded from the sync, so it stays at the code-tree root
# rather than moving under Codes/ with the package.
PY="${NVME_ROOT}/code/.venv/bin/python"
export PYTHONPATH="${NVME_ROOT}/code/Codes"
export R4D_OUT="${RESULTS}"

echo "=== provenance ==="
echo "run_id      : ${RUN_ID}"
echo "host        : $(hostname)"
echo "started_utc : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ -f PROVENANCE.txt ]; then
  echo "git_commit  : $(grep -m1 '^commit=' PROVENANCE.txt | cut -d= -f2)"
  echo "git_dirty   : $(grep -m1 '^dirty=' PROVENANCE.txt | cut -d= -f2)"
  cp -f PROVENANCE.txt "${RESULTS}/"
fi
echo "python      : $($PY -V 2>&1)"
echo "torch       : $($PY -c 'import torch;print(torch.__version__)' 2>&1)"
echo "cpus        : ${SLURM_CPUS_PER_TASK:-unknown}"
echo "data        : $(readlink -f data)"

# One process, many cores. BLAS threading is pinned for the NumPy stages so the
# linear algebra is reproducible across nodes with different core counts; torch
# is given the cores explicitly instead.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export R4D_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo
echo "=== 1/5 regression suite ==="
# Full output to the results directory. Piping to `tail -4` showed only the
# summary line, so when four tests errored on the cluster the job aborted with
# no record of which ones, and diagnosing it needed a second job. A truncated
# failure report is worse than none, because it looks like a report.
set +e
$PY -m unittest discover -s Codes/tests > "${RESULTS}/tests.log" 2>&1
TEST_STATUS=$?
set -e
tail -4 "${RESULTS}/tests.log"
if [ "${TEST_STATUS}" -ne 0 ]; then
  echo "--- failures and errors ---"
  grep -E "^(FAIL|ERROR):" "${RESULTS}/tests.log" | head -20
  echo "regression suite failed; see ${RESULTS}/tests.log" >&2
  exit "${TEST_STATUS}"
fi

echo
echo "=== 2/5 session term at the corrected parameterization (E28) ==="
R4D_BENCHMARK="data/derived/semisynthetic-real-v4" \
  $PY Codes/scripts/experiments/e28_session_offset.py > "${RESULTS}/e28_session_offset.log" 2>&1 || { tail -40 "${RESULTS}/e28_session_offset.log"; exit 1; }
tail -25 "${RESULTS}/e28_session_offset.log"

echo
echo "=== 3/5 prior under evidence scarcity (E29) ==="
R4D_BENCHMARK="data/derived/semisynthetic-real-v4" R4D_SAMPLES=40 \
  $PY Codes/scripts/experiments/e29_prior_scarcity.py > "${RESULTS}/e29_prior_scarcity.log" 2>&1 || { tail -40 "${RESULTS}/e29_prior_scarcity.log"; exit 1; }
tail -20 "${RESULTS}/e29_prior_scarcity.log"

echo
echo "=== 4/5 image-domain fitting at full scale (E30) ==="
# 32 modes and the whole test split, which the numerical Jacobian could not
# afford: it needed 486 s per sample against 15 s for the gradient method.
R4D_BENCHMARK="data/derived/semisynthetic-real-v3" \
  R4D_MODES=32 R4D_SAMPLES=60 R4D_ITERS=400 \
  OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" \
  $PY Codes/scripts/experiments/e30_image_torch.py > "${RESULTS}/e30_image_torch.log" 2>&1 || { tail -40 "${RESULTS}/e30_image_torch.log"; exit 1; }
tail -20 "${RESULTS}/e30_image_torch.log"

echo
echo "=== 5/5 segmentation front end, longer schedule (E31) ==="
R4D_EPOCHS=60 R4D_BATCH=24 R4D_BUDGET_S=7200 \
  R4D_SPLITS="${NVME_ROOT}/code/results_seed/acdc_patient_splits.json" \
  OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" \
  $PY Codes/scripts/experiments/e31_segmentation.py > "${RESULTS}/e31_segmentation.log" 2>&1 || { tail -40 "${RESULTS}/e31_segmentation.log"; exit 1; }
tail -25 "${RESULTS}/e31_segmentation.log"

echo
echo "finished_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "results in  : ${RESULTS}"
