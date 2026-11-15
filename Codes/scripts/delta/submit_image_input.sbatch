#!/bin/bash
#SBATCH --job-name=rooted4d-imgin
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
RESULTS="${NVME_ROOT}/results/${SLURM_JOB_ID:-manual}"
cd "${NVME_ROOT}/code"; mkdir -p logs "${RESULTS}"

# Data lives on the HDD tier and code on NVMe; join them with a symlink rather
# than copying 25 GB. rm -rf first because ln -sfn onto an existing directory
# creates the link *inside* it, silently.
rm -rf data && ln -sT "${HDD_DATA}" data
[ -L data ] || { echo "data is not a symlink, refusing to run"; exit 1; }

source .venv/bin/activate
export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export R4D_THREADS="${SLURM_CPUS_PER_TASK}"
export R4D_OUT="${RESULTS}"
export R4D_SEGMENTER="${NVME_ROOT}/results/21900728/E31_segmenter.pt"
# The split lock is staged under results_seed/ on the cluster; results/ is not
# synced, so the in-repo default path does not resolve here. Same file: md5
# 3b5c815e73e45cb1cd3f299d497d2f1d, identical to the local copy.
export R4D_SPLITS="${NVME_ROOT}/code/results_seed/acdc_patient_splits.json"
[ -f "${R4D_SPLITS}" ] || { echo "split file missing: ${R4D_SPLITS}"; exit 1; }
[ -f "${R4D_SEGMENTER}" ] || { echo "checkpoint missing: ${R4D_SEGMENTER}"; exit 1; }
export CUDA_VISIBLE_DEVICES=""

[ -f PROVENANCE.txt ] && { cp -f PROVENANCE.txt "${RESULTS}/"; grep -E '^(commit|dirty)=' PROVENANCE.txt; }

python Codes/scripts/experiments/e36_image_input_chamber.py > "${RESULTS}/e36.log" 2>&1
tail -12 "${RESULTS}/e36.log"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
