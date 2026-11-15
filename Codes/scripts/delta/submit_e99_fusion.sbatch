#!/bin/bash
# E99 part 2 (Docs/E99_MITEA_FUSION_PREREGISTRATION_2026-09-29.md): the training
# subjects' geometry (the model), the code check on the validation subjects with
# nnU-Net's fold-0 validation predictions, then the test subjects' geometry,
# fits and analysis.
#SBATCH --job-name=rooted4d-e99-fusion
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
cd "${NVME_ROOT}/code"; mkdir -p logs
if [ -e data ] && [ ! -L data ]; then echo "data is not a symlink, refusing"; exit 1; fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then ln -sfT "${HDD_DATA}" data; fi
grep -E '^(commit|dirty)=' PROVENANCE.txt
if grep -q '^dirty=yes' PROVENANCE.txt; then echo "refusing: dirty tree"; exit 1; fi
source .venv/bin/activate
export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
S=Codes/scripts/experiments/e99_mitea_fusion.py
export R4D_MITEA="data/raw/mitea"
echo "### training subjects' geometry (the model)"
R4D_MODE=geometry R4D_SPLIT=train python -u "$S" | tail -3
echo "### code check: validation subjects, nnU-Net's fold-0 validation predictions"
VALPRED="${NVME_ROOT}/nnunet_results/Dataset504_MITEA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation"
R4D_MODE=geometry R4D_SPLIT=val R4D_MITEA_PRED="$VALPRED" python -u "$S" | tail -3
R4D_MODE=fit R4D_SPLIT=val python -u "$S" | tail -3
R4D_MODE=analyze R4D_SPLIT=val python -u "$S" > /dev/null
echo "### test subjects"
R4D_MODE=geometry R4D_SPLIT=test R4D_ALLOW_TEST=1 R4D_MITEA_PRED="data/nnunet/predictions_mitea_test" python -u "$S" | tail -3
R4D_MODE=fit R4D_SPLIT=test R4D_ALLOW_TEST=1 python -u "$S" | tail -3
R4D_MODE=analyze R4D_SPLIT=test python -u "$S" | head -60
mkdir -p "${NVME_ROOT}/results/e99"
cp -f results/benchmark/E99/*.json "${NVME_ROOT}/results/e99/"
cp -f PROVENANCE.txt "${NVME_ROOT}/results/e99/PROVENANCE_${SLURM_JOB_ID}.txt"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
