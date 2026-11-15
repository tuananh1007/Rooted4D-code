#!/bin/bash
#SBATCH --job-name=rooted4d-gtval
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
D="${NVME_ROOT}/external/4D-Myocardium-Reconstruction-with-Decoupled-Motion-and-Shape-Model"
cd "${NVME_ROOT}/code"; mkdir -p logs
RESULTS="${NVME_ROOT}/results/${SLURM_JOB_ID}_gtval"; mkdir -p "${RESULTS}"

# Several jobs share this working directory, so the data symlink must not be
# rewritten on every run: replacing it, even with an identical target, opens a
# window in which a concurrent job reading through it gets a FileNotFoundError
# on a file that exists. Only create it when it is absent or points elsewhere.
if [ -e data ] && [ ! -L data ]; then
  echo "data exists and is not a symlink, refusing to run"; exit 1
fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then
  ln -sfT "${HDD_DATA}" data
fi

export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export R4D_4DMR_OUTPUTS="${D}/outputs"
export R4D_4DMR_INPUTS="${D}/inputs"
export R4D_ACDC="${HDD_DATA}/raw/acdc"
export R4D_TETHEART_GT="${HDD_DATA}/tetheart_gt"
export R4D_OUT="${RESULTS}"
mkdir -p "${R4D_TETHEART_GT}"

"${NVME_ROOT}/external/.venv-4dmr/bin/python" Codes/scripts/experiments/e44_4dmr_to_tetheart.py \
  > "${RESULTS}/e44.log" 2>&1 || { echo "e44 failed"; tail -30 "${RESULTS}/e44.log"; exit 1; }
tail -40 "${RESULTS}/e44.log"
echo "=== accepted frames on disk: $(find ${R4D_TETHEART_GT} -name '*_mesh_lv.obj' | wc -l)"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
