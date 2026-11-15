#!/bin/bash
# Resolution-sensitivity build for Stage B: donors and v3 at a 48-node annulus.
#SBATCH --job-name=rooted4d-build-n48
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
cd "${NVME_ROOT}/code"; mkdir -p logs
if [ -e data ] && [ ! -L data ]; then echo "data is not a symlink, refusing"; exit 1; fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then ln -sfT "${HDD_DATA}" data; fi
source .venv/bin/activate
export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
grep -E '^(commit|dirty)=' PROVENANCE.txt
if grep -q '^dirty=yes' PROVENANCE.txt; then echo "refusing: dirty tree"; exit 1; fi

# 48 angular samples per ring so the denser ring carries fitted information
# rather than interpolation of the 32-point polyline the default would give.
if [ ! -f data/derived/donors_v3_n48/donors.json ]; then
  echo "### donors at n = 48"
  python Codes/scripts/build_donor_pool.py --acdc-root data/raw/acdc --mvseg-root data/raw/mvseg2023 \
    --output data/derived/donors_v3_n48 --n-annulus 48 --n-angles 48 --valve-annulus 48
fi
if [ ! -f data/derived/semisynthetic-real-v3-n48/audit.json ]; then
  echo "### v3 at n = 48"
  python -m rooted4d.synthetic.dataset_cli generate \
    --config Codes/configs/semisynthetic_real_donors_v3_n48.yaml \
    --output data/derived/semisynthetic-real-v3-n48
fi
echo "### audit"
python -m rooted4d.synthetic.dataset_cli audit --root data/derived/semisynthetic-real-v3-n48 | tail -20
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
