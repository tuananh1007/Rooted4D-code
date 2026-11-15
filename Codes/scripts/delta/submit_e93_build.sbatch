#!/bin/bash
# E93 (Docs/E93_REALISTIC_SESSION_96_PREREGISTRATION_2026-09-28.md): the
# realistic-session benchmark at 96 ring nodes, from the 96-node donors and the
# transferred MITEA session subspace.
#SBATCH --job-name=rooted4d-e93-build
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48g
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
test -f data/derived/donors_v3_n96/donors.json || { echo "96-node donors missing"; exit 1; }
test -f results/benchmark/E93_session_subspace_split_n96.npz || { echo "transferred subspace missing"; exit 1; }
OUT=data/derived/semisynthetic-real-v4-n96
if [ ! -f "${OUT}/audit.json" ]; then
  python -m rooted4d.synthetic.dataset_cli generate \
    --config Codes/configs/semisynthetic_real_donors_v4_n96.yaml --output "${OUT}"
fi
python -m rooted4d.synthetic.dataset_cli audit --root "${OUT}" | tail -20
for split in train val test; do echo "${split}: $(ls ${OUT}/samples/${split} | wc -l) samples"; done
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
