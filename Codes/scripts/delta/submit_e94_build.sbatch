#!/bin/bash
# E94 (Docs/E94_ORIENTED_COMPOSITES_PREREGISTRATION_2026-09-28.md): the
# realistic-session benchmark rebuilt with each valve oriented to its chamber.
#SBATCH --job-name=rooted4d-e94-build
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
OUT=data/derived/semisynthetic-real-v4-oriented
if [ ! -f "${OUT}/audit.json" ]; then
  python -m rooted4d.synthetic.dataset_cli generate \
    --config Codes/configs/semisynthetic_real_donors_v4_oriented.yaml --output "${OUT}"
fi
python -m rooted4d.synthetic.dataset_cli audit --root "${OUT}" | tail -8
echo "### orientation and correspondence with the original instance"
python - "${OUT}" data/derived/semisynthetic-real-v4 <<'PY'
import sys
from pathlib import Path
import numpy as np
from rooted4d.experiments.fitted_geometry import ring_groups
oriented, original = (Path(a) for a in sys.argv[1:3])
g = ring_groups(24)
def height(x):
    ring = x[g["annulus"]].mean(0); apex = x[g["apex"][0]]
    axis = (apex - ring) / np.linalg.norm(apex - ring)
    return float(((x[g["free"]] - ring) @ axis).mean())
for split in ("train", "val", "test"):
    a = sorted(p.name for p in (oriented / "samples" / split).glob("*.npz"))
    b = sorted(p.name for p in (original / "samples" / split).glob("*.npz"))
    heights = [height(np.load(oriented / "samples" / split / n)["template_vertices"]) for n in a]
    before = [height(np.load(original / "samples" / split / n)["template_vertices"]) for n in b]
    print(split, "samples", len(a), "same sample ids as the original:", a == b,
          "| free-edge height: oriented %.1f [%.1f, %.1f] mm, original %.1f mm"
          % (np.mean(heights), np.min(heights), np.max(heights), np.mean(before)))
PY
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
