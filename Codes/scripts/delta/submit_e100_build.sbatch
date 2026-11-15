#!/bin/bash
# E100 (Docs/E100_ORIENTED_REBUILD_PREREGISTRATION_2026-09-29.md): build one
# oriented instance and check it: the audit, a positive free-edge height for every
# sample, and the counterpart's sample ids where a non-oriented counterpart exists.
#   sbatch --export=ALL,R4D_KEY=v4n96o submit_e100_build.sbatch
#SBATCH --job-name=rooted4d-e100-build
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
: "${R4D_KEY:?set R4D_KEY}"
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
OUT="data/derived/e100-${R4D_KEY}"
if [ ! -f "${OUT}/audit.json" ]; then
  python -m rooted4d.synthetic.dataset_cli generate --config "Codes/configs/e100/${R4D_KEY}.yaml" --output "${OUT}"
fi
python -m rooted4d.synthetic.dataset_cli audit --root "${OUT}" | tail -6
echo "### checks"
python - "${R4D_KEY}" "${OUT}" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "Codes/scripts/experiments")
import e100_rebuild as e100
from rooted4d.experiments.fitted_geometry import ring_groups
key, out = sys.argv[1], Path(sys.argv[2])
audit = json.loads((out / "audit.json").read_text())
assert audit.get("passed", True) and not audit.get("errors"), "audit failed"
_, _, _, counterpart = e100.resolve(key)
def height(x):
    g = ring_groups((len(x) - 1) // 3)
    ring = x[g["annulus"]].mean(0); apex = x[g["apex"][0]]
    axis = (apex - ring) / np.linalg.norm(apex - ring)
    return float(((x[g["free"]] - ring) @ axis).mean())
failed = False
for split in ("train", "val", "test"):
    names = sorted(p.name for p in (out / "samples" / split).glob("*.npz"))
    heights = np.array([height(np.load(out / "samples" / split / n)["template_vertices"]) for n in names])
    line = f"{split}: {len(names)} samples, free-edge height {heights.mean():+.1f} mm [{heights.min():+.1f}, {heights.max():+.1f}]"
    if (heights <= 0).any():
        failed = True; line += "  NON-POSITIVE"
    if counterpart:
        other = sorted(p.name for p in (Path(counterpart) / "samples" / split).glob("*.npz"))
        same = names == other
        failed |= not same; line += f" | same sample ids as {counterpart}: {same}"
    print(line)
if failed:
    raise SystemExit("a build check failed")
print("checks passed")
PY
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
