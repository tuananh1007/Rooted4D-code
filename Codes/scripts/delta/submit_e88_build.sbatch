#!/bin/bash
# E88 (Docs/E88_REPLICATION_PREREGISTRATION_2026-09-28.md): the ten replication
# instances (v4 and v3, five split seeds), then a regeneration check of the
# original v4 instance on this machine against the copy built locally.
#SBATCH --job-name=rooted4d-e88-build
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96g
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_ROOT="/work/hdd/bhio/avu2/Rooted4D"
HDD_DATA="${HDD_ROOT}/data"
cd "${NVME_ROOT}/code"; mkdir -p logs
if [ -e data ] && [ ! -L data ]; then echo "data is not a symlink, refusing"; exit 1; fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then ln -sfT "${HDD_DATA}" data; fi
source .venv/bin/activate
export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
grep -E '^(commit|dirty)=' PROVENANCE.txt
if grep -q '^dirty=yes' PROVENANCE.txt; then echo "refusing: dirty tree"; exit 1; fi
test -f results/benchmark/E25b_session_subspace_split.npz || { echo "session subspace missing"; exit 1; }

build() {
  local bench="$1" k="$2" out="data/derived/semisynthetic-real-$1-split$2" rc=0
  if [ -f "${out}/audit.json" ]; then echo "${out} exists"; return 0; fi
  # The exit status is captured explicitly: under set -e a failing command would
  # end the subshell before any report of it.
  if python -m rooted4d.synthetic.dataset_cli generate \
      --config "Codes/configs/e88/semisynthetic_real_donors_${bench}_split${k}.yaml" \
      --output "${out}" > "logs/e88_build_${bench}_split${k}_${SLURM_JOB_ID}.json" 2>&1; then rc=0; else rc=$?; fi
  echo "${out}: exit ${rc}"
  return "${rc}"
}
pids=()
for bench in v4 v3; do for k in 1 2 3 4 5; do build "${bench}" "${k}" & pids+=($!); done; done
regenerated="${HDD_ROOT}/scratch/e88_v4_regenerated"
rm -rf "${regenerated}"; mkdir -p "${HDD_ROOT}/scratch"
python -m rooted4d.synthetic.dataset_cli generate --config Codes/configs/semisynthetic_real_donors_v4.yaml \
  --output "${regenerated}" > "logs/e88_v4_regenerated_${SLURM_JOB_ID}.json" 2>&1 &
regeneration=$!
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
# The regeneration is a reported check only, so its failure does not fail the job.
wait "${regeneration}" || echo "regeneration of the original v4 failed; see its log"
echo "### audits"
for bench in v4 v3; do for k in 1 2 3 4 5; do
  root="data/derived/semisynthetic-real-${bench}-split${k}"
  if [ ! -f "${root}/audit.json" ]; then echo "${root}: not built"; status=1; continue; fi
  python - "${root}" <<'PY' || status=1
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
dataset = json.loads((root / "dataset.json").read_text())
audit = json.loads((root / "audit.json").read_text())
counts = {split: len(list((root / "samples" / split).glob("*.npz"))) for split in ("train", "val", "test")}
print(root.name, "audit passed:", audit.get("passed"), "samples:", counts)
PY
done; done
echo "### regeneration check of the original v4 (reported; it does not gate E88)"
# The tree fingerprint would always differ: config.resolved.json records the
# session subspace's absolute path and dataset.json the generator's source hash,
# which changed after the local build. The data are compared instead: every
# sample array, the manifests and the seeds.
sha256sum results/benchmark/E25b_session_subspace_split.npz
python - data/derived/semisynthetic-real-v4 "${regenerated}" <<'PY' || true
import hashlib, sys
from pathlib import Path
import numpy as np
left, right = (Path(a) for a in sys.argv[1:3])
def files(root):
    return sorted(p.relative_to(root) for p in list(root.glob("samples/*/*.npz"))
                  + list(root.glob("manifests/*.jsonl")) + [root / "seeds.json"])
a, b = files(left), files(right)
print("files:", len(a), "local copy,", len(b), "regenerated; same names:", a == b)
different = [p for p in a if p in set(b) and hashlib.sha256((left / p).read_bytes()).digest()
             != hashlib.sha256((right / p).read_bytes()).digest()]
print("byte-identical:", len(a) - len(different), "of", len(a))
for p in different[:5]:
    if p.suffix == ".npz":
        x, y = np.load(left / p, allow_pickle=True), np.load(right / p, allow_pickle=True)
        worst = max((float(np.abs(x[k].astype(float) - y[k].astype(float)).max())
                     for k in x.files if x[k].dtype.kind in "fiu" and x[k].shape == y[k].shape), default=0.0)
        print("  differs:", p, "largest numeric difference", worst)
    else:
        print("  differs:", p)
PY
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ) status ${status}"
exit "${status}"
