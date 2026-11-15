#!/bin/bash
# Re-measure the valve component after correcting the MVSeg2023 label convention.
# The dataset card gives 1 = posterior, 2 = anterior; this repository had them
# reversed, which swapped the two leaflets in every valve number and in the
# donor pool's leaflet free edges.
#SBATCH --job-name=rooted4d-valve
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
export PYTHONUNBUFFERED=1
NVME_ROOT="/work/nvme/bhio/${USER}/Rooted4D"
HDD_ROOT="/work/hdd/bhio/${USER}/Rooted4D"
RESULTS="${NVME_ROOT}/results/${SLURM_JOB_ID:-manual}"
cd "${NVME_ROOT}/code"; mkdir -p logs "${RESULTS}"
rm -rf data && ln -sT "${HDD_ROOT}/data" data
[ -L data ] || { echo "ERROR: data is not a symlink" >&2; exit 1; }
PY="${NVME_ROOT}/code/.venv/bin/python"
export PYTHONPATH="${NVME_ROOT}/code/Codes" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export R4D_OUT="${RESULTS}"
[ -f PROVENANCE.txt ] && { cp -f PROVENANCE.txt "${RESULTS}/"; grep -E '^(commit|dirty)=' PROVENANCE.txt; }

$PY - > "${RESULTS}/valve_recheck.log" 2>&1 <<'PYEOF'
import json
import numpy as np
from rooted4d.experiments.component_eval import evaluate_valve_component
from rooted4d.donors.valve import ANTERIOR_LABEL, POSTERIOR_LABEL
print("label convention in use: anterior=%d posterior=%d" % (ANTERIOR_LABEL, POSTERIOR_LABEL))
r = evaluate_valve_component("data/raw/mvseg2023", limit=None)
import os
json.dump(r, open(os.environ["R4D_OUT"] + "/component_valve_corrected.json", "w"), indent=2)
print("cases %d" % r["n_cases"])
for k, v in r["summary"].items():
    print("  %-28s median %7.3f  p95 %7.3f" % (k, v["median"], v["p95"]))
a = r["summary"]["anterior_span_mm"]["median"]; p = r["summary"]["posterior_span_mm"]["median"]
print("\nanterior/posterior span ratio %.2f (mitral anatomy expects the anterior leaflet longer)" % (a / p))
PYEOF
tail -20 "${RESULTS}/valve_recheck.log"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
