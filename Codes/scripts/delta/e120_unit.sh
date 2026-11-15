#!/bin/bash
# One E120 unit, a 16-core step inside a submit_e120_units.sbatch allocation:
#   e91:<key>:<arms joined by +>:<k/n>   one shard of a group of E91 arms (R4D_ARMS)
#   e96:v3n96o:<arms joined by +>:<k/n>  the same for E96
#   e30:-:-:<k/n>, e32:-:-:<k/n>         one shard of the preliminary study (R4D_SHARD)
# The E91/E96 rows go to results/e120/parts/<study>_<key>_<arms>/<study>_<key>_test
# and are resumable per sample; e120_merge_parts.py joins the parts afterwards.
# An E30/E32 shard writes its rows at the end; a cut-off one is rerun whole.
set -euo pipefail
IFS=: read -r STUDY KEY ARMS SHARD <<< "$1"
E120="/work/nvme/bhio/avu2/Rooted4D/results/e120"
CORES="${SLURM_CPUS_PER_TASK:-16}"
case "${STUDY}" in
  e91|e96)
    SCRIPT="e91_image_search.py"; [ "${STUDY}" = e96 ] && SCRIPT="e96_commissure_precision.py"
    OUT="${E120}/parts/${STUDY^^}_${KEY}_${ARMS}"
    mkdir -p "${OUT}"
    # A step stopped at the time limit can leave a cut-off last row; drop it so the resume parses.
    python - "${OUT}" <<'PY'
import json, sys
from pathlib import Path
for shard in Path(sys.argv[1]).glob("*/shard_*.jsonl"):
    keep = []
    for line in shard.read_text().splitlines(keepends=True):
        try:
            json.loads(line); keep.append(line if line.endswith("\n") else line + "\n")
        except json.JSONDecodeError:
            print("dropped a cut-off row in", shard)
    shard.write_text("".join(keep))
PY
    R4D_BENCH="${KEY}" R4D_OUT="${OUT}" R4D_SHARD="${SHARD}" R4D_SPLIT=test R4D_ALLOW_TEST=1 \
      R4D_ARMS="${ARMS//+/,}" R4D_WORKERS="$((CORES - 1))" \
      python -u "Codes/scripts/experiments/${SCRIPT}"
    ;;
  e30)
    mkdir -p "${E120}/e30"
    R4D_BENCHMARK="data/derived/semisynthetic-real-v3" R4D_MODES=32 R4D_SAMPLES=60 R4D_ITERS=400 \
      OMP_NUM_THREADS="${CORES}" R4D_OUT="${E120}/e30" R4D_SHARD="${SHARD}" \
      python -u Codes/scripts/experiments/e30_image_torch.py
    ;;
  e32)
    mkdir -p "${E120}/e32"
    R4D_BENCHMARK="data/derived/semisynthetic-real-v3" R4D_MODES=32 R4D_SAMPLES=30 \
      OMP_NUM_THREADS="${CORES}" R4D_OUT="${E120}/e32" R4D_SHARD="${SHARD}" \
      python -u Codes/scripts/experiments/e32_image_landmarks.py
    ;;
  *) echo "unknown unit $1"; exit 1 ;;
esac
echo "unit $1 finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
