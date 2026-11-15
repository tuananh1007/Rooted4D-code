#!/bin/bash
# E120 (Docs/E120_IMAGE_GAUGE_RERUN_PREREGISTRATION_2026-11-13.md): the image
# studies rerun with the session gauge applied. Stages (R4D_STAGE):
#   components  R4D_KEY, R4D_STUDIES: e100_components.py run into results/e120,
#               whose instance roots link E100's except the rerun directories.
#               Resumable: rerun the same stage until it reports complete.
#   e106-twin   build E106's no-shadow twin of v3n96o, audit it, run E106's check
#   e106-fit    R4D_INSTANCE=original|twin: E106's fits into results/e120/e106
#   e30, e32    the preliminary image studies at their original settings
# Partition, cores and time are set on the command line, e.g.
#   sbatch -p cpu-interactive -t 00:59:00 -c 64 --mem=120g \
#     --export=ALL,R4D_STAGE=components,R4D_KEY=v3n96o,R4D_STUDIES=e70 submit_e120.sbatch
#SBATCH --job-name=rooted4d-e120
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
: "${R4D_STAGE:?set R4D_STAGE}"
NVME_ROOT="/work/nvme/bhio/avu2/Rooted4D"
HDD_DATA="/work/hdd/bhio/avu2/Rooted4D/data"
E120="${NVME_ROOT}/results/e120"
cd "${NVME_ROOT}/code"; mkdir -p logs "${E120}"
if [ -e data ] && [ ! -L data ]; then echo "data is not a symlink, refusing"; exit 1; fi
if [ ! -L data ] || [ "$(readlink data)" != "${HDD_DATA}" ]; then ln -sfT "${HDD_DATA}" data; fi
source .venv/bin/activate
export PYTHONPATH="${NVME_ROOT}/code/Codes:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export R4D_WORKERS="${R4D_WORKERS:-${SLURM_CPUS_PER_TASK}}"
grep -E '^(commit|dirty)=' PROVENANCE.txt
if grep -q '^dirty=yes' PROVENANCE.txt; then echo "refusing: dirty tree"; exit 1; fi
STAMP="${E120}/PROVENANCE_${R4D_STAGE}_${SLURM_JOB_ID}.txt"
cp -f PROVENANCE.txt "${STAMP}"

echo "### the gauge tests"
python -m unittest discover -s Codes/tests -p "test_image_gauge.py" > "logs/e120_gauge_${SLURM_JOB_ID}.log" 2>&1 || {
  echo "gauge tests failed"; tail -20 "logs/e120_gauge_${SLURM_JOB_ID}.log"; exit 1; }
tail -1 "logs/e120_gauge_${SLURM_JOB_ID}.log"
if [ "${R4D_FULL_SUITE:-0}" = "1" ]; then
  echo "### full suite"
  python -m unittest discover -s Codes/tests -q > "logs/e120_tests_${SLURM_JOB_ID}.log" 2>&1 || {
    echo "tests failed"; tail -20 "logs/e120_tests_${SLURM_JOB_ID}.log"; exit 1; }
  tail -2 "logs/e120_tests_${SLURM_JOB_ID}.log"
fi

#: Component directories that E120 refits. Everything else is linked from E100.
RERUN="E70_v3n96o_test E71_v3n96o_test E74_v3n96o_test E91_v3n96o_test E96_v3n96o_test E91_v4n96o_test"

link_root() {  # an E120 instance root that links E100's except the rerun directories
  local key="$1" source dest name entry
  source="${NVME_ROOT}/results/e100/${key}"; dest="${E120}/${key}"
  [ -d "${source}/components" ] || { echo "no E100 components for ${key}"; exit 1; }
  mkdir -p "${dest}/components"
  for entry in "${source}"/*; do
    name="$(basename "${entry}")"; [ "${name}" = components ] && continue
    [ -e "${dest}/${name}" ] || ln -s "${entry}" "${dest}/${name}"
  done
  for entry in "${source}/components"/*; do
    name="$(basename "${entry}")"
    case " ${RERUN} " in *" ${name} "*) continue ;; esac
    [ -e "${dest}/components/${name}" ] || ln -s "${entry}" "${dest}/components/${name}"
  done
}

case "${R4D_STAGE}" in
  components)
    : "${R4D_KEY:?set R4D_KEY}"; : "${R4D_STUDIES:?set R4D_STUDIES}"
    link_root "${R4D_KEY}"
    export R4D_E88_OUT="${E120}"
    echo "### ${R4D_KEY}: ${R4D_STUDIES} with ${R4D_WORKERS} workers"
    # shellcheck disable=SC2086
    python -u Codes/scripts/experiments/e100_components.py run "${R4D_KEY}" ${R4D_STUDIES}
    ;;
  e106-twin)
    mkdir -p "${E120}/e106"
    TWIN="data/derived/e120-v3n96o-noshadow"
    # Build from the committed twin config: its donor path is relative to Codes/configs/e106
    # and does not resolve from a copy in the results root (see e120_twin_build.sh).
    CONFIG="Codes/configs/e106/v3n96o_noshadow.yaml"
    python Codes/scripts/experiments/e106_image_diagnosis.py config "${E120}/e106/twin.yaml"
    cmp "${E120}/e106/twin.yaml" "${CONFIG}"
    if [ ! -f "${TWIN}/audit.json" ]; then
      python -m rooted4d.synthetic.dataset_cli generate --config "${CONFIG}" --output "${TWIN}"
    fi
    python -m rooted4d.synthetic.dataset_cli audit --root "${TWIN}" | tail -6
    # E106's check writes into results/benchmark/E106; keep the stored one.
    CHECK="results/benchmark/E106/E106_twin_check.json"
    [ -f "${CHECK}" ] && cp -f "${CHECK}" "${E120}/e106/E106_twin_check_stored.json"
    python Codes/scripts/experiments/e106_image_diagnosis.py check data/derived/e100-v3n96o "${TWIN}"
    cp -f "${CHECK}" "${E120}/e106/E106_twin_check.json"
    [ -f "${E120}/e106/E106_twin_check_stored.json" ] && cp -f "${E120}/e106/E106_twin_check_stored.json" "${CHECK}"
    ;;
  e106-fit)
    : "${R4D_INSTANCE:?set R4D_INSTANCE to original or twin}"
    if [ "${R4D_INSTANCE}" = twin ]; then ROOT="data/derived/e120-v3n96o-noshadow"; else ROOT="data/derived/e100-v3n96o"; fi
    # A job stopped at its time limit can leave a cut-off last row; drop it so the resume parses.
    python - "${E120}/e106/E106_${R4D_INSTANCE}_test" <<'PY'
import json, sys
from pathlib import Path
for shard in Path(sys.argv[1]).glob("shard_*.jsonl"):
    lines = shard.read_text().splitlines(keepends=True)
    keep = []
    for line in lines:
        try:
            json.loads(line); keep.append(line if line.endswith("\n") else line + "\n")
        except json.JSONDecodeError:
            print("dropped a cut-off row in", shard)
    shard.write_text("".join(keep))
PY
    R4D_E69="${NVME_ROOT}/results/e100/v3n96o/components/E69_v3n96o_test" R4D_ALLOW_TEST=1 \
      R4D_OUT="${E120}/e106" R4D_SHARD=0/1 R4D_SPLIT=test \
      python -u Codes/scripts/experiments/e106_image_diagnosis.py fit "${ROOT}"
    ;;
  e30)
    mkdir -p "${E120}/e30"
    R4D_BENCHMARK="data/derived/semisynthetic-real-v3" R4D_MODES=32 R4D_SAMPLES=60 R4D_ITERS=400 \
      OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" R4D_OUT="${E120}/e30" \
      python -u Codes/scripts/experiments/e30_image_torch.py > "${E120}/e30/e30_image_torch.log" 2>&1 || {
        tail -40 "${E120}/e30/e30_image_torch.log"; exit 1; }
    tail -20 "${E120}/e30/e30_image_torch.log"
    ;;
  e32)
    mkdir -p "${E120}/e32"
    R4D_BENCHMARK="data/derived/semisynthetic-real-v3" R4D_MODES=32 R4D_SAMPLES=30 \
      OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" R4D_OUT="${E120}/e32" \
      python -u Codes/scripts/experiments/e32_image_landmarks.py > "${E120}/e32/e32_image_landmarks.log" 2>&1 || {
        tail -40 "${E120}/e32/e32_image_landmarks.log"; exit 1; }
    tail -25 "${E120}/e32/e32_image_landmarks.log"
    ;;
  *) echo "unknown stage ${R4D_STAGE}"; exit 1 ;;
esac
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
