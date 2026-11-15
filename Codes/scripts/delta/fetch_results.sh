#!/usr/bin/env bash
# Pull one run's results from Delta's NVMe tier into the local results/ tree.
#
# Only the small JSON/text summaries come back. The generated benchmark stays on
# the cluster HDD tier: it is large, regenerable from its recorded seeds, and
# does not belong in git.
set -euo pipefail

RUN_ID="${1:?usage: fetch_results.sh <slurm_job_id> [remote]}"
REMOTE="${2:-delta}"
# rsync does not expand $USER inside a remote path; resolve it first.
REMOTE_USER="$(ssh -o BatchMode=yes "$REMOTE" 'echo $USER')"
NVME="/work/nvme/bhio/${REMOTE_USER}/Rooted4D"
# Three levels up from Codes/scripts/delta is the workspace root; two
# levels lands in Codes/, which is where results were being fetched to
# after the reorganization.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEST="${HERE}/results/${RUN_ID}"

mkdir -p "${DEST}"
rsync -az "${REMOTE}:${NVME}/results/${RUN_ID}/" "${DEST}/"
rsync -az "${REMOTE}:${NVME}/code/logs/rooted4d-campaign_${RUN_ID}.out" \
  "${DEST}/slurm.out" 2>/dev/null || echo "note: slurm stdout log not found"
echo "fetched run ${RUN_ID} -> ${DEST}"
