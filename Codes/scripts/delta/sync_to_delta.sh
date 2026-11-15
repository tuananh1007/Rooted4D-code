#!/usr/bin/env bash
# Push the working tree to Delta's NVMe tier and stamp it with git provenance.
#
# .git is deliberately not synced (the cluster does not need history, and the
# results are committed from the local checkout). PROVENANCE.txt carries the
# commit identity instead, so every result directory can be traced to a tree.
set -euo pipefail

REMOTE="${1:-delta}"
# rsync does not expand $USER inside a remote path, so resolve it over the
# existing ControlMaster socket instead of hardcoding a username.
REMOTE_USER="$(ssh -o BatchMode=yes "$REMOTE" 'echo $USER')"
DEST="${2:-/work/nvme/bhio/${REMOTE_USER}/Rooted4D/code}"
# Codes/scripts/delta/ -> Codes/scripts -> Codes -> workspace. The workspace is
# what gets synced, because Docs and the governing artifacts at its root are
# what the readiness contracts check.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$HERE"

DIRTY=no
git diff --quiet && git diff --cached --quiet || DIRTY=yes
cat > PROVENANCE.txt <<EOF
commit=$(git rev-parse HEAD)
branch=$(git rev-parse --abbrev-ref HEAD)
dirty=${DIRTY}
synced_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
synced_from=$(hostname)
EOF

if [ "$DIRTY" = yes ]; then
  echo "WARNING: working tree is dirty; results will not match any commit exactly." >&2
fi

# Each sbatch stage is a separate `python -m` invocation, so overwriting the
# code while a job is running makes that job execute different versions in
# different stages and silently invalidates its PROVENANCE stamp. Refuse unless
# the caller explicitly accepts that.
# Guard on *any* of this project's jobs, not one name. The original filter
# matched only "rooted4d-campaign", and every job added since -- the ladder,
# the external comparisons, the TetHeart runs -- shares the same mutable code
# tree while carrying a different name, so they were syncable from under
# themselves.
RUNNING="$(ssh -o BatchMode=yes "$REMOTE" \
  "squeue --me -h -o '%i %j'" 2>/dev/null | awk '$2 ~ /^rooted4d/ {printf "%s ", $1}')"
if [ -n "${RUNNING// /}" ] && [ "${ALLOW_SYNC_DURING_RUN:-0}" != "1" ]; then
  echo "ERROR: rooted4d job(s) ${RUNNING}still running on ${REMOTE}." >&2
  echo "Syncing now would leave those runs straddling two commits." >&2
  echo "Wait for them, or set ALLOW_SYNC_DURING_RUN=1 to override." >&2
  rm -f PROVENANCE.txt
  exit 1
fi

# data/ and datasets/ are excluded deliberately. NVMe is the code tier and
# HDD is the bulk tier, and this rsync targets the code tier. Without these
# excludes it pushes roughly 25 GB of provider archives and extracted volumes
# onto NVMe, which contradicts the storage policy this repository states and
# takes long enough that the first attempt had to be killed part-way. Bulk
# data goes up once, to HDD, with push_data_to_delta.sh; it has no business
# in a sync that runs on every commit.
#
# The patterns are anchored with a leading slash. Unanchored, 'data/' matches a
# directory of that name at any depth, which silently swept up
# tests/data/example_split_lock.json -- and with --delete, removed it from the
# remote. Four readiness tests then errored on the cluster and passed locally.
#
# The unattended chains (E88, E93, E94) write their generated stage specs into
# Codes/configs/<experiment>/<instance>/ on the cluster, and those exist only
# there. Without the protect rule, --delete removed them while E88 and E93
# were running (29 September), and the E88 chain stopped at its next spec read.
rsync -az --delete \
  --filter 'P /Codes/configs/*/v*/' \
  --exclude '.git/' --exclude '.venv/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude 'outputs/' --exclude 'logs/' --exclude 'results/' \
  --exclude '/data/' --exclude '/datasets/' \
  ./ "${REMOTE}:${DEST}/"

echo "synced $(git rev-parse --short HEAD) (dirty=${DIRTY}) -> ${REMOTE}:${DEST}"
