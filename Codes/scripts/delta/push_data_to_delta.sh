#!/usr/bin/env bash
# Push the datasets the experiments actually read to Delta's HDD tier.
#
# Separate from sync_to_delta.sh on purpose. That script runs on every commit and
# targets the NVMe code tier; this one runs rarely and targets the bulk tier,
# which is what the project's storage policy says and what the volumes require.
#
# Provider archives are not pushed, only the extracted trees the loaders open.
# M&Ms-2 is filed locally but no experiment reads it, so 9.3 GB of it stays put:
# uploading data nothing consumes is cost without benefit, and it would be the
# largest single item here.
#
# Redistribution note: this moves licensed provider data onto NCSA storage under
# the same agreements that govern the local copy. ACDC is CC BY-NC-SA, MVSeg2023
# CC BY-NC-ND, MITEA CC BY-NC-SA with provider approval. All are non-commercial
# research use by the same investigator; none is being redistributed to a third
# party. Nothing here may be made world-readable on the cluster.
set -euo pipefail

REMOTE="${1:-delta}"
REMOTE_USER="$(ssh -o BatchMode=yes "$REMOTE" 'echo $USER')"
HDD="/work/hdd/bhio/${REMOTE_USER}/Rooted4D"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"

ssh -o BatchMode=yes "$REMOTE" "mkdir -p ${HDD}/data/raw ${HDD}/data/derived && chmod -R go-rwx ${HDD}"

RAW=(acdc mitea mvseg2023)
DERIVED=(donors_v3 semisynthetic-real-v3 semisynthetic-real-v4)

for name in "${RAW[@]}"; do
  if [ -d "data/raw/${name}" ]; then
    echo "=== raw/${name} ($(du -sh "data/raw/${name}" | cut -f1))"
    rsync -a --info=progress2 "data/raw/${name}/" \
      "${REMOTE}:${HDD}/data/raw/${name}/"
  else
    echo "=== raw/${name}: absent locally, skipped"
  fi
done

for name in "${DERIVED[@]}"; do
  if [ -d "data/derived/${name}" ]; then
    echo "=== derived/${name} ($(du -sh "data/derived/${name}" | cut -f1))"
    rsync -a --info=progress2 "data/derived/${name}/" \
      "${REMOTE}:${HDD}/data/derived/${name}/"
  else
    echo "=== derived/${name}: absent locally, skipped"
  fi
done

# Keep provider data unreadable to anyone else on a shared filesystem.
ssh -o BatchMode=yes "$REMOTE" "chmod -R go-rwx ${HDD}/data"
echo "pushed to ${REMOTE}:${HDD}/data"
