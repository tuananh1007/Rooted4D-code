#!/usr/bin/env python3
"""Freeze the global split lock from a built benchmark.

Gate 0 requires an immutable donor and patient level split policy. Until now the
repository carried only the example contract, with `status:
example_not_frozen`, so nothing actually bound the partitions that results were
computed against. This writes the real one from a benchmark that already exists,
so the lock records the split that was used rather than proposing one.

Source subject identifiers are never written. Each record carries a salted
SHA-256 of the identifier instead, and the salt lives outside the repository:
without it the hashes cannot be reversed onto ACDC patient numbers or MVSeg2023
case names, and with it the pool stays auditable.

The donor to subject mapping is recovered from the fitting order, which is
`sorted()` over provider paths in both loaders and therefore deterministic. The
script refuses to run unless that mapping is one to one, because a loader that
dropped a case would silently shift every pseudonym after it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path


def _load_or_create_salt(path: Path) -> tuple[str, str]:
    if path.exists():
        payload = json.loads(path.read_text())
        return payload["salt_id"], payload["salt"]
    salt = secrets.token_hex(32)
    salt_id = "rooted4d-linkage-salt-" + hashlib.sha256(salt.encode()).hexdigest()[:12]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"salt_id": salt_id, "salt": salt}, indent=2))
    print(
        f"created a new linkage salt at {path}.\n"
        "  This file is the only way to re-link the pool to provider subject ids.\n"
        "  It is gitignored on purpose. Back it up outside the repository, or the\n"
        "  linkage is lost and the lock can never be re-derived."
    )
    return salt_id, salt


def _hash(salt: str, dataset: str, subject: str) -> str:
    digest = hashlib.sha256(f"{salt}|{dataset}|{subject}".encode()).hexdigest()
    return f"sha256:{digest}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/derived/semisynthetic-real-v2")
    parser.add_argument("--donors", default="data/derived/donors_v2/donors.json")
    parser.add_argument("--acdc-root", default="data/raw/acdc")
    parser.add_argument("--mvseg-root", default="data/raw/mvseg2023")
    parser.add_argument("--salt-file", default="data/linkage_salt.json")
    parser.add_argument("--output", default="GLOBAL_SPLIT_LOCK.json")
    parser.add_argument("--lock-id", default="rooted4d-semisynthetic-real-v2")
    arguments = parser.parse_args(argv)

    donors = json.loads(Path(arguments.donors).read_text())["donors"]
    hearts = [d for d in donors if d["kind"] == "whole_heart"]
    valves = [d for d in donors if d["kind"] == "valve"]

    patients = sorted({p.parent.name for p in Path(arguments.acdc_root).rglob("*_gt.nii.gz")})
    cases = sorted(
        p.name.replace("-label.nii.gz", "")
        for p in Path(arguments.mvseg_root).rglob("*-label.nii.gz")
    )
    if len(patients) != len(hearts) or len(cases) != len(valves):
        raise SystemExit(
            "donor to subject mapping is not one to one "
            f"({len(patients)} patients / {len(hearts)} heart donors, "
            f"{len(cases)} cases / {len(valves)} valve donors); a loader dropped "
            "something and every pseudonym after it would be misattributed"
        )

    dataset = json.loads((Path(arguments.benchmark) / "dataset.json").read_text())
    partitions = dataset["donor_partitions"]
    donor_split = {
        donor_id: split
        for split, kinds in partitions.items()
        for donor_ids in kinds.values()
        for donor_id in donor_ids
    }

    salt_id, salt = _load_or_create_salt(Path(arguments.salt_file))
    records = []
    for donor, subject, source, modality in (
        [(d, s, "ACDC", "cmr") for d, s in zip(hearts, patients)]
        + [(d, s, "MVSeg2023", "tee") for d, s in zip(valves, cases)]
    ):
        split = donor_split.get(donor["donor_id"])
        if split is None:
            # An unused donor is not a policy violation, but it must not be
            # written as though it had been assigned to a split.
            continue
        records.append(
            {
                "global_person_id": donor["global_person_id"],
                "source_dataset": source,
                "source_subject_id_hash": _hash(salt, source, subject),
                "split": split,
                "session_ids": [f"{donor['donor_id']}:{modality}"],
                "donor_ids": [donor["donor_id"]],
                "modalities": [modality],
            }
        )

    lock = {
        "schema_version": "rooted4d-global-split-lock-v1",
        "lock_id": arguments.lock_id,
        "status": "frozen",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "identity_hash": {
            "algorithm": "sha256",
            "salt_id": salt_id,
            "raw_identifiers_stored": False,
            "linkage_table_location": "outside_repository",
        },
        "split_policy": {
            "unit": "global_person",
            "freeze_before_model_fitting": True,
            "pair_donors_only_within_split": True,
            "test_access": "one_final_locked_evaluation",
        },
        "records": records,
        "artifact_provenance": [],
    }

    from rooted4d.data.split_lock import validate_split_lock

    report = validate_split_lock(lock)
    if not report.passed:
        raise SystemExit("refusing to write an invalid lock:\n- " + "\n- ".join(report.errors))

    Path(arguments.output).write_text(json.dumps(lock, indent=2) + "\n")
    counts: dict[str, int] = {}
    for record in records:
        counts[record["split"]] = counts.get(record["split"], 0) + 1
    print(f"wrote {arguments.output}: {len(records)} people, {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
