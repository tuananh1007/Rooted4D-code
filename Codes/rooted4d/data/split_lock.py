"""Global identity split-lock validation and reusable-artifact provenance rules."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any


SCHEMA_VERSION = "rooted4d-global-split-lock-v1"
SPLITS = ("train", "val", "test")
TOP_LEVEL_FIELDS = {
    "schema_version",
    "lock_id",
    "status",
    "created_utc",
    "identity_hash",
    "split_policy",
    "records",
    "artifact_provenance",
}
RECORD_FIELDS = {
    "global_person_id",
    "source_dataset",
    "source_subject_id_hash",
    "split",
    "session_ids",
    "donor_ids",
    "modalities",
}
ARTIFACT_FIELDS = {
    "artifact_id",
    "artifact_type",
    "status",
    "fitted_on_splits",
    "selected_on_splits",
    "applied_to_splits",
    "source_record_ids",
    "sha256",
}
FORBIDDEN_IDENTIFIER_KEYS = {
    "name",
    "patient_name",
    "mrn",
    "medical_record_number",
    "date_of_birth",
    "dob",
    "accession_number",
    "accession",
    "raw_subject_id",
    "patient_id",
    "study_instance_uid",
    "series_instance_uid",
}
HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SplitLockReport:
    """Validation result for one global split lock."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    statistics: dict[str, Any]

    @property
    def passed(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "statistics": self.statistics,
        }


def load_split_lock(path: str | Path) -> dict[str, Any]:
    """Load a JSON split lock without silently accepting a non-object root."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Global split lock must contain a JSON object at its root")
    return payload


def split_lock_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return a stable digest of the complete lock payload."""

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(canonical.encode("utf-8")).hexdigest()


def _string_list(value: Any, path: str, errors: list[str]) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        errors.append(f"{path} must be a list of non-empty strings")
        return []
    if len(value) != len(set(value)):
        errors.append(f"{path} contains duplicates")
    return list(value)


def _check_forbidden_keys(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_IDENTIFIER_KEYS:
                errors.append(f"{path}.{key} is a forbidden raw-identifier field")
            _check_forbidden_keys(child, f"{path}.{key}", errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_forbidden_keys(child, f"{path}[{index}]", errors)


def _check_allowed_fields(
    value: Mapping[str, Any], allowed: set[str], path: str, errors: list[str]
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        errors.append(f"{path} contains unknown fields: {', '.join(unknown)}")


def _validate_identity_hash(payload: Any, errors: list[str]) -> None:
    if not isinstance(payload, Mapping):
        errors.append("identity_hash must be an object")
        return
    required = {"algorithm", "salt_id", "raw_identifiers_stored", "linkage_table_location"}
    _check_allowed_fields(payload, required, "identity_hash", errors)
    missing = sorted(required - set(payload))
    if missing:
        errors.append(f"identity_hash is missing fields: {', '.join(missing)}")
    if payload.get("algorithm") != "sha256":
        errors.append("identity_hash.algorithm must be sha256")
    if payload.get("raw_identifiers_stored") is not False:
        errors.append("identity_hash.raw_identifiers_stored must be false")
    if not isinstance(payload.get("salt_id"), str) or not payload.get("salt_id"):
        errors.append("identity_hash.salt_id must name the external linkage salt")
    if payload.get("linkage_table_location") != "outside_repository":
        errors.append("identity_hash.linkage_table_location must be outside_repository")


def _validate_policy(payload: Any, errors: list[str]) -> None:
    if not isinstance(payload, Mapping):
        errors.append("split_policy must be an object")
        return
    required = {
        "unit",
        "freeze_before_model_fitting",
        "pair_donors_only_within_split",
        "test_access",
    }
    _check_allowed_fields(payload, required, "split_policy", errors)
    missing = sorted(required - set(payload))
    if missing:
        errors.append(f"split_policy is missing fields: {', '.join(missing)}")
    if payload.get("unit") != "global_person":
        errors.append("split_policy.unit must be global_person")
    if payload.get("freeze_before_model_fitting") is not True:
        errors.append("split_policy.freeze_before_model_fitting must be true")
    if payload.get("pair_donors_only_within_split") is not True:
        errors.append("split_policy.pair_donors_only_within_split must be true")
    if payload.get("test_access") != "one_final_locked_evaluation":
        errors.append("split_policy.test_access must be one_final_locked_evaluation")


def _validate_records(
    records: Any, errors: list[str], warnings: list[str]
) -> tuple[dict[str, int], set[str]]:
    if not isinstance(records, list):
        errors.append("records must be a list")
        return {split: 0 for split in SPLITS}, set()

    counts = {split: 0 for split in SPLITS}
    person_ids: set[str] = set()
    source_keys: set[tuple[str, str]] = set()
    memberships: dict[str, dict[str, set[str]]] = {
        "session": defaultdict(set),
        "donor": defaultdict(set),
    }

    for index, record in enumerate(records):
        path = f"records[{index}]"
        if not isinstance(record, Mapping):
            errors.append(f"{path} must be an object")
            continue
        _check_allowed_fields(record, RECORD_FIELDS, path, errors)
        missing = sorted(RECORD_FIELDS - set(record))
        if missing:
            errors.append(f"{path} is missing fields: {', '.join(missing)}")

        person_id = record.get("global_person_id")
        if not isinstance(person_id, str) or not person_id:
            errors.append(f"{path}.global_person_id must be a non-empty pseudonym")
        elif person_id in person_ids:
            errors.append(f"{path}.global_person_id is duplicated: {person_id}")
        else:
            person_ids.add(person_id)

        source_dataset = record.get("source_dataset")
        source_hash = record.get("source_subject_id_hash")
        if not isinstance(source_dataset, str) or not source_dataset:
            errors.append(f"{path}.source_dataset must be a non-empty string")
        if not isinstance(source_hash, str) or not HASH_PATTERN.fullmatch(source_hash):
            errors.append(f"{path}.source_subject_id_hash must match sha256:<64 lowercase hex>")
        if isinstance(source_dataset, str) and isinstance(source_hash, str):
            source_key = (source_dataset, source_hash)
            if source_key in source_keys:
                errors.append(f"{path} duplicates a source subject record")
            source_keys.add(source_key)

        split = record.get("split")
        if split not in SPLITS:
            errors.append(f"{path}.split must be one of {', '.join(SPLITS)}")
            split = None
        else:
            counts[str(split)] += 1

        session_ids = _string_list(record.get("session_ids"), f"{path}.session_ids", errors)
        donor_ids = _string_list(record.get("donor_ids"), f"{path}.donor_ids", errors)
        modalities = _string_list(record.get("modalities"), f"{path}.modalities", errors)
        if not modalities:
            warnings.append(f"{path} has no modality labels")
        if split is not None:
            for session_id in session_ids:
                memberships["session"][session_id].add(str(split))
            for donor_id in donor_ids:
                memberships["donor"][donor_id].add(str(split))

    for kind, values in memberships.items():
        for identifier, splits in values.items():
            if len(splits) > 1:
                errors.append(
                    f"{kind} leakage: {identifier} occurs in {', '.join(sorted(splits))}"
                )
    if records and any(counts[split] == 0 for split in SPLITS):
        errors.append("A non-empty split lock must contain train, val, and test records")
    return counts, person_ids


def _validate_artifacts(
    artifacts: Any, person_ids: set[str], errors: list[str]
) -> dict[str, int]:
    if not isinstance(artifacts, list):
        errors.append("artifact_provenance must be a list")
        return {}
    counts: dict[str, int] = defaultdict(int)
    seen_ids: set[str] = set()
    valid_types = {"learned_reusable", "calibration", "fixed_external", "evaluation_only"}
    valid_statuses = {"planned", "frozen", "external"}
    for index, artifact in enumerate(artifacts):
        path = f"artifact_provenance[{index}]"
        if not isinstance(artifact, Mapping):
            errors.append(f"{path} must be an object")
            continue
        _check_allowed_fields(artifact, ARTIFACT_FIELDS, path, errors)
        required = ARTIFACT_FIELDS - {"sha256"}
        missing = sorted(required - set(artifact))
        if missing:
            errors.append(f"{path} is missing fields: {', '.join(missing)}")
        artifact_id = artifact.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            errors.append(f"{path}.artifact_id must be a non-empty string")
        elif artifact_id in seen_ids:
            errors.append(f"{path}.artifact_id is duplicated: {artifact_id}")
        else:
            seen_ids.add(artifact_id)
        artifact_type = artifact.get("artifact_type")
        if artifact_type not in valid_types:
            errors.append(f"{path}.artifact_type is invalid")
            artifact_type = None
        else:
            counts[str(artifact_type)] += 1
        status = artifact.get("status")
        if status not in valid_statuses:
            errors.append(f"{path}.status is invalid")
        digest = artifact.get("sha256")
        if status == "frozen" and (
            not isinstance(digest, str) or not HEX_PATTERN.fullmatch(digest)
        ):
            errors.append(f"{path}.sha256 is required for a frozen artifact")

        fitted = _string_list(
            artifact.get("fitted_on_splits"), f"{path}.fitted_on_splits", errors
        )
        selected = _string_list(
            artifact.get("selected_on_splits"), f"{path}.selected_on_splits", errors
        )
        applied = _string_list(
            artifact.get("applied_to_splits"), f"{path}.applied_to_splits", errors
        )
        sources = _string_list(
            artifact.get("source_record_ids"), f"{path}.source_record_ids", errors
        )
        for split in fitted + selected + applied:
            if split not in SPLITS:
                errors.append(f"{path} references an invalid split: {split}")
        if "test" in fitted or "test" in selected:
            errors.append(f"{path} fits or selects an artifact using the test split")
        if artifact_type == "learned_reusable":
            if fitted != ["train"]:
                errors.append(f"{path} learned_reusable artifacts must be fitted on train only")
            if any(split != "val" for split in selected):
                errors.append(f"{path} learned_reusable selection may use val only")
        elif artifact_type == "calibration":
            if fitted != ["val"]:
                errors.append(f"{path} calibration artifacts must be fitted on val only")
            if selected:
                errors.append(f"{path} calibration artifacts must not have selection splits")
        elif artifact_type in {"fixed_external", "evaluation_only"}:
            if fitted or selected:
                errors.append(f"{path} {artifact_type} artifacts cannot be fitted or selected")
        unknown_sources = sorted(set(sources) - person_ids)
        if unknown_sources:
            errors.append(f"{path} references unknown source records: {', '.join(unknown_sources)}")
    return dict(counts)


def validate_split_lock(payload: Mapping[str, Any]) -> SplitLockReport:
    """Validate identity isolation, provenance, privacy fields, and fit/selection policy."""

    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(payload, Mapping):
        return SplitLockReport(("split lock root must be an object",), (), {})
    _check_forbidden_keys(payload, "$", errors)
    _check_allowed_fields(payload, TOP_LEVEL_FIELDS, "$", errors)
    missing = sorted(TOP_LEVEL_FIELDS - set(payload))
    if missing:
        errors.append(f"split lock is missing fields: {', '.join(missing)}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("status") not in {"example_not_frozen", "frozen"}:
        errors.append("status must be example_not_frozen or frozen")
    elif payload.get("status") == "example_not_frozen":
        warnings.append("This is an example contract, not a frozen experimental split")
    if not isinstance(payload.get("lock_id"), str) or not payload.get("lock_id"):
        errors.append("lock_id must be a non-empty string")
    if not isinstance(payload.get("created_utc"), str) or not payload.get("created_utc"):
        errors.append("created_utc must be a non-empty ISO-8601 string")

    _validate_identity_hash(payload.get("identity_hash"), errors)
    _validate_policy(payload.get("split_policy"), errors)
    counts, person_ids = _validate_records(payload.get("records"), errors, warnings)
    artifact_counts = _validate_artifacts(
        payload.get("artifact_provenance"), person_ids, errors
    )
    statistics = {
        "status": payload.get("status"),
        "records_by_split": counts,
        "total_records": sum(counts.values()),
        "artifact_types": artifact_counts,
        "fingerprint": split_lock_fingerprint(payload),
    }
    return SplitLockReport(tuple(errors), tuple(warnings), statistics)


def validate_split_lock_file(path: str | Path) -> SplitLockReport:
    """Load and validate a split-lock file, returning parse errors as validation errors."""

    try:
        payload = load_split_lock(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return SplitLockReport((f"Could not load split lock: {exc}",), (), {})
    return validate_split_lock(payload)
