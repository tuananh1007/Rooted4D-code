"""Patient-level split leakage checks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence


def find_split_leakage(splits: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    """Return every patient identifier that appears in more than one split."""

    memberships: dict[str, list[str]] = defaultdict(list)
    for split_name, patient_ids in splits.items():
        seen_within_split: set[str] = set()
        for patient_id in patient_ids:
            identifier = str(patient_id)
            if identifier in seen_within_split:
                memberships[identifier].append(f"{split_name} (duplicate)")
            else:
                memberships[identifier].append(split_name)
                seen_within_split.add(identifier)
    return {
        patient_id: locations
        for patient_id, locations in memberships.items()
        if len(locations) > 1
    }

