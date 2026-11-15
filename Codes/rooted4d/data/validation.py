"""Conservative local dataset-layout checks.

These checks confirm only that expected file patterns exist. They do not verify
license compliance, annotation correctness, or patient-level split integrity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .catalog import DatasetEntry


@dataclass(frozen=True)
class ValidationReport:
    dataset: str
    root: str
    applicable: bool
    passed: bool
    matched_files: int
    minimum_matches: int
    patterns: tuple[str, ...]
    message: str

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "root": self.root,
            "applicable": self.applicable,
            "passed": self.passed,
            "matched_files": self.matched_files,
            "minimum_matches": self.minimum_matches,
            "patterns": list(self.patterns),
            "message": self.message,
        }


def validate_dataset(entry: DatasetEntry, root: str | Path) -> ValidationReport:
    data_root = Path(root).expanduser().resolve()
    if not entry.validation_globs:
        return ValidationReport(
            dataset=entry.key,
            root=str(data_root),
            applicable=False,
            passed=False,
            matched_files=0,
            minimum_matches=entry.minimum_matches,
            patterns=entry.validation_globs,
            message="No local validation pattern is defined because this is not an available dataset.",
        )
    if not data_root.exists():
        return ValidationReport(
            dataset=entry.key,
            root=str(data_root),
            applicable=True,
            passed=False,
            matched_files=0,
            minimum_matches=entry.minimum_matches,
            patterns=entry.validation_globs,
            message="Dataset root does not exist.",
        )
    matched: set[Path] = set()
    for pattern in entry.validation_globs:
        matched.update(path.resolve() for path in data_root.glob(pattern) if path.is_file())
    passed = len(matched) >= entry.minimum_matches
    message = (
        "Layout check passed."
        if passed
        else f"Found {len(matched)} matching files; expected at least {entry.minimum_matches}."
    )
    return ValidationReport(
        dataset=entry.key,
        root=str(data_root),
        applicable=True,
        passed=passed,
        matched_files=len(matched),
        minimum_matches=entry.minimum_matches,
        patterns=entry.validation_globs,
        message=message,
    )

