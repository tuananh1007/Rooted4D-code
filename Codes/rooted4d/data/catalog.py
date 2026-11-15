"""Load the audited Rooted4D dataset catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class DatasetEntry:
    """One provider resource and its permitted acquisition route."""

    key: str
    title: str
    modality: str
    role: str
    access: str
    status: str
    landing_url: str | None = None
    access_url: str | None = None
    license: str = "Unspecified"
    fetch_kind: str = "manual"
    command: str | None = None
    repo_id: str | None = None
    limitations: str = ""
    validation_globs: tuple[str, ...] = field(default_factory=tuple)
    minimum_matches: int = 0

    @classmethod
    def from_mapping(cls, key: str, value: Mapping[str, Any]) -> "DatasetEntry":
        known = {
            "title",
            "modality",
            "role",
            "access",
            "status",
            "landing_url",
            "access_url",
            "license",
            "fetch_kind",
            "command",
            "repo_id",
            "limitations",
            "validation_globs",
            "minimum_matches",
        }
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"Unknown catalog fields for {key}: {sorted(unknown)}")
        payload = dict(value)
        payload["validation_globs"] = tuple(payload.get("validation_globs", ()))
        return cls(key=key, **payload)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "modality": self.modality,
            "role": self.role,
            "access": self.access,
            "status": self.status,
            "landing_url": self.landing_url,
            "access_url": self.access_url,
            "license": self.license,
            "fetch_kind": self.fetch_kind,
            "command": self.command,
            "repo_id": self.repo_id,
            "limitations": self.limitations,
            "validation_globs": list(self.validation_globs),
            "minimum_matches": self.minimum_matches,
        }


def _default_catalog_text() -> str:
    return resources.files("rooted4d.data").joinpath("catalog.yaml").read_text(encoding="utf-8")


def load_catalog(path: str | Path | None = None) -> dict[str, DatasetEntry]:
    """Return catalog entries keyed by stable short names."""

    if path is None:
        raw = yaml.safe_load(_default_catalog_text())
    else:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("datasets"), dict):
        raise ValueError("Catalog must contain a top-level 'datasets' mapping")
    return {
        key: DatasetEntry.from_mapping(key, value)
        for key, value in raw["datasets"].items()
    }


def get_dataset(key: str, path: str | Path | None = None) -> DatasetEntry:
    catalog = load_catalog(path)
    try:
        return catalog[key]
    except KeyError as exc:
        choices = ", ".join(sorted(catalog))
        raise KeyError(f"Unknown dataset '{key}'. Available: {choices}") from exc

