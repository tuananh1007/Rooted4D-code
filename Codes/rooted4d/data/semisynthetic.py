"""Audit and manifest-only loading for Rooted4D semi-synthetic datasets."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from rooted4d.synthetic.config import SPLIT_NAMES
from rooted4d.synthetic.reproducibility import (
    canonical_sha256,
    derive_seed,
    generator_source_fingerprint,
    sha256_file,
    validate_seed,
)


DATASET_SCHEMA = "rooted4d-semisynthetic-dataset-v2"
SAMPLE_SCHEMA = "rooted4d-semisynthetic-sample-v2"


@dataclass(frozen=True)
class AuditReport:
    passed: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    statistics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "statistics": self.statistics,
        }

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise ValueError("semi-synthetic dataset audit failed:\n- " + "\n- ".join(self.errors))


def _safe_member(root: Path, relative: object) -> Path:
    value = Path(str(relative))
    if value.is_absolute():
        raise ValueError(f"manifest path must be relative: {value}")
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"manifest path escapes dataset root: {value}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path.name}:{line_number} is not a JSON object")
        records.append(value)
    return records


def _stored_scalar(arrays: Any, key: str) -> Any:
    if key not in arrays.files:
        raise KeyError(f"sample array is missing {key!r}")
    value = arrays[key]
    if value.shape != ():
        raise ValueError(f"stored field {key!r} must be scalar")
    return value.item()


def _pairwise_leaks(values: dict[str, set[str]], label: str) -> list[str]:
    errors = []
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            overlap = sorted(values[left] & values[right])
            if overlap:
                preview = ", ".join(overlap[:5])
                errors.append(f"{label} leakage between {left} and {right}: {preview}")
    return errors


def audit_semisynthetic_dataset(
    root: str | Path,
    *,
    verify_checksums: bool = True,
    verify_arrays: bool = True,
) -> AuditReport:
    """Validate manifests, split isolation, file integrity, and stored identities."""

    dataset_root = Path(root).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    statistics: dict[str, Any] = {"splits": {}}
    dataset_path = dataset_root / "dataset.json"
    if not dataset_path.is_file():
        return AuditReport(False, ("dataset.json is missing",), (), statistics)
    try:
        dataset = _read_json(dataset_path)
    except Exception as error:
        return AuditReport(False, (f"cannot parse dataset.json: {error}",), (), statistics)
    if dataset.get("schema_version") != DATASET_SCHEMA:
        errors.append(f"unsupported dataset schema: {dataset.get('schema_version')!r}")
    global_lock = dataset.get("global_split_lock", {})
    if global_lock.get("status") == "frozen":
        try:
            lock_path = _safe_member(dataset_root, global_lock["path"])
            if not lock_path.is_file():
                errors.append("frozen global split-lock snapshot is missing")
            elif verify_checksums and sha256_file(lock_path) != global_lock.get("sha256"):
                errors.append("frozen global split-lock snapshot checksum mismatch")
        except Exception as error:
            errors.append(f"invalid frozen global split-lock descriptor: {error}")
    elif global_lock.get("status") != "generated_from_split_seed":
        errors.append("global split-lock status is missing or invalid")
    recorded_source = dataset.get("reproducibility", {}).get("generator_source_sha256")
    if not isinstance(recorded_source, str) or len(recorded_source) != 64:
        errors.append("generator source fingerprint is missing or invalid")
    elif recorded_source != generator_source_fingerprint():
        warnings.append(
            "installed generator source differs from the source that created this dataset"
        )

    for descriptor_name, path_key, hash_key in (
        ("config", "path", "sha256"),
        ("seed manifest", "seed_manifest", "seed_manifest_sha256"),
    ):
        container = dataset.get("config", {}) if descriptor_name == "config" else dataset.get(
            "reproducibility", {}
        )
        try:
            path = _safe_member(dataset_root, container[path_key])
            if not path.is_file():
                errors.append(f"{descriptor_name} file is missing")
            elif verify_checksums and sha256_file(path) != container.get(hash_key):
                errors.append(f"{descriptor_name} checksum mismatch")
        except Exception as error:
            errors.append(f"invalid {descriptor_name} descriptor: {error}")

    declared_partitions: dict[str, dict[str, set[str]]] = {
        field: {split: set() for split in SPLIT_NAMES}
        for field in ("whole_heart", "valve", "whole_heart_people", "valve_people")
    }
    raw_partitions = dataset.get("donor_partitions", {})
    for split in SPLIT_NAMES:
        split_partitions = raw_partitions.get(split, {})
        for field in declared_partitions:
            values = split_partitions.get(field, [])
            if not isinstance(values, list):
                errors.append(f"declared {field} partition for {split} is not a list")
                continue
            declared_partitions[field][split] = {str(value) for value in values}
    for field, values in declared_partitions.items():
        errors.extend(_pairwise_leaks(values, f"declared {field} donor"))

    observed: dict[str, dict[str, set[str]]] = {
        "sample_id": {split: set() for split in SPLIT_NAMES},
        "identity_id": {split: set() for split in SPLIT_NAMES},
        "whole_heart_donor_id": {split: set() for split in SPLIT_NAMES},
        "valve_donor_id": {split: set() for split in SPLIT_NAMES},
        "whole_heart_global_person_id": {split: set() for split in SPLIT_NAMES},
        "valve_global_person_id": {split: set() for split in SPLIT_NAMES},
        "source_global_person_id": {split: set() for split in SPLIT_NAMES},
    }
    listed_paths: set[str] = set()
    split_descriptors = dataset.get("splits", {})
    for split in SPLIT_NAMES:
        descriptor = split_descriptors.get(split)
        if not isinstance(descriptor, dict):
            errors.append(f"missing split descriptor for {split}")
            continue
        try:
            manifest_path = _safe_member(dataset_root, descriptor["manifest"])
            if verify_checksums and sha256_file(manifest_path) != descriptor.get("manifest_sha256"):
                errors.append(f"{split} manifest checksum mismatch")
            entries = _read_jsonl(manifest_path)
        except Exception as error:
            errors.append(f"cannot load {split} manifest: {error}")
            continue
        if not entries:
            errors.append(f"{split} split is empty")
        if len(entries) != descriptor.get("sample_count"):
            errors.append(f"{split} sample count does not match dataset.json")

        for entry in entries:
            sample_id = str(entry.get("sample_id", ""))
            identity_id = str(entry.get("identity_id", ""))
            heart_id = str(entry.get("whole_heart_donor_id", ""))
            valve_id = str(entry.get("valve_donor_id", ""))
            heart_person_id = str(entry.get("whole_heart_global_person_id", ""))
            valve_person_id = str(entry.get("valve_global_person_id", ""))
            if entry.get("schema_version") != SAMPLE_SCHEMA:
                errors.append(f"sample {sample_id or '<missing>'} has an unsupported schema")
            if entry.get("split") != split:
                errors.append(f"sample {sample_id or '<missing>'} is labeled for the wrong split")
            for field, value in (
                ("sample_id", sample_id),
                ("identity_id", identity_id),
                ("whole_heart_donor_id", heart_id),
                ("valve_donor_id", valve_id),
                ("whole_heart_global_person_id", heart_person_id),
                ("valve_global_person_id", valve_person_id),
            ):
                if not value:
                    errors.append(f"entry in {split} has an empty {field}")
                if field == "sample_id" and value in observed[field][split]:
                    errors.append(f"duplicate sample ID within {split}: {value}")
                observed[field][split].add(value)
            observed["source_global_person_id"][split].update(
                (heart_person_id, valve_person_id)
            )
            if heart_id not in declared_partitions["whole_heart"][split]:
                errors.append(f"sample {sample_id} uses a heart donor outside {split}")
            if valve_id not in declared_partitions["valve"][split]:
                errors.append(f"sample {sample_id} uses a valve donor outside {split}")
            if heart_person_id not in declared_partitions["whole_heart_people"][split]:
                errors.append(f"sample {sample_id} uses a heart source person outside {split}")
            if valve_person_id not in declared_partitions["valve_people"][split]:
                errors.append(f"sample {sample_id} uses a valve source person outside {split}")

            relative = str(entry.get("relative_path", ""))
            expected_prefix = f"samples/{split}/"
            if not relative.startswith(expected_prefix):
                errors.append(f"sample {sample_id} path is outside its split directory")
            if relative in listed_paths:
                errors.append(f"multiple manifest entries refer to {relative}")
            listed_paths.add(relative)
            try:
                sample_path = _safe_member(dataset_root, relative)
            except Exception as error:
                errors.append(f"invalid path for sample {sample_id}: {error}")
                continue
            if not sample_path.is_file():
                errors.append(f"sample file is missing: {relative}")
                continue
            if verify_checksums and sha256_file(sample_path) != entry.get("sha256"):
                errors.append(f"sample checksum mismatch: {sample_id}")
            if verify_arrays:
                try:
                    with np.load(sample_path, allow_pickle=False) as arrays:
                        expected_scalars = {
                            "sample_id": sample_id,
                            "split": split,
                            "identity_id": identity_id,
                            "whole_heart_donor_id": heart_id,
                            "valve_donor_id": valve_id,
                            "whole_heart_global_person_id": heart_person_id,
                            "valve_global_person_id": valve_person_id,
                            "variant_index": int(entry.get("variant_index")),
                        }
                        for key, expected in expected_scalars.items():
                            if _stored_scalar(arrays, key) != expected:
                                errors.append(f"stored {key} mismatch for sample {sample_id}")
                        for name, seed in entry.get("seeds", {}).items():
                            if int(_stored_scalar(arrays, f"seed__{name}")) != int(seed):
                                errors.append(f"stored seed {name} mismatch for sample {sample_id}")
                        attachment = float(_stored_scalar(arrays, "attachment_residual_mm"))
                        seam = float(_stored_scalar(arrays, "periodic_seam_mm"))
                        increment = float(_stored_scalar(arrays, "phase_minimum_increment"))
                        if attachment > 1e-10:
                            errors.append(f"attachment invariant failed for sample {sample_id}")
                        if seam > 1e-10:
                            errors.append(f"periodic invariant failed for sample {sample_id}")
                        if increment <= 0.0:
                            errors.append(f"phase monotonicity failed for sample {sample_id}")
                        domain = entry.get("observation_domain")
                        if domain not in {"surface_points", "image_likelihood"}:
                            errors.append(f"invalid observation domain for sample {sample_id}")
                        if domain == "image_likelihood":
                            required_images = (
                                "cmr_surface_likelihood",
                                "cmr_plane_affine_lps",
                                "cmr_slice_offsets_mm",
                                "tee_surface_likelihood",
                                "tee_grid_apex",
                                "tee_grid_axis",
                                "tee_grid_depths_mm",
                                "tee_grid_azimuths_rad",
                                "tee_grid_elevations_rad",
                            )
                            missing_images = [
                                name for name in required_images if name not in arrays.files
                            ]
                            if missing_images:
                                errors.append(
                                    f"image-domain sample {sample_id} is missing "
                                    + ", ".join(missing_images)
                                )
                            else:
                                cmr_image = np.asarray(arrays["cmr_surface_likelihood"])
                                tee_image = np.asarray(arrays["tee_surface_likelihood"])
                                if cmr_image.ndim != 4 or cmr_image.shape[0] != len(
                                    arrays["cmr_times"]
                                ):
                                    errors.append(f"invalid CMR image shape for sample {sample_id}")
                                if tee_image.ndim != 4 or tee_image.shape[0] != len(
                                    arrays["tee_times"]
                                ):
                                    errors.append(f"invalid TEE image shape for sample {sample_id}")
                                for name, image in (("CMR", cmr_image), ("TEE", tee_image)):
                                    if not np.all(np.isfinite(image)):
                                        errors.append(
                                            f"{name} image has non-finite values for sample {sample_id}"
                                        )
                                    elif np.min(image) < 0.0 or np.max(image) > 1.0:
                                        errors.append(
                                            f"{name} image lies outside [0, 1] for sample {sample_id}"
                                        )
                except Exception as error:
                    errors.append(f"cannot validate arrays for sample {sample_id}: {error}")

        identity_count = len(observed["identity_id"][split])
        if identity_count != descriptor.get("identity_count"):
            errors.append(f"{split} identity count does not match dataset.json")
        statistics["splits"][split] = {
            "samples": len(entries),
            "identities": identity_count,
            "whole_heart_donors": len(observed["whole_heart_donor_id"][split]),
            "valve_donors": len(observed["valve_donor_id"][split]),
            "source_people": len(observed["source_global_person_id"][split]),
        }

    for field, values in observed.items():
        errors.extend(_pairwise_leaks(values, field))

    if dataset_root.joinpath("samples").exists():
        actual_paths = {
            path.relative_to(dataset_root).as_posix()
            for path in dataset_root.joinpath("samples").rglob("*.npz")
            if path.is_file()
        }
        extras = sorted(actual_paths - listed_paths)
        missing_from_disk = sorted(listed_paths - actual_paths)
        if extras:
            errors.append(f"unlisted sample files found: {', '.join(extras[:5])}")
        if missing_from_disk:
            errors.append(f"listed sample files missing: {', '.join(missing_from_disk[:5])}")
    else:
        errors.append("samples directory is missing")

    statistics["total_samples"] = sum(
        values.get("samples", 0) for values in statistics["splits"].values()
    )
    statistics["checksums_verified"] = verify_checksums
    statistics["arrays_verified"] = verify_arrays
    return AuditReport(not errors, tuple(errors), tuple(warnings), statistics)


def dataset_fingerprint(root: str | Path) -> str:
    """Hash all reproducibility-relevant files, excluding the derived audit report."""

    dataset_root = Path(root).expanduser().resolve()
    records = []
    for path in sorted(value for value in dataset_root.rglob("*") if value.is_file()):
        relative = path.relative_to(dataset_root).as_posix()
        if relative == "audit.json":
            continue
        records.append({"path": relative, "sha256": sha256_file(path)})
    return canonical_sha256(records)


def data_fingerprint(root: str | Path) -> str:
    """Hash only the generated arrays, excluding manifest and audit files.

    ``dataset_fingerprint`` covers the manifest too, which records the package
    version and a source hash derived from it. That makes it provenance
    sensitive by design: bumping ``__version__`` changes it even when every
    generated array is byte identical. This second hash answers the narrower and
    more often useful question -- did the data itself change? -- so a version
    bump and a genuine regeneration can be told apart.
    """

    dataset_root = Path(root).expanduser().resolve()
    excluded = {"audit.json", "dataset.json"}
    records = []
    for path in sorted(value for value in dataset_root.rglob("*") if value.is_file()):
        relative = path.relative_to(dataset_root).as_posix()
        if relative in excluded or relative.endswith("manifest.json"):
            continue
        records.append({"path": relative, "sha256": sha256_file(path)})
    return canonical_sha256(records)


def compare_semisynthetic_datasets(left: str | Path, right: str | Path) -> bool:
    return dataset_fingerprint(left) == dataset_fingerprint(right)


def compare_semisynthetic_data(left: str | Path, right: str | Path) -> bool:
    """Whether two datasets hold identical arrays, ignoring provenance metadata."""

    return data_fingerprint(left) == data_fingerprint(right)


class SemisyntheticDataset(Sequence[dict[str, Any]]):
    """Load exactly one split from its signed manifest; filesystem globs are never used."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        verify_checksums: bool = True,
        strict: bool = True,
    ) -> None:
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.verify_checksums = verify_checksums
        if strict:
            report = audit_semisynthetic_dataset(
                self.root,
                verify_checksums=verify_checksums,
                verify_arrays=False,
            )
            report.raise_if_failed()
        self.dataset_manifest = _read_json(self.root / "dataset.json")
        descriptor = self.dataset_manifest["splits"][split]
        manifest_path = _safe_member(self.root, descriptor["manifest"])
        self.entries = _read_jsonl(manifest_path)
        if not self.entries:
            raise ValueError(f"requested split {split!r} is empty")
        if any(entry.get("split") != split for entry in self.entries):
            raise ValueError(f"manifest for {split!r} contains another split")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        path = _safe_member(self.root, entry["relative_path"])
        if self.verify_checksums and sha256_file(path) != entry["sha256"]:
            raise ValueError(f"checksum mismatch for {entry['sample_id']}")
        with np.load(path, allow_pickle=False) as stored:
            arrays = {name: stored[name].copy() for name in stored.files}
        checks = {
            "sample_id": entry["sample_id"],
            "split": self.split,
            "identity_id": entry["identity_id"],
            "whole_heart_donor_id": entry["whole_heart_donor_id"],
            "valve_donor_id": entry["valve_donor_id"],
            "whole_heart_global_person_id": entry["whole_heart_global_person_id"],
            "valve_global_person_id": entry["valve_global_person_id"],
        }
        for key, expected in checks.items():
            if arrays[key].shape != () or arrays[key].item() != expected:
                raise ValueError(f"stored {key} mismatch for {entry['sample_id']}")
        return {"arrays": arrays, "metadata": entry}


class BatchLoader:
    """Small deterministic batching layer; use the optional Torch loader later if needed."""

    def __init__(
        self,
        dataset: SemisyntheticDataset,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = bool(shuffle)
        self.seed = validate_seed(seed, "loader seed")
        self.drop_last = bool(drop_last)
        self.epoch = 0

    @property
    def split(self) -> str:
        return self.dataset.split

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        indices = np.arange(len(self.dataset), dtype=np.int64)
        if self.shuffle:
            epoch_seed = derive_seed(self.seed, "loader", self.split, self.epoch)
            indices = np.random.default_rng(epoch_seed).permutation(indices)
        for start in range(0, len(indices), self.batch_size):
            selected = indices[start : start + self.batch_size]
            if self.drop_last and len(selected) < self.batch_size:
                break
            samples = [self.dataset[int(index)] for index in selected]
            keys = samples[0]["arrays"].keys()
            arrays = {
                key: np.stack([sample["arrays"][key] for sample in samples], axis=0)
                for key in keys
            }
            yield {
                "split": self.split,
                "arrays": arrays,
                "metadata": [sample["metadata"] for sample in samples],
            }


def build_split_loaders(
    root: str | Path,
    batch_size: int,
    *,
    verify_checksums: bool = True,
    train_shuffle: bool = True,
    seed: int | None = None,
) -> dict[str, BatchLoader]:
    """Create isolated train/val/test loaders after one full leakage audit."""

    report = audit_semisynthetic_dataset(
        root,
        verify_checksums=verify_checksums,
        verify_arrays=True,
    )
    report.raise_if_failed()
    dataset_manifest = _read_json(Path(root).expanduser().resolve() / "dataset.json")
    loader_seed = (
        int(dataset_manifest["reproducibility"]["root_seed"])
        if seed is None
        else validate_seed(seed, "loader seed")
    )
    datasets = {
        split: SemisyntheticDataset(
            root,
            split,
            verify_checksums=False,
            strict=False,
        )
        for split in SPLIT_NAMES
    }
    return {
        split: BatchLoader(
            datasets[split],
            batch_size,
            shuffle=(train_shuffle if split == "train" else False),
            seed=derive_seed(loader_seed, "split-loader", split),
        )
        for split in SPLIT_NAMES
    }
