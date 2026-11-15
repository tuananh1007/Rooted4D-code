"""Dataset discovery and local-layout validation."""

from .catalog import DatasetEntry, get_dataset, load_catalog
from .semisynthetic import (
    AuditReport,
    BatchLoader,
    SemisyntheticDataset,
    audit_semisynthetic_dataset,
    build_split_loaders,
    compare_semisynthetic_datasets,
    data_fingerprint,
    dataset_fingerprint,
)
from .split_lock import (
    SplitLockReport,
    load_split_lock,
    split_lock_fingerprint,
    validate_split_lock,
    validate_split_lock_file,
)
from .splits import find_split_leakage

__all__ = [
    "AuditReport",
    "BatchLoader",
    "DatasetEntry",
    "SemisyntheticDataset",
    "SplitLockReport",
    "audit_semisynthetic_dataset",
    "build_split_loaders",
    "compare_semisynthetic_datasets",
    "data_fingerprint",
    "dataset_fingerprint",
    "find_split_leakage",
    "get_dataset",
    "load_split_lock",
    "load_catalog",
    "split_lock_fingerprint",
    "validate_split_lock",
    "validate_split_lock_file",
]
