"""Stable seed derivation and hashing for generated research artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import yaml

from rooted4d import __version__


MAX_SEED = 2**63 - 1


def validate_seed(value: int, name: str = "seed") -> int:
    """Return a validated non-negative integer seed."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= MAX_SEED:
        raise ValueError(f"{name} must lie in [0, {MAX_SEED}]")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically for fingerprints and seed paths."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def derive_seed(root_seed: int, *path: object) -> int:
    """Derive an order-independent NumPy seed from a named hierarchy path.

    Named paths avoid the common reproducibility bug where inserting one random
    draw changes every downstream result. NumPy ``SeedSequence`` remains the
    entropy mixer; SHA-256 converts arbitrary stable labels into its spawn key.
    """

    root_seed = validate_seed(root_seed, "root_seed")
    digest = hashlib.sha256(canonical_json_bytes(list(path))).digest()
    spawn_key = tuple(int(value) for value in np.frombuffer(digest[:16], dtype="<u4"))
    sequence = np.random.SeedSequence(entropy=root_seed, spawn_key=spawn_key)
    return int(sequence.generate_state(1, dtype=np.uint64)[0] & np.uint64(MAX_SEED))


def make_rng(root_seed: int, *path: object) -> np.random.Generator:
    return np.random.default_rng(derive_seed(root_seed, *path))


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def environment_record() -> dict[str, str]:
    """Return versions that can affect generated numerical bytes."""

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pyyaml": yaml.__version__,
        "rooted4d": __version__,
    }


def generator_source_fingerprint() -> str:
    """Fingerprint the source files that determine generated dataset bytes."""

    package_root = Path(__file__).parents[1]
    relative_paths = (
        "__init__.py",
        "constraints/attachment.py",
        "geometry/rigid.py",
        "geometry/toy.py",
        "geometry/template.py",
        "motion/fourier.py",
        "motion/phase.py",
        "observation/cmr.py",
        "observation/distance.py",
        "observation/tee.py",
        "synthetic/benchmark.py",
        "synthetic/config.py",
        "synthetic/dataset_builder.py",
        "synthetic/donors.py",
        "synthetic/reproducibility.py",
    )
    records = [
        {"path": relative, "sha256": sha256_file(package_root / relative)}
        for relative in relative_paths
    ]
    return canonical_sha256(records)
