"""Fixed-topology donor adapters for real-derived and procedural geometry."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from rooted4d.geometry import HeterogeneousTemplate, make_toy_complex

from .reproducibility import make_rng, sha256_file


DONOR_SCHEMA = "rooted4d-fixed-topology-donors-v2"


@dataclass(frozen=True)
class Donor:
    donor_id: str
    global_person_id: str
    kind: str
    vertices: np.ndarray
    source_dataset: str
    source_sha256: str | None

    def public_record(self) -> dict[str, str | None]:
        return {
            "donor_id": self.donor_id,
            "global_person_id": self.global_person_id,
            "kind": self.kind,
            "source_dataset": self.source_dataset,
            "source_sha256": self.source_sha256,
        }


def _check_vertices(vertices: np.ndarray, expected_shape: tuple[int, int], donor_id: str) -> None:
    if vertices.shape != expected_shape:
        raise ValueError(
            f"donor {donor_id!r} has shape {vertices.shape}; expected {expected_shape}"
        )
    if not np.all(np.isfinite(vertices)):
        raise ValueError(f"donor {donor_id!r} contains non-finite coordinates")


def _procedural_heart_vertices(
    donor_id: str,
    root_seed: int,
    n_annulus: int,
) -> np.ndarray:
    toy = make_toy_complex(n_annulus=n_annulus)
    rng = make_rng(root_seed, "donor", "whole_heart", donor_id)
    vertices = toy.global_vertices.copy()
    scale = rng.uniform([0.88, 0.88, 0.90], [1.14, 1.14, 1.12])
    vertices *= scale

    annulus = toy.groups["annulus"]
    theta = np.linspace(0.0, 2.0 * np.pi, len(annulus), endpoint=False)
    radial = rng.normal(0.0, 0.035)
    vertices[annulus, 0] *= 1.0 + radial * np.cos(theta)
    vertices[annulus, 1] *= 1.0 - radial * np.cos(theta)
    vertices[annulus, 2] += rng.uniform(-0.8, 0.8) * np.cos(2.0 * theta)

    chamber_unique = toy.groups["chamber_unique"]
    vertices[chamber_unique, 0] += rng.normal(0.0, 0.8)
    vertices[chamber_unique, 1] += rng.normal(0.0, 0.8)
    vertices[chamber_unique, 2] *= rng.uniform(0.94, 1.06)
    return vertices


def _procedural_valve_vertices(
    donor_id: str,
    root_seed: int,
    n_annulus: int,
) -> np.ndarray:
    toy = make_toy_complex(n_annulus=n_annulus)
    rng = make_rng(root_seed, "donor", "valve", donor_id)
    vertices = toy.global_vertices.copy()
    annulus = toy.groups["annulus"]
    half = n_annulus // 2
    anterior_root = annulus[:half]
    posterior_root = annulus[half:]
    leaflet_free = toy.groups["leaflet_free"]
    anterior_free = leaflet_free[:half]
    posterior_free = leaflet_free[half:]

    annular_scale = rng.uniform([0.88, 0.88, 0.90], [1.12, 1.12, 1.10])
    vertices[annulus] *= annular_scale
    anterior_displacement = vertices[anterior_free] - vertices[anterior_root]
    posterior_displacement = vertices[posterior_free] - vertices[posterior_root]
    anterior_displacement[:, :2] *= rng.uniform(0.82, 1.18)
    posterior_displacement[:, :2] *= rng.uniform(0.82, 1.18)
    anterior_displacement[:, 2] *= rng.uniform(0.78, 1.22)
    posterior_displacement[:, 2] *= rng.uniform(0.78, 1.22)
    vertices[anterior_free] = vertices[anterior_root] + anterior_displacement
    vertices[posterior_free] = vertices[posterior_root] + posterior_displacement
    return vertices


def make_procedural_donors(
    root_seed: int,
    whole_heart_count: int,
    valve_count: int,
    n_annulus: int,
) -> tuple[list[Donor], list[Donor]]:
    hearts = []
    valves = []
    for index in range(whole_heart_count):
        donor_id = f"proc-heart-{index:04d}"
        hearts.append(
            Donor(
                donor_id=donor_id,
                global_person_id=f"synthetic-person-{donor_id}",
                kind="whole_heart",
                vertices=_procedural_heart_vertices(donor_id, root_seed, n_annulus),
                source_dataset="procedural",
                source_sha256=None,
            )
        )
    for index in range(valve_count):
        donor_id = f"proc-valve-{index:04d}"
        valves.append(
            Donor(
                donor_id=donor_id,
                global_person_id=f"synthetic-person-{donor_id}",
                kind="valve",
                vertices=_procedural_valve_vertices(donor_id, root_seed, n_annulus),
                source_dataset="procedural",
                source_sha256=None,
            )
        )
    return hearts, valves


def load_donor_manifest(
    manifest_path: str | Path,
    n_annulus: int | None = None,
    *,
    template: HeterogeneousTemplate | None = None,
    chamber_levels: int = 1,
    leaflet_rows: int = 1,
) -> tuple[list[Donor], list[Donor]]:
    """Load donors already registered to the frozen Rooted4D topology.

    The adapter intentionally does not accept raw segmentations. Real inputs must
    first be fitted to the frozen topology, preventing per-case remeshing from
    silently changing correspondence.
    """

    manifest = Path(manifest_path).resolve()
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    if raw.get("schema_version") != DONOR_SCHEMA:
        raise ValueError(f"unsupported donor manifest schema: {raw.get('schema_version')!r}")
    expected_kind = "frozen_production_v1" if template is not None else "procedural_test_v1"
    if raw.get("topology_kind") != expected_kind:
        raise ValueError(f"donor manifest topology_kind must be {expected_kind}")
    if template is not None and raw.get("template_id") != template.template_id:
        raise ValueError("donor manifest template_id does not match the frozen template")
    records = raw.get("donors")
    if not isinstance(records, list):
        raise TypeError("donor manifest 'donors' must be a list")

    if template is None:
        if n_annulus is None:
            raise ValueError("n_annulus is required for the procedural test topology")
        expected_shape = make_toy_complex(n_annulus=n_annulus, chamber_levels=chamber_levels,
                                          leaflet_rows=leaflet_rows).global_vertices.shape
        if (int(raw.get("chamber_levels", 1)), int(raw.get("leaflet_rows", 1))) != (chamber_levels, leaflet_rows):
            raise ValueError("the donor manifest was written for another procedural topology")
    else:
        expected_shape = template.vertices.shape
    seen: set[str] = set()
    grouped: dict[str, list[Donor]] = {"whole_heart": [], "valve": []}
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("each donor record must be a mapping")
        donor_id = str(record.get("donor_id", ""))
        global_person_id = str(record.get("global_person_id", ""))
        kind = str(record.get("kind", ""))
        if not donor_id or donor_id in seen:
            raise ValueError(f"donor IDs must be non-empty and unique: {donor_id!r}")
        if not global_person_id:
            raise ValueError(f"donor {donor_id!r} must have a global_person_id pseudonym")
        if kind not in grouped:
            raise ValueError(f"donor {donor_id!r} has unsupported kind {kind!r}")
        path = Path(str(record.get("path", "")))
        if not path.is_absolute():
            path = manifest.parent / path
        path = path.resolve()
        expected_sha = record.get("sha256")
        actual_sha = sha256_file(path)
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError(f"donor {donor_id!r} must have an exact SHA-256 digest")
        if expected_sha != actual_sha:
            raise ValueError(f"checksum mismatch for donor {donor_id!r}")
        with np.load(path, allow_pickle=False) as data:
            if "vertices" not in data.files:
                raise ValueError(f"donor {donor_id!r} file has no 'vertices' array")
            vertices = np.asarray(data["vertices"], dtype=np.float64).copy()
        _check_vertices(vertices, expected_shape, donor_id)
        grouped[kind].append(
            Donor(
                donor_id=donor_id,
                global_person_id=global_person_id,
                kind=kind,
                vertices=vertices,
                source_dataset=str(record.get("source_dataset", "unspecified")),
                source_sha256=actual_sha,
            )
        )
        seen.add(donor_id)

    for kind, donors in grouped.items():
        if len(donors) < 3:
            raise ValueError(f"donor manifest needs at least three {kind} donors")
        donors.sort(key=lambda donor: donor.donor_id)
    return grouped["whole_heart"], grouped["valve"]


def _similarity_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    covariance = source_centered.T @ target_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    denominator = float(np.sum(source_centered**2))
    if denominator <= 1e-12:
        raise ValueError("degenerate annulus in valve donor")
    scale = float(np.sum(singular_values) / denominator)
    return rotation, scale


def _ring_handedness(vertices: np.ndarray, annulus: np.ndarray, side: np.ndarray) -> float:
    """Sign of the ring's right-hand vector area along the direction to ``side``."""

    ring = vertices[annulus]
    centre = ring.mean(axis=0)
    closed = np.vstack([ring, ring[:1]])
    area = 0.5 * np.cross(closed[:-1] - centre, closed[1:] - centre).sum(axis=0)
    return float(np.sign(area @ (vertices[side].mean(axis=0) - centre)))


def oriented_valve_vertices(heart: np.ndarray, valve: np.ndarray, annulus: np.ndarray,
                            chamber: np.ndarray, leaflet_free: np.ndarray) -> np.ndarray:
    """The valve re-indexed, if needed, so its ring has the chamber's handedness.

    The free edge and the chamber lie on the same (ventricular) side of the
    annulus. If the valve's ring, read with the right-hand rule, points toward
    its free edge while the heart's ring points away from its chamber (or the
    reverse), the valve's ring and free edge are re-indexed by
    ``k -> (n/2 - 1 - k) mod n``. That reverses the ring's direction while
    keeping each leaflet on its own arc (``0..n/2-1`` anterior, the rest
    posterior), so a proper rotation matching vertex k to vertex k then carries
    the free edge to the ventricular side. The valve's shape is not mirrored.
    """

    if (_ring_handedness(heart, annulus, chamber)
            == _ring_handedness(valve, annulus, leaflet_free)):
        return valve
    n = len(annulus)
    order = (n // 2 - 1 - np.arange(n)) % n
    oriented = valve.copy()
    oriented[annulus] = valve[annulus[order]]
    oriented[leaflet_free] = valve[leaflet_free[order]]
    return oriented


def combine_donors(
    heart: Donor,
    valve: Donor,
    n_annulus: int | None = None,
    *,
    template: HeterogeneousTemplate | None = None,
    orient_valve: bool = False,
    chamber_levels: int = 1,
    leaflet_rows: int = 1,
) -> np.ndarray:
    """Create one composite identity while preserving one physical annular ring.

    ``orient_valve`` re-indexes the valve to the chamber's ring handedness first
    (``oriented_valve_vertices``). Off by default, which reproduces every
    composite built before the option existed, including their known defect:
    MVSeg2023 and ACDC rings have opposite handedness, so without it the valve's
    free edge is carried to the atrial side.
    """

    if heart.kind != "whole_heart" or valve.kind != "valve":
        raise ValueError("combine_donors expects one whole-heart donor and one valve donor")
    if orient_valve and template is not None:
        raise NotImplementedError("valve orientation is implemented for the procedural topology")
    if template is None:
        if n_annulus is None:
            raise ValueError("n_annulus is required for the procedural test topology")
        fixture = make_toy_complex(n_annulus=n_annulus, chamber_levels=chamber_levels,
                                   leaflet_rows=leaflet_rows)
        expected_shape = fixture.global_vertices.shape
        annulus = fixture.groups["annulus"]
        leaflet_free = fixture.groups["leaflet_free"]
        # Every leaflet vertex the valve supplies: the free edge, and on a richer
        # instance (W2.6) the rows between it and the root, row by row.
        leaflet_rows_ids = np.setdiff1d(fixture.groups["tee_visible"], annulus)
    else:
        expected_shape = template.vertices.shape
        annulus = template.annulus_global_indices
        leaflet_vertices = np.unique(
            np.concatenate(
                (
                    template.components["anterior_leaflet"].local_to_global,
                    template.components["posterior_leaflet"].local_to_global,
                )
            )
        )
        leaflet_free = np.setdiff1d(leaflet_vertices, annulus, assume_unique=False)
        leaflet_rows_ids = leaflet_free
    _check_vertices(heart.vertices, expected_shape, heart.donor_id)
    _check_vertices(valve.vertices, expected_shape, valve.donor_id)

    combined = heart.vertices.copy()
    valve_vertices = valve.vertices
    if orient_valve:
        chamber = np.setdiff1d(fixture.groups["cmr_visible"], annulus)
        valve_vertices = oriented_valve_vertices(heart.vertices, valve.vertices, annulus,
                                                 chamber, leaflet_free)
        if leaflet_rows > 1 and valve_vertices is not valve.vertices:
            # The interior rows are re-indexed as the free edge is, row by row.
            n = len(annulus)
            order = (n // 2 - 1 - np.arange(n)) % n
            for row in leaflet_rows_ids.reshape(-1, n)[:-1]:
                valve_vertices[row] = valve.vertices[row[order]]
    rotation, scale = _similarity_transform(valve_vertices[annulus], heart.vertices[annulus])
    source_center = valve_vertices[annulus].mean(axis=0)
    target_center = heart.vertices[annulus].mean(axis=0)
    combined[leaflet_rows_ids] = (
        scale * ((valve_vertices[leaflet_rows_ids] - source_center) @ rotation.T) + target_center
    )
    combined[annulus] = heart.vertices[annulus]
    return combined


def donor_manifest_template() -> dict[str, Any]:
    return {
        "schema_version": DONOR_SCHEMA,
        "topology_kind": "frozen_production_v1",
        "template_id": "replace-with-frozen-template-id",
        "donors": [
            {
                "donor_id": "pseudonymized-heart-0001",
                "global_person_id": "global-person-pseudonym-0001",
                "kind": "whole_heart",
                "path": "fixed_topology/heart-0001.npz",
                "sha256": "replace-with-file-sha256",
                "source_dataset": "ACDC-or-local",
            },
            {
                "donor_id": "pseudonymized-valve-0001",
                "global_person_id": "global-person-pseudonym-0002",
                "kind": "valve",
                "path": "fixed_topology/valve-0001.npz",
                "sha256": "replace-with-file-sha256",
                "source_dataset": "MVSeg2023-or-local",
            },
        ],
    }
