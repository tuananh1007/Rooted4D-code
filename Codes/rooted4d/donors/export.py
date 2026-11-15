r"""Export ACDC and MVSeg2023 fits as donors for the semi-synthetic benchmark.

`load_donors` deliberately refuses raw segmentations: donors must already be
registered to the shared topology, so per-case remeshing cannot silently change
correspondence. This performs that registration and writes the manifest the
loader expects.

Both donor kinds are stored as full-topology vertex arrays even though each
supplies only part of the anatomy, because `combine_donors` reads a heart
donor's chamber and a valve donor's annulus and free edges from the same
indexing. Slots a donor does not supply keep their template values and are never
read for that donor.

The topology used is the procedural one, declared honestly as
`procedural_test_v1`. The production template cannot be used because freezing it
is still blocked on the annular-resolution study and clinical review. That has a
real cost, recorded here rather than buried: the procedural chamber carries one
annular ring, one mid ring and an apex, so an eight-ring ACDC fit is compressed
into three levels, and the leaflets carry a single free-edge row, so a six-row
valve fit is reduced to its edge. The donors are real geometry at real scale in
consistent correspondence; they are not a faithful reproduction of every detail
the source segmentations contain.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rooted4d.geometry import make_toy_complex

DONOR_SCHEMA = "rooted4d-fixed-topology-donors-v2"


def _resample_ring(ring: np.ndarray, count: int) -> np.ndarray:
    """Resample a closed ring of ordered points to ``count`` samples by arc length.

    Parameterized by cumulative chord length in the order the points are given,
    which is the order the fitters already produce.

    The previous implementation sorted and interpolated by angle about the
    centroid *in the XY plane*, and both halves of that were wrong for this
    anatomy. The mitral annulus is tilted -- a measured plane normal of
    (0.34, 0.02, -0.94), around twenty degrees off the z axis -- so its XY
    projection is a foreshortened ellipse, and it is not star-shaped in that
    projection either, so the angular sort silently reordered neighbouring
    vertices. Resampling a 128-node ring back to 128 nodes, which must be a
    no-op, moved vertices by up to 6.7 mm and shortened the perimeter by
    2.8 percent. Arc length needs no star-shape assumption and no privileged
    axis.
    """

    points = np.asarray(ring, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("ring must have shape (points, 3) with at least three points")
    if count < 3:
        raise ValueError("count must be at least three")
    closed = np.vstack([points, points[:1]])
    step = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(step)])
    total = arc[-1]
    if total <= 0.0:
        raise ValueError("ring has zero length")
    targets = np.linspace(0.0, total, count, endpoint=False)
    return np.stack(
        [np.interp(targets, arc, closed[:, axis]) for axis in range(3)], axis=1
    )


def chamber_donor_vertices(fit, n_annulus: int = 24, chamber_levels: int = 1,
                           leaflet_rows: int = 1) -> np.ndarray:
    """Map an ACDC left-ventricle fit into the procedural topology.

    The reduced instance takes the fit's middle ring as its one lower ring. A
    richer one (``chamber_levels`` > 1) takes that many of the fit's rings,
    evenly spaced between the base ring and the last ring, whose centroid is the
    apex in both.
    """

    fixture = make_toy_complex(n_annulus=n_annulus, chamber_levels=chamber_levels,
                               leaflet_rows=leaflet_rows)
    vertices = fixture.global_vertices.copy()
    rings = fit.vertices[: fit.n_rings * fit.n_angles].reshape(
        fit.n_rings, fit.n_angles, 3
    )
    first = np.linalg.norm(rings[0, :, :2] - rings[0, :, :2].mean(axis=0), axis=1).mean()
    last = np.linalg.norm(rings[-1, :, :2] - rings[-1, :, :2].mean(axis=0), axis=1).mean()
    ordered = rings if first >= last else rings[::-1]

    vertices[fixture.groups["annulus"]] = _resample_ring(ordered[0], n_annulus)
    lower = np.setdiff1d(fixture.groups["cmr_visible"], fixture.groups["annulus"])
    if chamber_levels == 1:
        mid = ordered[len(ordered) // 2]
        lower_ring = lower[:-1]
        vertices[lower_ring] = _resample_ring(mid, len(lower_ring))
    else:
        if len(ordered) < chamber_levels + 2:
            raise ValueError(f"a {len(ordered)}-ring fit cannot supply {chamber_levels} levels")
        levels = lower[:-1].reshape(chamber_levels, n_annulus)
        for k, ids in enumerate(levels, start=1):
            ring = int(round(k * (len(ordered) - 1) / (chamber_levels + 1)))
            vertices[ids] = _resample_ring(ordered[ring], n_annulus)
    vertices[lower[-1]] = ordered[-1].mean(axis=0)
    return vertices


def valve_donor_vertices(fit, n_annulus: int = 24, chamber_levels: int = 1,
                         leaflet_rows: int = 1) -> np.ndarray:
    """Map an MVSeg2023 valve fit into the procedural topology.

    Each leaflet row is resampled from one row of the fit's depth profile: the
    free edge from the deepest, and on a richer instance (``leaflet_rows`` > 1)
    the others from rows evenly spaced between the root and the free edge.
    """

    fixture = make_toy_complex(n_annulus=n_annulus, chamber_levels=chamber_levels,
                               leaflet_rows=leaflet_rows)
    vertices = fixture.global_vertices.copy()
    vertices[fixture.groups["annulus"]] = _resample_ring(fit.annulus, n_annulus)

    if leaflet_rows == 1:
        free = fixture.groups["leaflet_free"]
        half = len(free) // 2
        vertices[free[:half]] = _resample_ring(fit.anterior[:, -1], half)
        vertices[free[half:]] = _resample_ring(fit.posterior[:, -1], len(free) - half)
        return vertices
    depth = fit.anterior.shape[1]
    if depth < leaflet_rows + 1:
        raise ValueError(f"a {depth}-row valve fit cannot supply {leaflet_rows} leaflet rows")
    rows = np.setdiff1d(fixture.groups["tee_visible"], fixture.groups["annulus"]).reshape(leaflet_rows, n_annulus)
    half = n_annulus // 2
    for r, ids in enumerate(rows, start=1):
        level = int(round(r * (depth - 1) / leaflet_rows))
        vertices[ids[:half]] = _resample_ring(fit.anterior[:, level], half)
        vertices[ids[half:]] = _resample_ring(fit.posterior[:, level], n_annulus - half)
    return vertices


def _canonicalize(vertices: np.ndarray) -> np.ndarray:
    """Move a donor to a canonical position by removing its centroid.

    Where a patient happened to lie in the scanner is not anatomy, and leaving it
    in is not free. Across the ACDC pool donor centroids are spread by about
    41 mm standard deviation while the hearts themselves are only about 30 mm in
    mean radius, so position dominates the variance: fitted on the raw pool, the
    first two principal modes come out 99.9 and 99.7 percent pure scanner x and
    y, and a twelve-mode basis spends two of its modes on placement. Centring
    lowers the representation error on unseen donors from 3.31 to 2.99 mm at
    sixteen modes and leaves the shape space describing shape.

    Only the translation is removed. Rotational Procrustes alignment was measured
    and does not help -- 3.00 against 2.99 mm at sixteen modes, and worse at
    twenty and twenty-four -- because the short-axis acquisition already aligns
    the long axis, so the extra freedom only fits noise across ninety donors.
    """

    points = np.asarray(vertices, dtype=np.float64)
    return points - points.mean(axis=0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def export_donors(
    chamber_fits: list,
    valve_fits: list,
    output_root: str | Path,
    *,
    n_annulus: int = 24,
    chamber_levels: int = 1,
    leaflet_rows: int = 1,
) -> Path:
    """Write donor arrays and the manifest `load_donors` validates.

    Person pseudonyms are generated per source case and are distinct across the
    two pools, because a chamber donor and a valve donor are different people and
    the split lock must never see them as one.
    """

    root = Path(output_root)
    (root / "fixed_topology").mkdir(parents=True, exist_ok=True)
    records = []

    for index, fit in enumerate(chamber_fits):
        vertices = _canonicalize(chamber_donor_vertices(fit, n_annulus=n_annulus, chamber_levels=chamber_levels,
                                                         leaflet_rows=leaflet_rows))
        name = f"heart-{index:04d}"
        path = root / "fixed_topology" / f"{name}.npz"
        np.savez_compressed(path, vertices=vertices)
        records.append(
            {
                "donor_id": f"acdc-{name}",
                "global_person_id": f"acdc-person-{index:04d}",
                "kind": "whole_heart",
                "path": f"fixed_topology/{name}.npz",
                "sha256": _sha256(path),
                "source_dataset": "ACDC",
            }
        )

    for index, fit in enumerate(valve_fits):
        # Valves are centred too, for uniformity. It makes no difference to the
        # composite, because ``combine_donors`` maps each valve onto its
        # chamber's annulus by similarity and discards the valve's own placement.
        vertices = _canonicalize(valve_donor_vertices(fit, n_annulus=n_annulus, chamber_levels=chamber_levels,
                                                       leaflet_rows=leaflet_rows))
        name = f"valve-{index:04d}"
        path = root / "fixed_topology" / f"{name}.npz"
        np.savez_compressed(path, vertices=vertices)
        records.append(
            {
                "donor_id": f"mvseg-{name}",
                "global_person_id": f"mvseg-person-{index:04d}",
                "kind": "valve",
                "path": f"fixed_topology/{name}.npz",
                "sha256": _sha256(path),
                "source_dataset": "MVSeg2023",
            }
        )

    manifest = {
        "schema_version": DONOR_SCHEMA,
        "topology_kind": "procedural_test_v1",
        "n_annulus": n_annulus,
        "provenance": {
            "chamber": "ACDC left-ventricle cavity fits",
            "valve": "MVSeg2023 mitral leaflet fits",
            "licence_note": (
                "ACDC is CC BY-NC-SA and MVSeg2023 is CC BY-NC-ND. These donors "
                "are derivative works of both, so the pool may be used here and "
                "cannot be redistributed; NoDerivatives blocks release of the "
                "valve side entirely."
            ),
        },
        "donors": records,
    }
    if chamber_levels != 1 or leaflet_rows != 1:
        # A richer procedural instance (W2.6); the reduced instance's manifest is unchanged.
        manifest["chamber_levels"] = chamber_levels
        manifest["leaflet_rows"] = leaflet_rows
    manifest_path = root / "donors.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
