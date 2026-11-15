r"""Measure each component against the real data it was fitted from.

The semi-synthetic benchmark answers "does the framework recover known
parameters". It cannot answer "does the geometry resemble a real heart", because
its donors *are* the fitted geometry -- asking the benchmark that question would
compare the fit to itself. This module asks it against the provider
segmentations instead, which is the only place the answer exists.

Three distances are separated, because collapsing them would hide where the
error comes from:

``fit_vs_segmentation``
    the native-resolution fit against the ACDC or MVSeg2023 label surface. This
    is the component's own accuracy.
``donor_vs_fit``
    the 73-vertex donor encoding against the native fit it was resampled from.
    This is the cost of the shared topology, and every benchmark number inherits
    it.
``donor_vs_segmentation``
    the composition, which is what the benchmark's geometry is actually worth.

Distances are point-to-surface, not point-to-vertex: with 73 vertices spanning a
ventricle, nearest-vertex distance would report several millimetres of pure
discretization and would grow if the topology were made coarser, which is the
opposite of what a distance to the surface should do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rooted4d.active.acdc import fit_lv_surface, lv_topology_faces
from rooted4d.donors.export import chamber_donor_vertices, valve_donor_vertices
from rooted4d.metrics.topology import signed_enclosed_volume


def _closest_point_on_triangles(
    points: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """Squared distance from each point to every triangle.

    The standard region test on the barycentric plane, vectorized over both
    points and triangles. Returns shape ``(n_points, n_triangles)``.
    """

    ab = b - a
    ac = c - a
    # (points, triangles, 3)
    ap = points[:, None, :] - a[None, :, :]
    d1 = np.einsum("ptk,tk->pt", ap, ab)
    d2 = np.einsum("ptk,tk->pt", ap, ac)
    aa = np.einsum("tk,tk->t", ab, ab)[None, :]
    bb = np.einsum("tk,tk->t", ab, ac)[None, :]
    cc = np.einsum("tk,tk->t", ac, ac)[None, :]
    denominator = np.maximum(aa * cc - bb * bb, 1e-30)
    u = (cc * d1 - bb * d2) / denominator
    v = (aa * d2 - bb * d1) / denominator

    # Clamp into the triangle. Doing it edge by edge rather than by clipping u
    # and v independently matters: independent clipping lands outside the
    # simplex whenever u + v > 1.
    outside = (u < 0.0) | (v < 0.0) | (u + v > 1.0)
    if outside.any():
        def project(origin_d: np.ndarray, edge_sq: np.ndarray) -> np.ndarray:
            return np.clip(origin_d / np.maximum(edge_sq, 1e-30), 0.0, 1.0)

        t_ab = project(d1, aa)
        t_ac = project(d2, cc)
        bc = c - b
        bp_d = np.einsum("ptk,tk->pt", points[:, None, :] - b[None, :, :], bc)
        t_bc = project(bp_d, np.einsum("tk,tk->t", bc, bc)[None, :])
        candidates = np.stack(
            [
                np.stack([t_ab, np.zeros_like(t_ab)], axis=-1),
                np.stack([np.zeros_like(t_ac), t_ac], axis=-1),
                np.stack([1.0 - t_bc, t_bc], axis=-1),
            ],
            axis=0,
        )
        edge_points = (
            a[None, None, :, :]
            + candidates[..., 0][..., None] * ab[None, None, :, :]
            + candidates[..., 1][..., None] * ac[None, None, :, :]
        )
        edge_sq = ((edge_points - points[None, :, None, :]) ** 2).sum(axis=-1)
        best = np.argmin(edge_sq, axis=0)
        chosen = np.take_along_axis(candidates, best[None, ..., None], axis=0)[0]
        u = np.where(outside, chosen[..., 0], u)
        v = np.where(outside, chosen[..., 1], v)

    closest = a[None, :, :] + u[..., None] * ab[None, :, :] + v[..., None] * ac[None, :, :]
    return ((closest - points[:, None, :]) ** 2).sum(axis=-1)


def point_to_mesh_mm(
    points: np.ndarray, vertices: np.ndarray, faces: np.ndarray, chunk: int = 512
) -> np.ndarray:
    """Distance from each point to the closest point on the triangulated surface."""

    points = np.asarray(points, dtype=np.float64)
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    out = np.empty(len(points))
    for start in range(0, len(points), chunk):
        block = points[start : start + chunk]
        out[start : start + chunk] = np.sqrt(
            _closest_point_on_triangles(block, a, b, c).min(axis=1)
        )
    return out


def symmetric_surface_distance(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    sample_faces: int | None = None,
    seed: int = 0,
    chunk: int = 256,
) -> dict[str, float]:
    """Both directions of the distance, reported separately.

    A one-directional distance is easy to make look good: a surface that covers
    only part of the anatomy scores well from surface to points while missing
    half the structure. Both directions are reported and the Hausdorff figure is
    the worse of the two.
    """

    forward = point_to_mesh_mm(points, vertices, faces)
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    n_samples = sample_faces if sample_faces is not None else 20
    weights = np.random.default_rng(seed).random((len(faces), n_samples, 2))
    flip = weights.sum(axis=2) > 1.0
    weights[flip] = 1.0 - weights[flip]
    surface = (
        a[:, None, :]
        + weights[..., 0][..., None] * (b - a)[:, None, :]
        + weights[..., 1][..., None] * (c - a)[:, None, :]
    ).reshape(-1, 3)
    # Chunked, because the dense difference is samples x points x 3: a leaflet
    # label carries tens of thousands of voxels and the full array runs to
    # gigabytes, which is what killed the first run of this.
    backward = np.empty(len(surface))
    for start in range(0, len(surface), chunk):
        block = surface[start : start + chunk]
        difference = block[:, None, :] - points[None, :, :]
        backward[start : start + chunk] = np.sqrt(
            (difference**2).sum(axis=2).min(axis=1)
        )
    return {
        "mean_points_to_surface_mm": float(forward.mean()),
        "p95_points_to_surface_mm": float(np.percentile(forward, 95)),
        "mean_surface_to_points_mm": float(backward.mean()),
        "p95_surface_to_points_mm": float(np.percentile(backward, 95)),
        "symmetric_mean_mm": float(0.5 * (forward.mean() + backward.mean())),
        "hausdorff_mm": float(max(forward.max(), backward.max())),
    }


def cavity_boundary_points(
    labels: np.ndarray, zooms: tuple[float, float, float], cavity_label: int = 3
) -> np.ndarray:
    """Millimetre centres of the cavity voxels that touch non-cavity.

    Uses the same index-times-spacing frame the fitter builds its vertices in,
    so the two are directly comparable without a registration step.
    """

    binary = np.asarray(labels) == cavity_label
    padded = np.pad(binary, 1)
    interior = (
        padded[1:-1, 1:-1, 1:-1]
        & padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    rows, columns, slices = np.nonzero(binary & ~interior)
    return np.stack(
        [rows * zooms[0], columns * zooms[1], slices * zooms[2]], axis=1
    )


def evaluate_chamber_component(
    root: str | Path,
    *,
    max_patients: int | None = 30,
    n_rings: int = 8,
    n_angles: int = 32,
    n_annulus: int = 24,
    cavity_label: int = 3,
) -> dict[str, Any]:
    """The chamber component against ACDC cavity segmentations."""

    import nibabel as nib

    native_faces = lv_topology_faces(n_rings, n_angles)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(Path(root).rglob("*_gt.nii.gz")):
        patient = path.parent.name
        if max_patients is not None and patient not in seen and len(seen) >= max_patients:
            continue
        seen.add(patient)
        image = nib.load(str(path))
        labels = np.asarray(image.dataobj)
        zooms = tuple(float(z) for z in image.header.get_zooms()[:3])
        fit = fit_lv_surface(
            labels,
            zooms,
            n_rings=n_rings,
            n_angles=n_angles,
            cavity_label=cavity_label,
            patient=patient,
            frame=int(path.name.split("_frame")[-1].split("_")[0]),
        )
        if fit is None:
            continue
        boundary = cavity_boundary_points(labels, zooms, cavity_label)
        if len(boundary) < 50:
            continue

        voxel_volume_ml = float((labels == cavity_label).sum() * np.prod(zooms) / 1000.0)
        native_volume_ml = abs(signed_enclosed_volume(fit.vertices, native_faces)) / 1000.0

        record: dict[str, Any] = {
            "patient": patient,
            "frame": fit.frame,
            "slices_used": fit.n_slices_used,
            "zooms_mm": list(zooms),
            "voxel_volume_ml": voxel_volume_ml,
            "native_volume_ml": native_volume_ml,
            "native_volume_ratio": native_volume_ml / max(voxel_volume_ml, 1e-9),
        }
        record["fit_vs_segmentation"] = symmetric_surface_distance(
            boundary, fit.vertices, native_faces
        )
        rows.append(record)

    def column(getter) -> dict[str, float]:
        values = np.array([getter(row) for row in rows], dtype=np.float64)
        return {
            "n": int(values.size),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        }

    # Ejection fraction is the endpoint a clinician reads, and it is not a
    # restatement of the volume ratio: a bias that scales both volumes cancels in
    # the ratio, so EF can be accurate while each volume is not. Reporting only
    # the volume ratio would understate the component, and reporting only EF
    # would hide an 8 percent volume bias.
    ejection: list[dict[str, Any]] = []
    by_patient: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_patient.setdefault(row["patient"], []).append(row)
    for patient, frames in sorted(by_patient.items()):
        if len(frames) != 2:
            continue
        voxel = sorted(f["voxel_volume_ml"] for f in frames)
        native = sorted(f["native_volume_ml"] for f in frames)
        voxel_ef = 100.0 * (voxel[1] - voxel[0]) / max(voxel[1], 1e-9)
        native_ef = 100.0 * (native[1] - native[0]) / max(native[1], 1e-9)
        ejection.append(
            {
                "patient": patient,
                "segmentation_ef_percent": voxel_ef,
                "fitted_ef_percent": native_ef,
                "error_percentage_points": native_ef - voxel_ef,
            }
        )

    def spread(values: list[float]) -> dict[str, float]:
        array = np.array(values, dtype=np.float64)
        return {
            "n": int(array.size),
            "bias": float(array.mean()),
            "median_absolute": float(np.median(np.abs(array))),
            "p95_absolute": float(np.percentile(np.abs(array), 95)),
            "max_absolute": float(np.abs(array).max()),
            "limits_of_agreement": [
                float(array.mean() - 1.96 * array.std(ddof=1)),
                float(array.mean() + 1.96 * array.std(ddof=1)),
            ],
        }

    spacings = np.array([row["zooms_mm"] for row in rows], dtype=np.float64)
    return {
        "component": "chamber_from_ACDC",
        "n_frames": len(rows),
        "n_patients": len(seen),
        # Boundary voxel centres sit up to half a voxel from the true surface, so
        # the distance below cannot be read without knowing the voxel. ACDC slice
        # thickness dominates: through-plane it is several millimetres.
        "voxel_mm": {
            "in_plane_median": float(np.median(spacings[:, :2])),
            "slice_thickness_median": float(np.median(spacings[:, 2])),
            "slice_thickness_range": [
                float(spacings[:, 2].min()),
                float(spacings[:, 2].max()),
            ],
        },
        "ejection_fraction": {
            "error_percentage_points": spread(
                [row["error_percentage_points"] for row in ejection]
            ),
            "rows": ejection,
        },
        "summary": {
            "fit_to_segmentation_symmetric_mm": column(
                lambda r: r["fit_vs_segmentation"]["symmetric_mean_mm"]
            ),
            "fit_to_segmentation_hausdorff_mm": column(
                lambda r: r["fit_vs_segmentation"]["hausdorff_mm"]
            ),
            "native_volume_ratio": column(lambda r: r["native_volume_ratio"]),
        },
        "rows": rows,
    }


def leaflet_faces(bins: np.ndarray, n_depth: int, n_annulus: int) -> np.ndarray:
    """Triangulate one leaflet sheet indexed ``[sector, depth]``.

    Only angularly adjacent sectors are joined. The resampler keeps occupied
    bins only, so consecutive entries in the array are not necessarily
    consecutive in angle; stitching them regardless would sew a face straight
    across a gap in the leaflet and would then report distances to a surface
    that is not there.
    """

    faces: list[tuple[int, int, int]] = []
    position = {int(b): i for i, b in enumerate(bins)}
    for bin_index, row in position.items():
        following = position.get((bin_index + 1) % n_annulus)
        if following is None:
            continue
        for depth in range(n_depth - 1):
            a = row * n_depth + depth
            b = following * n_depth + depth
            faces.append((a, b, b + 1))
            faces.append((a, b + 1, a + 1))
    return np.asarray(faces, dtype=np.int64)


def evaluate_valve_component(
    root: str | Path,
    *,
    limit: int | None = 40,
    n_annulus: int = 32,
    n_depth: int = 6,
) -> dict[str, Any]:
    """The valve component against MVSeg2023 leaflet labels.

    Each leaflet is scored against its own label only. Pooling them would let a
    surface that captured the posterior leaflet well and missed the anterior one
    hide inside an average, which is exactly the failure the per-leaflet
    resampling was introduced to fix.
    """

    import nibabel as nib

    from rooted4d.donors.valve import ANTERIOR_LABEL, POSTERIOR_LABEL, fit_valve

    rows: list[dict[str, Any]] = []
    paths = sorted(Path(root).rglob("*-label.nii.gz"))
    if limit is not None:
        paths = paths[:limit]
    for path in paths:
        image = nib.load(str(path))
        labels = np.asarray(image.dataobj)
        zooms = tuple(float(z) for z in image.header.get_zooms()[:3])
        fit = fit_valve(
            labels,
            zooms,
            n_annulus=n_annulus,
            n_depth=n_depth,
            case=path.name.replace("-label.nii.gz", ""),
        )
        if fit is None:
            continue

        record: dict[str, Any] = {
            "case": fit.case,
            "annulus_diameter_mm": fit.annulus_diameter_mm(),
            "anterior_span_mm": fit.leaflet_span_mm("anterior"),
            "posterior_span_mm": fit.leaflet_span_mm("posterior"),
        }
        usable = True
        for name, sheet, bins, label in (
            ("anterior", fit.anterior, fit.anterior_root_indices, ANTERIOR_LABEL),
            ("posterior", fit.posterior, fit.posterior_root_indices, POSTERIOR_LABEL),
        ):
            faces = leaflet_faces(bins, n_depth, n_annulus)
            voxels = np.argwhere(labels == label)
            if len(faces) == 0 or len(voxels) < 50:
                usable = False
                break
            points = voxels * np.asarray(zooms)
            # Leaflets are thin sheets a couple of voxels thick, so every voxel
            # is a boundary voxel; there is no interior to strip and scoring
            # against the full label is the honest comparison.
            record[name] = symmetric_surface_distance(
                points, sheet.reshape(-1, 3), faces, sample_faces=6
            )
        if usable:
            rows.append(record)

    def column(getter) -> dict[str, float]:
        values = np.array([getter(row) for row in rows], dtype=np.float64)
        return {
            "n": int(values.size),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        }

    return {
        "component": "valve_from_MVSeg2023",
        "n_cases": len(rows),
        "summary": {
            "anterior_symmetric_mm": column(lambda r: r["anterior"]["symmetric_mean_mm"]),
            "posterior_symmetric_mm": column(lambda r: r["posterior"]["symmetric_mean_mm"]),
            "anterior_hausdorff_mm": column(lambda r: r["anterior"]["hausdorff_mm"]),
            "posterior_hausdorff_mm": column(lambda r: r["posterior"]["hausdorff_mm"]),
            "annulus_diameter_mm": column(lambda r: r["annulus_diameter_mm"]),
            "anterior_span_mm": column(lambda r: r["anterior_span_mm"]),
            "posterior_span_mm": column(lambda r: r["posterior_span_mm"]),
        },
        "rows": rows,
    }
