r"""Anatomical clocking of composite valves (E108, weak point 1b).

The donor pools number their rings from directions that carry no anatomy: an
ACDC chamber from the image's column axis, an MVSeg2023 valve from the first
principal axis of its voxel cloud, whose sign is arbitrary. ``combine_donors``
matches ring vertex k to vertex k, so where the valve's leaflets sit around its
chamber, its rotation about the annular axis, is arbitrary too.

This encodes both donors from anatomy instead, so that the same vertex-to-vertex
matching carries each valve onto its chamber in an anatomical relation:

* **Valve.** The ring is numbered from the commissures found from the leaflet
  labels (``fit_valve_commissural``, ``fit_valve_fixed`` with equal node counts
  and segmentation gaps at a commissure bridged): vertices ``0 .. n/2 - 1`` span
  the anterior leaflet's annular arc and the rest the posterior's, at equal
  angular steps within each arc, each with its own leaflet's free edge above it.
* **Chamber.** The septum is found from the right-ventricular cavity, as the
  standard segmentation of the left ventricle finds it: the arc of the left
  ventricle's myocardium in contact with the right ventricle runs between the two
  right-ventricular insertion points, and its bisector is the septal direction.
  Every chamber ring is resampled at fixed angles about the long axis: the
  anterior half of the vertices over an arc of width ``alpha`` centred on the
  septal direction, the posterior half over the rest.

The anterior (aortic, septal) leaflet then faces the septum and the posterior
(mural) leaflet the free wall. Neither dataset records patient orientation, so
which commissure is anterolateral is not assigned, and the anterior leaflet's
offset from the septal bisector toward the anterior wall is not reproduced.

Both angular conventions are right-handed about the axis pointing into the
ventricle (the valve normal from annulus to coaptation, the chamber axis from
base to apex), so the ring handedness agrees by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rooted4d.geometry import make_toy_complex

RV_LABEL = 1
MYOCARDIUM_LABEL = 2
CAVITY_LABEL = 3
#: A slice contributes when this many myocardium pixels touch the right ventricle.
MINIMUM_CONTACT = 5


@dataclass(frozen=True)
class SeptalDirection:
    """The septal direction of one labelled short-axis stack, in the fit's coordinates.

    ``direction`` is a unit vector in the short-axis plane, in the coordinates
    ``fit_lv_surface`` uses (row and column in mm, then slice position).
    """

    direction: np.ndarray
    slices: tuple[int, ...]
    arc_widths_deg: tuple[float, ...]
    deviations_deg: tuple[float, ...]


def _contact_arc(myocardium: np.ndarray, rv: np.ndarray, centre: np.ndarray,
                 spacing: tuple[float, float]) -> tuple[float, float] | None:
    """Bisector angle and width (radians) of the myocardium touching the right ventricle."""

    padded = np.pad(rv, 1)
    touches = (padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:])
    rows, columns = np.nonzero(myocardium & touches)
    if rows.size < MINIMUM_CONTACT:
        return None
    relative = np.stack([rows * spacing[0], columns * spacing[1]], axis=1) - centre
    angles = np.sort(np.arctan2(relative[:, 0], relative[:, 1]))
    # The contact arc is the circle minus its largest empty gap.
    gaps = np.diff(np.concatenate([angles, angles[:1] + 2.0 * np.pi]))
    largest = int(np.argmax(gaps))
    start = angles[(largest + 1) % len(angles)]
    width = 2.0 * np.pi - float(gaps[largest])
    return float(start + 0.5 * width), width


def septal_direction(labels: np.ndarray, zooms, *, base_fraction: float = 0.5) -> SeptalDirection | None:
    """The bisector of the right-ventricular insertion points, over the basal slices.

    Slices are taken in stack order (ACDC stacks start at the base) among those
    with at least 20 cavity pixels, the slices ``fit_lv_surface`` can use; the
    first ``base_fraction`` of them are the basal ones. In each, the arc of
    myocardium touching the right-ventricular cavity runs between the two
    insertion points; its bisector is measured about the cavity centroid, the
    point ``slice_cavity_contour`` measures angles about. The direction is the
    normalised mean of the slices' bisectors. None if no basal slice has contact.
    """

    volume = np.asarray(labels)
    spacing = (float(zooms[0]), float(zooms[1]))
    cavity_slices = [k for k in range(volume.shape[2]) if (volume[:, :, k] == CAVITY_LABEL).sum() >= 20]
    if not cavity_slices:
        return None
    basal = cavity_slices[: max(1, int(np.ceil(base_fraction * len(cavity_slices))))]
    vectors, used, widths = [], [], []
    for k in basal:
        plane = volume[:, :, k]
        rows, columns = np.nonzero(plane == CAVITY_LABEL)
        centre = np.array([rows.mean() * spacing[0], columns.mean() * spacing[1]])
        arc = _contact_arc(plane == MYOCARDIUM_LABEL, plane == RV_LABEL, centre, spacing)
        if arc is None:
            continue
        bisector, width = arc
        # (row, column) components, as fit_lv_surface places a contour point at angle t.
        vectors.append(np.array([np.sin(bisector), np.cos(bisector)]))
        used.append(k)
        widths.append(np.degrees(width))
    if not vectors:
        return None
    mean = np.mean(vectors, axis=0)
    mean /= np.linalg.norm(mean)
    deviations = tuple(float(np.degrees(np.arccos(np.clip(v @ mean, -1.0, 1.0)))) for v in vectors)
    return SeptalDirection(direction=np.array([mean[0], mean[1], 0.0]), slices=tuple(used),
                           arc_widths_deg=tuple(float(w) for w in widths), deviations_deg=deviations)


def ring_target_angles(n_annulus: int, alpha: float) -> np.ndarray:
    """Angles of the ring vertices about the septal direction (radians, right-handed about the axis).

    The anterior half spans ``[-alpha/2, alpha/2]`` and the posterior half the
    rest, each at equal steps with the first and last vertex half a step inside
    its arc, as ``fit_valve_fixed`` places its nodes between the commissures.
    """

    half = n_annulus // 2
    anterior = -0.5 * alpha + (np.arange(half) + 0.5) * alpha / half
    posterior = 0.5 * alpha + (np.arange(n_annulus - half) + 0.5) * (2.0 * np.pi - alpha) / (n_annulus - half)
    return np.concatenate([anterior, posterior])


def _frame(axis: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = axis / np.linalg.norm(axis)
    e1 = reference - (reference @ u) * u
    e1 /= np.linalg.norm(e1)
    return u, e1, np.cross(u, e1)


def resample_ring_at_angles(ring: np.ndarray, axis: np.ndarray, reference: np.ndarray,
                            angles: np.ndarray) -> np.ndarray:
    """A closed ring re-sampled at ``angles`` about its centroid, by linear interpolation in angle.

    Angles are right-handed about ``axis`` from ``reference`` (projected onto the
    plane normal to ``axis``). The ring must be star-shaped about its centroid in
    that plane, as the ACDC fits are by construction (one radius per angle).
    """

    points = np.asarray(ring, dtype=np.float64)
    u, e1, e2 = _frame(axis, reference)
    centred = points - points.mean(axis=0)
    psi = np.arctan2(centred @ e2, centred @ e1)
    order = np.argsort(psi)
    # Star-shaped: sorting by angle only rotates (or reverses) the ring's own order.
    steps = np.diff(np.concatenate([order, order[:1]])) % len(order)
    if not (np.all(steps == 1) or np.all(steps == len(order) - 1)):
        raise ValueError("ring is not star-shaped about its centroid")
    return np.stack([np.interp(np.asarray(angles), psi[order], points[order, axis_index], period=2.0 * np.pi)
                     for axis_index in range(3)], axis=1)


def chamber_vertices_clocked(fit, septal: np.ndarray, alpha: float, n_annulus: int = 24,
                             chamber_levels: int = 1, leaflet_rows: int = 1) -> np.ndarray:
    """``chamber_donor_vertices`` with every ring sampled at the anatomical angles.

    The same rings are taken as there (the base ring, the middle ring or evenly
    spaced levels, the last ring's centroid as the apex); only where each ring's
    vertices fall changes: ``ring_target_angles`` about the long axis (base ring
    centroid to apex), from ``septal``.
    """

    fixture = make_toy_complex(n_annulus=n_annulus, chamber_levels=chamber_levels, leaflet_rows=leaflet_rows)
    vertices = fixture.global_vertices.copy()
    rings = fit.vertices[: fit.n_rings * fit.n_angles].reshape(fit.n_rings, fit.n_angles, 3)
    first = np.linalg.norm(rings[0, :, :2] - rings[0, :, :2].mean(axis=0), axis=1).mean()
    last = np.linalg.norm(rings[-1, :, :2] - rings[-1, :, :2].mean(axis=0), axis=1).mean()
    ordered = rings if first >= last else rings[::-1]
    apex = ordered[-1].mean(axis=0)
    axis = apex - ordered[0].mean(axis=0)
    angles = ring_target_angles(n_annulus, alpha)

    vertices[fixture.groups["annulus"]] = resample_ring_at_angles(ordered[0], axis, septal, angles)
    lower = np.setdiff1d(fixture.groups["cmr_visible"], fixture.groups["annulus"])
    if chamber_levels == 1:
        vertices[lower[:-1]] = resample_ring_at_angles(ordered[len(ordered) // 2], axis, septal, angles)
    else:
        if len(ordered) < chamber_levels + 2:
            raise ValueError(f"a {len(ordered)}-ring fit cannot supply {chamber_levels} levels")
        for k, ids in enumerate(lower[:-1].reshape(chamber_levels, n_annulus), start=1):
            ring = int(round(k * (len(ordered) - 1) / (chamber_levels + 1)))
            vertices[ids] = resample_ring_at_angles(ordered[ring], axis, septal, angles)
    vertices[lower[-1]] = apex
    return vertices


def fit_valve_commissural(labels: np.ndarray, zooms, *, n_per_leaflet: int, n_depth: int = 6, case: str = "",
                          max_missing_fraction: float = 0.25):
    """``fit_valve_fixed`` with equal node counts per leaflet, and tissue gaps at a commissure bridged.

    The commissure walk stops at the first angle where the anterior leaflet no
    longer dominates near the annulus, which can be inside a gap of the
    segmentation where neither leaflet has voxels. At fine resolution the nodes
    next to such a commissure then hold no tissue even with their sector widened
    threefold, and ``fit_valve_fixed`` gives up. Here such a node is missing, and
    its whole depth profile is interpolated in angle between the nearest sampled
    nodes on either side, around the ring and so across the commissure. Every
    sampled node is exactly ``fit_valve_fixed``'s.

    Returns ``(fit, missing)``, the number of interpolated nodes; None if the walk
    fails or more than ``max_missing_fraction`` of either leaflet's nodes are missing.
    """

    from rooted4d.donors.valve import ANTERIOR_LABEL, POSTERIOR_LABEL
    from rooted4d.donors.valve_fixed import (MINIMUM_VOXELS, FixedValveFit, _commissure_frame,
                                             _labelled_points, _profile, _wrap)

    labelled = _labelled_points(labels, zooms)
    if labelled is None:
        return None
    points, which = labelled
    _, height, angle, angles = _commissure_frame(points, which)
    if angles is None:
        return None
    low, high = angles
    spans = (float(_wrap(high - low) % (2.0 * np.pi)),)
    spans += (2.0 * np.pi - spans[0],)
    theta, profiles, valid = [], [], []
    for label, start, span in ((ANTERIOR_LABEL, low, spans[0]), (POSTERIOR_LABEL, low + spans[0], spans[1])):
        mine = which == label
        half = 0.5 * span / n_per_leaflet
        for k in range(n_per_leaflet):
            unwrapped = start + (k + 0.5) * span / n_per_leaflet
            target = _wrap(unwrapped)
            for widen in (1.0, 2.0, 3.0):
                in_sector = mine & (np.abs(_wrap(angle - target)) <= widen * half)
                if in_sector.sum() >= MINIMUM_VOXELS:
                    profiles.append(_profile(points[in_sector], height[in_sector], n_depth))
                    valid.append(True)
                    break
            else:
                profiles.append(np.full((n_depth, 3), np.nan))
                valid.append(False)
            theta.append(unwrapped)
    profiles, valid, theta = np.stack(profiles), np.asarray(valid), np.asarray(theta)
    limit = max_missing_fraction * n_per_leaflet
    if (~valid[:n_per_leaflet]).sum() > limit or (~valid[n_per_leaflet:]).sum() > limit:
        return None
    if (~valid).any():
        for depth in range(n_depth):
            for axis in range(3):
                profiles[~valid, depth, axis] = np.interp(theta[~valid], theta[valid], profiles[valid, depth, axis],
                                                          period=2.0 * np.pi)
    anterior, posterior = profiles[:n_per_leaflet], profiles[n_per_leaflet:]
    fit = FixedValveFit(annulus=np.vstack([anterior[:, 0], posterior[:, 0]]), anterior=anterior,
                        posterior=posterior, commissure_angles=(low, high), case=case)
    return fit, int((~valid).sum())


def valve_vertices_commissural(fit, n_annulus: int = 24, chamber_levels: int = 1,
                               leaflet_rows: int = 1) -> np.ndarray:
    """A ``FixedValveFit`` with ``n_annulus / 2`` nodes per leaflet, in the procedural topology.

    The fit's anterior nodes become ring vertices ``0 .. n/2 - 1`` and its
    posterior nodes the rest, in the fit's own order (right-handed about the
    normal from annulus to coaptation, from the first commissure). Each leaflet
    row is the fit's depth row as ``valve_donor_vertices`` picks it: the deepest
    for the free edge, evenly spaced rows on a richer instance.
    """

    half = n_annulus // 2
    if fit.n_anterior != half or fit.n_posterior != n_annulus - half:
        raise ValueError(f"the fit has {fit.n_anterior} + {fit.n_posterior} nodes, not {half} + {n_annulus - half}")
    fixture = make_toy_complex(n_annulus=n_annulus, chamber_levels=chamber_levels, leaflet_rows=leaflet_rows)
    vertices = fixture.global_vertices.copy()
    vertices[fixture.groups["annulus"]] = fit.annulus
    rows = np.setdiff1d(fixture.groups["tee_visible"], fixture.groups["annulus"]).reshape(leaflet_rows, n_annulus)
    depth = fit.n_depth
    if depth < leaflet_rows + 1:
        raise ValueError(f"a {depth}-row valve fit cannot supply {leaflet_rows} leaflet rows")
    for r, ids in enumerate(rows, start=1):
        level = depth - 1 if r == leaflet_rows else int(round(r * (depth - 1) / leaflet_rows))
        vertices[ids[:half]] = fit.anterior[:, level]
        vertices[ids[half:]] = fit.posterior[:, level]
    return vertices


def anterior_span(fit) -> float:
    """The anterior leaflet's annular arc (radians) between the fit's commissures."""

    low, high = fit.commissure_angles
    return float((high - low) % (2.0 * np.pi))


def valve_clocking(heart: np.ndarray, valve: np.ndarray, anterior_direction: np.ndarray, septal: np.ndarray,
                   n_annulus: int, *, orient: bool = True, chamber_levels: int = 1, leaflet_rows: int = 1) -> float:
    """Signed angle (degrees) from the septum to where a composite turns the valve's anterior leaflet.

    ``anterior_direction`` is the valve's own anterior direction (from its centre
    to its anterior leaflet's voxel centroid, in the donor's coordinates, as
    ``valve_commissures`` gives it). It is turned by the rotation
    ``combine_donors`` carries the valve with (after the same orientation
    re-indexing when ``orient``), and its angle is measured about the chamber's
    long axis (annulus centroid to apex), right-handed, from ``septal``. Zero when
    the anterior leaflet faces the septal bisector. The leaflet's own direction is
    used rather than its free edge, whose centroid can lie near or past the
    annulus centre where the anterior leaflet reaches across at coaptation.
    """

    from rooted4d.synthetic.donors import _similarity_transform, oriented_valve_vertices

    fixture = make_toy_complex(n_annulus=n_annulus, chamber_levels=chamber_levels, leaflet_rows=leaflet_rows)
    groups = fixture.groups
    annulus = groups["annulus"]
    chamber = np.setdiff1d(groups["cmr_visible"], annulus)
    if orient:
        valve = oriented_valve_vertices(heart, valve, annulus, chamber, groups["leaflet_free"])
    rotation, _ = _similarity_transform(valve[annulus], heart[annulus])
    centre = heart[annulus].mean(axis=0)
    u, e1, e2 = _frame(heart[chamber[-1]] - centre, np.asarray(septal, dtype=np.float64))
    turned = rotation @ np.asarray(anterior_direction, dtype=np.float64)
    return float(np.degrees(np.arctan2(turned @ e2, turned @ e1)))
