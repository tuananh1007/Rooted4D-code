r"""Candidate cine-CMR planes and the evidence a plane would supply.

ActivePlaneTwin selects *where to put the next imaging plane*. That is a
different axis from the two closest prior works: AcquisitionFocus optimizes plane
orientation but against a reconstruction loss, one shot, for exactly two views;
CUTE-MRI adapts how much k-space to sample while the plane geometry stays fixed.

Evidence from a plane is its analytic intersection with the current mesh,
sampled along the resulting contour. That operator is exact and cheap, which
matters because the selection criterion evaluates many candidates per decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rooted4d.observation import (
    CMRPlane,
    intersect_surface_with_plane,
    sample_plane_segments,
)


def _orthonormal_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane with the given normal."""

    unit = np.asarray(normal, dtype=np.float64)
    unit = unit / np.linalg.norm(unit)
    seed = np.array([0.0, 0.0, 1.0]) if abs(unit[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    row = np.cross(seed, unit)
    row = row / np.linalg.norm(row)
    return row, np.cross(unit, row)


@dataclass(frozen=True)
class PlaneCandidate:
    """One acquirable plane, with the cost of acquiring it."""

    name: str
    origin: np.ndarray
    normal: np.ndarray
    cost: float = 1.0

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin, dtype=np.float64)
        normal = np.asarray(self.normal, dtype=np.float64)
        if origin.shape != (3,) or normal.shape != (3,):
            raise ValueError("origin and normal must have shape (3,)")
        norm = float(np.linalg.norm(normal))
        if norm < 1e-12:
            raise ValueError("normal must be non-degenerate")
        if not np.isfinite(self.cost) or self.cost <= 0.0:
            raise ValueError("cost must be positive and finite")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "normal", normal / norm)

    def to_plane(self) -> CMRPlane:
        row, column = _orthonormal_frame(self.normal)
        return CMRPlane(origin=self.origin, row_direction=row, column_direction=column)


def short_axis_stack(
    vertices: np.ndarray,
    *,
    n_planes: int = 9,
    cost: float = 1.0,
    span: float = 0.8,
) -> list[PlaneCandidate]:
    """The conventional short-axis stack, used as the protocol-order baseline."""

    points = np.asarray(vertices, dtype=np.float64)
    centre = 0.5 * (points.min(axis=0) + points.max(axis=0))
    extent = float(points[:, 2].max() - points[:, 2].min())
    offsets = np.linspace(-0.5 * span * extent, 0.5 * span * extent, n_planes)
    return [
        PlaneCandidate(
            name=f"SA{index:02d}",
            origin=centre + np.array([0.0, 0.0, float(offset)]),
            normal=np.array([0.0, 0.0, 1.0]),
            cost=cost,
        )
        for index, offset in enumerate(offsets)
    ]


def long_axis_candidates(
    vertices: np.ndarray,
    *,
    n_rotations: int = 6,
    cost: float = 1.0,
) -> list[PlaneCandidate]:
    """Long-axis planes through the centre at evenly spaced rotations."""

    points = np.asarray(vertices, dtype=np.float64)
    centre = 0.5 * (points.min(axis=0) + points.max(axis=0))
    angles = np.pi * np.arange(n_rotations) / n_rotations
    return [
        PlaneCandidate(
            name=f"LA{index:02d}",
            origin=centre,
            normal=np.array([np.cos(angle), np.sin(angle), 0.0]),
            cost=cost,
        )
        for index, angle in enumerate(angles)
    ]


def oblique_candidates(
    vertices: np.ndarray,
    *,
    n_rotations: int = 6,
    tilts: tuple[float, ...] = (0.35, 0.7),
    cost: float = 1.2,
) -> list[PlaneCandidate]:
    """Tilted planes, which a fixed clinical protocol would never prescribe.

    These exist so the selector can be shown choosing something a protocol does
    not contain. Without them the comparison would only be about ordering.
    """

    points = np.asarray(vertices, dtype=np.float64)
    centre = 0.5 * (points.min(axis=0) + points.max(axis=0))
    candidates = []
    for tilt_index, tilt in enumerate(tilts):
        for index in range(n_rotations):
            angle = 2.0 * np.pi * index / n_rotations
            normal = np.array(
                [np.sin(tilt) * np.cos(angle), np.sin(tilt) * np.sin(angle), np.cos(tilt)]
            )
            candidates.append(
                PlaneCandidate(
                    name=f"OB{tilt_index}{index:02d}",
                    origin=centre,
                    normal=normal,
                    cost=cost,
                )
            )
    return candidates


def plane_contour_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    candidate: PlaneCandidate,
    *,
    samples_per_segment: int = 2,
) -> np.ndarray:
    """Points where this plane cuts the mesh, in plane coordinates.

    Returns an empty array when the plane misses the mesh, which is a legitimate
    outcome the selector must be able to score rather than crash on.
    """

    plane = candidate.to_plane()
    intersection = intersect_surface_with_plane(vertices, faces, plane)
    if not len(intersection.segments_2d):
        return np.empty((0, 2), dtype=np.float64)
    return sample_plane_segments(
        intersection.segments_2d, samples_per_segment=samples_per_segment
    )
