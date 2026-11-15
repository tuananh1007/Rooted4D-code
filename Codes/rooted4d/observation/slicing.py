"""Analytic surface--plane intersection for CMR contour evaluation.

The finite-thickness likelihood renderer remains Rooted4D's image-domain
observation operator.  This module provides an independent, exact geometric
path for contour supervision, hidden-plane evaluation, and renderer checks.
It is deliberately NumPy-based and does not claim autograd support.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must have finite shape (3,)")
    length = float(np.linalg.norm(value))
    if length <= 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return value / length


@dataclass(frozen=True)
class CMRPlane:
    origin: np.ndarray
    row_direction: np.ndarray
    column_direction: np.ndarray

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin, dtype=np.float64)
        row = _unit(self.row_direction, "row_direction")
        column = _unit(self.column_direction, "column_direction")
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("origin must have finite shape (3,)")
        if abs(float(row @ column)) > 1e-8:
            raise ValueError("row_direction and column_direction must be orthogonal")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "row_direction", row)
        object.__setattr__(self, "column_direction", column)

    @property
    def normal(self) -> np.ndarray:
        return np.cross(self.row_direction, self.column_direction)


@dataclass(frozen=True)
class MeshPlaneIntersection:
    segments_3d: np.ndarray
    segments_2d: np.ndarray
    source_face_indices: np.ndarray
    coplanar_face_indices: np.ndarray


def _append_unique(points: list[np.ndarray], point: np.ndarray, tolerance: float) -> None:
    if not any(np.linalg.norm(point - existing) <= tolerance for existing in points):
        points.append(point)


def intersect_surface_with_plane(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane: CMRPlane,
    *,
    tolerance: float = 1e-9,
) -> MeshPlaneIntersection:
    """Intersect each triangle with a plane and return non-degenerate segments.

    A triangle lying entirely in the plane does not have a unique one-dimensional
    intersection.  Such faces are reported separately rather than silently
    converted into arbitrary contour edges.
    """

    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("vertices must have finite shape (vertices, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("faces must have shape (faces, 3)")
    if len(triangles) and (triangles.min() < 0 or triangles.max() >= len(points)):
        raise ValueError("faces contain an out-of-range vertex index")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    segments: list[np.ndarray] = []
    source_faces: list[int] = []
    coplanar_faces: list[int] = []
    edge_pairs = ((0, 1), (1, 2), (2, 0))
    for face_index, face in enumerate(triangles):
        triangle = points[face]
        distances = (triangle - plane.origin) @ plane.normal
        on_plane = np.abs(distances) <= tolerance
        if np.all(on_plane):
            coplanar_faces.append(face_index)
            continue
        intersections: list[np.ndarray] = []
        for first, second in edge_pairs:
            first_on = bool(on_plane[first])
            second_on = bool(on_plane[second])
            if first_on:
                _append_unique(intersections, triangle[first], tolerance)
            if second_on:
                _append_unique(intersections, triangle[second], tolerance)
            if not first_on and not second_on and distances[first] * distances[second] < 0.0:
                fraction = distances[first] / (distances[first] - distances[second])
                point = triangle[first] + fraction * (triangle[second] - triangle[first])
                _append_unique(intersections, point, tolerance)
        if len(intersections) == 2:
            segment = np.stack(intersections, axis=0)
            if np.linalg.norm(segment[1] - segment[0]) > tolerance:
                segments.append(segment)
                source_faces.append(face_index)

    if segments:
        segments_3d = np.stack(segments, axis=0)
        relative = segments_3d - plane.origin
        segments_2d = np.stack(
            [
                np.einsum("svi,i->sv", relative, plane.row_direction),
                np.einsum("svi,i->sv", relative, plane.column_direction),
            ],
            axis=-1,
        )
    else:
        segments_3d = np.empty((0, 2, 3), dtype=np.float64)
        segments_2d = np.empty((0, 2, 2), dtype=np.float64)
    return MeshPlaneIntersection(
        segments_3d=segments_3d,
        segments_2d=segments_2d,
        source_face_indices=np.asarray(source_faces, dtype=np.int64),
        coplanar_face_indices=np.asarray(coplanar_faces, dtype=np.int64),
    )


def sample_plane_segments(
    segments_2d: np.ndarray,
    *,
    samples_per_segment: int = 4,
) -> np.ndarray:
    """Sample segment interiors uniformly without duplicating endpoints."""

    segments = np.asarray(segments_2d, dtype=np.float64)
    if segments.ndim != 3 or segments.shape[1:] != (2, 2):
        raise ValueError("segments_2d must have shape (segments, 2, 2)")
    if samples_per_segment < 1:
        raise ValueError("samples_per_segment must be positive")
    if not len(segments):
        return np.empty((0, 2), dtype=np.float64)
    fractions = (np.arange(samples_per_segment, dtype=np.float64) + 0.5) / (
        samples_per_segment
    )
    samples = segments[:, 0:1, :] + fractions[None, :, None] * (
        segments[:, 1:2, :] - segments[:, 0:1, :]
    )
    return samples.reshape(-1, 2)


def symmetric_contour_chamfer(
    segments_2d: np.ndarray,
    target_points_2d: np.ndarray,
    *,
    samples_per_segment: int = 4,
) -> float:
    """Symmetric mean nearest-neighbour distance in plane coordinates."""

    samples = sample_plane_segments(
        segments_2d,
        samples_per_segment=samples_per_segment,
    )
    targets = np.asarray(target_points_2d, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 2 or not np.all(np.isfinite(targets)):
        raise ValueError("target_points_2d must have finite shape (points, 2)")
    if not len(samples) or not len(targets):
        raise ValueError("both predicted segments and target points must be non-empty")
    distances = np.linalg.norm(samples[:, None, :] - targets[None, :, :], axis=-1)
    return float(np.mean(np.min(distances, axis=1)) + np.mean(np.min(distances, axis=0)))
