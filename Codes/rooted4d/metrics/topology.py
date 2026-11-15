"""Dependency-light topology and self-intersection diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


def _surface_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("vertices must have finite shape (vertices, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("faces must have shape (faces, 3)")
    if len(triangles) and (triangles.min() < 0 or triangles.max() >= len(points)):
        raise ValueError("faces contain an out-of-range vertex index")
    return points, triangles


def edge_incidence_counts(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    triangles = np.asarray(faces, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("faces must have shape (faces, 3)")
    if not len(triangles):
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64)
    raw = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]],
        axis=0,
    )
    return np.unique(np.sort(raw, axis=1), axis=0, return_counts=True)


def surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    points, triangles = _surface_arrays(vertices, faces)
    if not len(triangles):
        return 0.0
    xyz = points[triangles]
    return float(
        0.5
        * np.sum(
            np.linalg.norm(
                np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]),
                axis=1,
            )
        )
    )


def signed_enclosed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Divergence-theorem volume for an oriented closed triangle surface."""

    points, triangles = _surface_arrays(vertices, faces)
    if not len(triangles):
        return 0.0
    xyz = points[triangles]
    return float(
        np.sum(np.einsum("ij,ij->i", xyz[:, 0], np.cross(xyz[:, 1], xyz[:, 2])))
        / 6.0
    )


def euler_characteristic(vertices: np.ndarray, faces: np.ndarray) -> int:
    points, triangles = _surface_arrays(vertices, faces)
    edges, _ = edge_incidence_counts(triangles)
    used_vertices = len(np.unique(triangles)) if len(triangles) else 0
    return int(used_vertices - len(edges) + len(triangles))


def _orient_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _point_on_segment_2d(
    point: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    tolerance: float,
) -> bool:
    if abs(_orient_2d(first, second, point)) > tolerance:
        return False
    return bool(
        np.all(point >= np.minimum(first, second) - tolerance)
        and np.all(point <= np.maximum(first, second) + tolerance)
    )


def _segments_intersect_2d(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    tolerance: float,
) -> bool:
    orientations = (
        _orient_2d(a, b, c),
        _orient_2d(a, b, d),
        _orient_2d(c, d, a),
        _orient_2d(c, d, b),
    )
    if orientations[0] * orientations[1] < -tolerance**2 and (
        orientations[2] * orientations[3] < -tolerance**2
    ):
        return True
    return any(
        (
            abs(value) <= tolerance
            and _point_on_segment_2d(point, first, second, tolerance)
        )
        for value, point, first, second in (
            (orientations[0], c, a, b),
            (orientations[1], d, a, b),
            (orientations[2], a, c, d),
            (orientations[3], b, c, d),
        )
    )


def _point_in_triangle_2d(
    point: np.ndarray,
    triangle: np.ndarray,
    tolerance: float,
) -> bool:
    signs = np.array(
        [
            _orient_2d(triangle[0], triangle[1], point),
            _orient_2d(triangle[1], triangle[2], point),
            _orient_2d(triangle[2], triangle[0], point),
        ]
    )
    return bool(np.all(signs >= -tolerance) or np.all(signs <= tolerance))


def _coplanar_triangles_intersect(
    first: np.ndarray,
    second: np.ndarray,
    normal: np.ndarray,
    tolerance: float,
) -> bool:
    drop_axis = int(np.argmax(np.abs(normal)))
    first_2d = np.delete(first, drop_axis, axis=1)
    second_2d = np.delete(second, drop_axis, axis=1)
    for edge_a in ((0, 1), (1, 2), (2, 0)):
        for edge_b in ((0, 1), (1, 2), (2, 0)):
            if _segments_intersect_2d(
                first_2d[edge_a[0]],
                first_2d[edge_a[1]],
                second_2d[edge_b[0]],
                second_2d[edge_b[1]],
                tolerance,
            ):
                return True
    return _point_in_triangle_2d(first_2d[0], second_2d, tolerance) or (
        _point_in_triangle_2d(second_2d[0], first_2d, tolerance)
    )


def triangles_intersect(
    first: np.ndarray,
    second: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> bool:
    """Return whether two non-degenerate closed triangles intersect.

    The coplanar case is projected to 2D explicitly.  This avoids the common
    failure of using only face normals and edge-cross-edge axes, which contain
    no in-plane separating direction for coplanar triangles.
    """

    first_triangle = np.asarray(first, dtype=np.float64)
    second_triangle = np.asarray(second, dtype=np.float64)
    if first_triangle.shape != (3, 3) or second_triangle.shape != (3, 3):
        raise ValueError("each triangle must have shape (3, 3)")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if not np.all(np.isfinite(first_triangle)) or not np.all(np.isfinite(second_triangle)):
        raise ValueError("triangle coordinates must be finite")
    first_edges = np.roll(first_triangle, -1, axis=0) - first_triangle
    second_edges = np.roll(second_triangle, -1, axis=0) - second_triangle
    first_normal = np.cross(first_edges[0], first_edges[1])
    second_normal = np.cross(second_edges[0], second_edges[1])
    first_norm = float(np.linalg.norm(first_normal))
    second_norm = float(np.linalg.norm(second_normal))
    if first_norm <= tolerance or second_norm <= tolerance:
        return False
    first_unit = first_normal / first_norm
    second_unit = second_normal / second_norm
    normals_parallel = np.linalg.norm(np.cross(first_unit, second_unit)) <= tolerance
    plane_distance = np.max(np.abs((second_triangle - first_triangle[0]) @ first_unit))
    if normals_parallel and plane_distance <= tolerance:
        return _coplanar_triangles_intersect(
            first_triangle,
            second_triangle,
            first_unit,
            tolerance,
        )

    axes = [first_normal, second_normal]
    axes.extend(np.cross(edge_a, edge_b) for edge_a in first_edges for edge_b in second_edges)
    for axis in axes:
        length = float(np.linalg.norm(axis))
        if length <= tolerance:
            continue
        direction = axis / length
        projection_first = first_triangle @ direction
        projection_second = second_triangle @ direction
        if projection_first.max() < projection_second.min() - tolerance or (
            projection_second.max() < projection_first.min() - tolerance
        ):
            return False
    return True


def count_self_intersections(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    cell_size: float | None = None,
    tolerance: float = 1e-10,
) -> int:
    """Count intersecting triangle pairs that do not share a mesh vertex.

    A uniform-grid broad phase keeps the reference implementation practical on
    moderate meshes.  Production meshes above roughly 200k faces should use a
    verified BVH implementation while retaining these regression cases.
    """

    points, triangles = _surface_arrays(vertices, faces)
    if len(triangles) < 2:
        return 0
    xyz = points[triangles]
    lower = xyz.min(axis=1)
    upper = xyz.max(axis=1)
    if cell_size is None:
        diameters = np.linalg.norm(upper - lower, axis=1)
        positive = diameters[diameters > tolerance]
        if not len(positive):
            return 0
        cell_size = 2.0 * float(np.median(positive))
    if not np.isfinite(cell_size) or cell_size <= 0.0:
        raise ValueError("cell_size must be positive and finite")

    buckets: dict[tuple[int, int, int], list[int]] = {}
    for face_index in range(len(triangles)):
        first_cell = np.floor(lower[face_index] / cell_size).astype(np.int64)
        last_cell = np.floor(upper[face_index] / cell_size).astype(np.int64)
        for x in range(first_cell[0], last_cell[0] + 1):
            for y in range(first_cell[1], last_cell[1] + 1):
                for z in range(first_cell[2], last_cell[2] + 1):
                    buckets.setdefault((x, y, z), []).append(face_index)

    checked: set[tuple[int, int]] = set()
    intersections = 0
    for candidates in buckets.values():
        for first_index, second_index in combinations(candidates, 2):
            pair = (
                (first_index, second_index)
                if first_index < second_index
                else (second_index, first_index)
            )
            if pair in checked:
                continue
            checked.add(pair)
            if np.intersect1d(triangles[pair[0]], triangles[pair[1]]).size:
                continue
            if np.any(lower[pair[0]] > upper[pair[1]] + tolerance) or np.any(
                lower[pair[1]] > upper[pair[0]] + tolerance
            ):
                continue
            if triangles_intersect(xyz[pair[0]], xyz[pair[1]], tolerance=tolerance):
                intersections += 1
    return intersections


@dataclass(frozen=True)
class SurfaceTopologyReport:
    n_vertices: int
    n_faces: int
    n_edges: int
    boundary_edges: int
    nonmanifold_edges: int
    euler_characteristic: int
    minimum_triangle_area: float
    surface_area: float
    signed_enclosed_volume: float | None
    self_intersections: int

    @property
    def edge_manifold(self) -> bool:
        return self.nonmanifold_edges == 0

    @property
    def closed(self) -> bool:
        return (
            self.n_faces > 0
            and self.boundary_edges == 0
            and self.nonmanifold_edges == 0
        )

    def as_dict(self) -> dict[str, int | float | bool | None]:
        return {
            "n_vertices": self.n_vertices,
            "n_faces": self.n_faces,
            "n_edges": self.n_edges,
            "boundary_edges": self.boundary_edges,
            "nonmanifold_edges": self.nonmanifold_edges,
            "edge_manifold": self.edge_manifold,
            "closed": self.closed,
            "euler_characteristic": self.euler_characteristic,
            "minimum_triangle_area": self.minimum_triangle_area,
            "surface_area": self.surface_area,
            "signed_enclosed_volume": self.signed_enclosed_volume,
            "self_intersections": self.self_intersections,
        }


def surface_topology_report(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    compute_self_intersections: bool = True,
) -> SurfaceTopologyReport:
    """Report topology without treating an intentionally open leaflet as closed."""

    points, triangles = _surface_arrays(vertices, faces)
    edges, counts = edge_incidence_counts(triangles)
    boundary_edges = int(np.count_nonzero(counts == 1))
    nonmanifold_edges = int(np.count_nonzero(counts > 2))
    if len(triangles):
        xyz = points[triangles]
        areas = 0.5 * np.linalg.norm(
            np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]),
            axis=1,
        )
        minimum_area = float(np.min(areas))
    else:
        minimum_area = 0.0
    closed = boundary_edges == 0 and nonmanifold_edges == 0 and len(triangles) > 0
    return SurfaceTopologyReport(
        n_vertices=len(points),
        n_faces=len(triangles),
        n_edges=len(edges),
        boundary_edges=boundary_edges,
        nonmanifold_edges=nonmanifold_edges,
        euler_characteristic=euler_characteristic(points, triangles),
        minimum_triangle_area=minimum_area,
        surface_area=surface_area(points, triangles),
        signed_enclosed_volume=(
            signed_enclosed_volume(points, triangles) if closed else None
        ),
        self_intersections=(
            count_self_intersections(points, triangles) if compute_self_intersections else -1
        ),
    )


def component_overlap_volumes(
    masks: dict[str, np.ndarray],
    *,
    voxel_volume: float,
) -> dict[str, float]:
    """Return pairwise and total overlap volumes for baseline occupancy masks."""

    if not masks:
        raise ValueError("masks must be non-empty")
    if not np.isfinite(voxel_volume) or voxel_volume <= 0.0:
        raise ValueError("voxel_volume must be positive and finite")
    arrays = {name: np.asarray(mask, dtype=bool) for name, mask in masks.items()}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("all component masks must have the same shape")
    result: dict[str, float] = {}
    total = 0.0
    for first, second in combinations(sorted(arrays), 2):
        overlap = float(np.count_nonzero(arrays[first] & arrays[second])) * voxel_volume
        result[f"{first}|{second}"] = overlap
        total += overlap
    result["total_overlap_volume"] = total
    return result
