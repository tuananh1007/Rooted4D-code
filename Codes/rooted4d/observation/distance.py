"""Dependency-light, exact unsigned point-to-triangle distance queries."""

from __future__ import annotations

import numpy as np


def _validated_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("vertices must have finite shape (vertices, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise ValueError("faces must have non-empty shape (faces, 3)")
    if triangles.min() < 0 or triangles.max() >= len(points):
        raise ValueError("faces contain an out-of-range vertex index")
    xyz = points[triangles]
    doubled_area = np.linalg.norm(
        np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
    )
    if np.any(doubled_area <= 1e-12):
        raise ValueError("faces contain a degenerate triangle")
    return points, triangles


def _triangle_distance_squared(query: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Vectorized closest distance to one non-degenerate triangle.

    The edge/plane classification is the exact unsigned triangle-distance
    construction, not a nearest-vertex approximation.
    """

    a, b, c = triangle
    ab = b - a
    ac = c - a
    ap = query - a
    d1 = ap @ ab
    d2 = ap @ ac
    bp = query - b
    d3 = bp @ ab
    d4 = bp @ ac
    cp = query - c
    d5 = cp @ ab
    d6 = cp @ ac
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    closest = np.empty_like(query)
    assigned = np.zeros(len(query), dtype=bool)

    def assign(mask: np.ndarray, values: np.ndarray) -> None:
        selected = mask & ~assigned
        closest[selected] = values[selected] if values.ndim == 2 else values
        assigned[selected] = True

    assign((d1 <= 0.0) & (d2 <= 0.0), np.broadcast_to(a, query.shape))
    assign((d3 >= 0.0) & (d4 <= d3), np.broadcast_to(b, query.shape))
    edge_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    parameter_ab = d1 / np.maximum(d1 - d3, 1e-30)
    assign(edge_ab, a + parameter_ab[:, None] * ab)
    assign((d6 >= 0.0) & (d5 <= d6), np.broadcast_to(c, query.shape))
    edge_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    parameter_ac = d2 / np.maximum(d2 - d6, 1e-30)
    assign(edge_ac, a + parameter_ac[:, None] * ac)
    edge_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    parameter_bc = (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), 1e-30)
    assign(edge_bc, b + parameter_bc[:, None] * (c - b))

    face = ~assigned
    denominator = np.maximum(va + vb + vc, 1e-30)
    barycentric_b = vb / denominator
    barycentric_c = vc / denominator
    closest[face] = (
        a
        + barycentric_b[face, None] * ab
        + barycentric_c[face, None] * ac
    )
    residual = query - closest
    return np.sum(residual * residual, axis=1)


def point_to_mesh_distance(
    query_points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Return exact unsigned distances from arbitrary query points to a triangle mesh."""

    query = np.asarray(query_points, dtype=np.float64)
    if query.shape[-1:] != (3,) or not np.all(np.isfinite(query)):
        raise ValueError("query_points must have finite trailing coordinate dimension 3")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    mesh_vertices, mesh_faces = _validated_mesh(vertices, faces)
    original_shape = query.shape[:-1]
    flattened = query.reshape(-1, 3)
    result = np.empty(len(flattened), dtype=np.float64)
    mesh_triangles = mesh_vertices[mesh_faces]
    for start in range(0, len(flattened), chunk_size):
        stop = min(start + chunk_size, len(flattened))
        points = flattened[start:stop]
        minimum = np.full(len(points), np.inf, dtype=np.float64)
        for triangle in mesh_triangles:
            minimum = np.minimum(minimum, _triangle_distance_squared(points, triangle))
        result[start:stop] = np.sqrt(np.maximum(minimum, 0.0))
    return result.reshape(original_shape)


def surface_likelihood(
    query_points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    sigma_mm: float,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Convert exact point-to-mesh distance into a Gaussian surface likelihood."""

    if sigma_mm <= 0.0:
        raise ValueError("sigma_mm must be positive")
    distance = point_to_mesh_distance(
        query_points, vertices, faces, chunk_size=chunk_size
    )
    return np.exp(-0.5 * (distance / sigma_mm) ** 2)
