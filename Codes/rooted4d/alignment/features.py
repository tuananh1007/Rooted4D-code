"""Geometric features for initializing patient-to-template correspondence.

These NumPy implementations are executable references for the later PyTorch
pipeline.  They deliberately keep the fixed assembly operator ``A`` separate
from patient-dependent correspondence weights ``Pi`` and barycentric projection
``P(Pi)``.
"""

from __future__ import annotations

import numpy as np


def _points(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (items, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _unit_vectors(values: np.ndarray, name: str, eps: float = 1e-12) -> np.ndarray:
    array = _points(values, name)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError(f"{name} contains an undefined zero-length vector")
    return array / norms


def area_weighted_vertex_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return oriented vertex normals accumulated from area-weighted faces.

    Call this separately for every local manifold.  In particular, do not
    average LV, LA, and leaflet normals at physical annular vertices even when
    those vertices share coordinates through the assembly map.
    """

    points = _points(vertices, "vertices")
    triangles = np.asarray(faces, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise ValueError("faces must have non-empty shape (faces, 3)")
    if triangles.min() < 0 or triangles.max() >= len(points):
        raise ValueError("faces contain an out-of-range vertex index")
    face_cross = np.cross(
        points[triangles[:, 1]] - points[triangles[:, 0]],
        points[triangles[:, 2]] - points[triangles[:, 0]],
    )
    accumulated = np.zeros_like(points)
    for corner in range(3):
        np.add.at(accumulated, triangles[:, corner], face_cross)
    norms = np.linalg.norm(accumulated, axis=1, keepdims=True)
    if np.any(norms <= eps):
        undefined = np.flatnonzero(norms[:, 0] <= eps).tolist()
        raise ValueError(f"Undefined normals at vertices {undefined}")
    return accumulated / norms


def patch_admissibility_mask(
    source_patch_labels: np.ndarray,
    target_patch_labels: np.ndarray,
) -> np.ndarray:
    """Allow correspondence only between samples with the same anatomy label."""

    source = np.asarray(source_patch_labels)
    target = np.asarray(target_patch_labels)
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("patch labels must be one-dimensional")
    if source.size == 0 or target.size == 0:
        raise ValueError("patch labels must be non-empty")
    return source[:, None] == target[None, :]


def soft_correspondence_weights(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    source_normals: np.ndarray | None = None,
    target_normals: np.ndarray | None = None,
    admissible_mask: np.ndarray | None = None,
    distance_scale: float = 5.0,
    normal_weight: float = 1.0,
    temperature: float = 1.0,
) -> np.ndarray:
    """Compute row-stochastic, optionally patch-restricted soft assignments.

    This is a dependency-light reference rather than a replacement for a
    production point-to-triangle or entropic optimal-transport implementation.
    Normal disagreement is additive to position cost; it never multiplies the
    positional residual.
    """

    source = _points(source_points, "source_points")
    target = _points(target_points, "target_points")
    if distance_scale <= 0.0 or temperature <= 0.0:
        raise ValueError("distance_scale and temperature must be positive")
    if normal_weight < 0.0:
        raise ValueError("normal_weight must be non-negative")

    differences = source[:, None, :] - target[None, :, :]
    cost = np.sum(differences * differences, axis=-1) / (distance_scale**2)
    if (source_normals is None) != (target_normals is None):
        raise ValueError("source_normals and target_normals must be supplied together")
    if source_normals is not None and target_normals is not None:
        left = _unit_vectors(source_normals, "source_normals")
        right = _unit_vectors(target_normals, "target_normals")
        if len(left) != len(source) or len(right) != len(target):
            raise ValueError("normal counts must match their point counts")
        dots = np.clip(left @ right.T, -1.0, 1.0)
        cost += normal_weight * 0.5 * (1.0 - dots)

    if admissible_mask is None:
        mask = np.ones(cost.shape, dtype=bool)
    else:
        mask = np.asarray(admissible_mask, dtype=bool)
        if mask.shape != cost.shape:
            raise ValueError(f"admissible_mask must have shape {cost.shape}")
    if np.any(~np.any(mask, axis=1)):
        raise ValueError("Every source sample must have at least one admissible target")

    masked_cost = np.where(mask, cost, np.inf)
    row_minimum = np.min(masked_cost, axis=1, keepdims=True)
    unnormalized = np.where(
        mask,
        np.exp(-(masked_cost - row_minimum) / temperature),
        0.0,
    )
    return unnormalized / np.sum(unnormalized, axis=1, keepdims=True)


def normal_compatibility_loss(
    source_normals: np.ndarray,
    target_normals: np.ndarray,
    correspondence: np.ndarray,
) -> float:
    """Mass-normalized oriented-normal disagreement in the range [0, 1]."""

    source = _unit_vectors(source_normals, "source_normals")
    target = _unit_vectors(target_normals, "target_normals")
    mass = np.asarray(correspondence, dtype=np.float64)
    expected = (len(source), len(target))
    if mass.shape != expected:
        raise ValueError(f"correspondence must have shape {expected}")
    if np.any(mass < 0.0) or not np.all(np.isfinite(mass)):
        raise ValueError("correspondence must contain finite non-negative mass")
    total = float(np.sum(mass))
    if total <= 0.0:
        raise ValueError("correspondence must contain positive mass")
    dots = np.clip(source @ target.T, -1.0, 1.0)
    return float(np.sum(mass * 0.5 * (1.0 - dots)) / total)


def directed_arc_coordinates(
    curve_vertices: np.ndarray,
    *,
    anchor_index: int = 0,
    orientation: int = 1,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return cumulative coordinates on an oriented cyclic ring in [0, 1).

    The ring must contain each physical vertex once; do not repeat the first
    vertex at the end.  ``orientation`` is fixed by anatomical landmarks, not
    selected independently for each patient.
    """

    curve = _points(curve_vertices, "curve_vertices")
    n_vertices = len(curve)
    if n_vertices < 3:
        raise ValueError("A cyclic curve requires at least three vertices")
    if anchor_index < 0 or anchor_index >= n_vertices:
        raise ValueError("anchor_index is out of range")
    if orientation not in (-1, 1):
        raise ValueError("orientation must be +1 or -1")
    if np.linalg.norm(curve[0] - curve[-1]) <= eps:
        raise ValueError("Represent the cyclic ring without a duplicated endpoint")

    order = [(anchor_index + orientation * step) % n_vertices for step in range(n_vertices)]
    cumulative = np.zeros(n_vertices, dtype=np.float64)
    traversed = 0.0
    for step in range(1, n_vertices):
        previous = order[step - 1]
        current = order[step]
        traversed += float(np.linalg.norm(curve[current] - curve[previous]))
        cumulative[current] = traversed
    closing = float(np.linalg.norm(curve[order[-1]] - curve[order[0]]))
    total = traversed + closing
    if total <= eps:
        raise ValueError("The cyclic curve has zero total length")
    return cumulative / total


def circular_coordinate_residual(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Signed shortest residual between normalized coordinates on S1."""

    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    difference = first - second
    return np.arctan2(
        np.sin(2.0 * np.pi * difference),
        np.cos(2.0 * np.pi * difference),
    ) / (2.0 * np.pi)


def directed_arc_alignment_loss(
    template_coordinates: np.ndarray,
    target_coordinates: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Weighted squared circular error for known annular landmark/arc matches."""

    template = np.asarray(template_coordinates, dtype=np.float64)
    target = np.asarray(target_coordinates, dtype=np.float64)
    if template.shape != target.shape or template.ndim != 1 or template.size == 0:
        raise ValueError("template_coordinates and target_coordinates need equal 1D shape")
    residual = circular_coordinate_residual(template, target)
    if weights is None:
        mass = np.ones_like(residual)
    else:
        mass = np.asarray(weights, dtype=np.float64)
        if mass.shape != residual.shape or np.any(mass < 0.0):
            raise ValueError("weights must be non-negative with matching shape")
    total = float(np.sum(mass))
    if total <= 0.0:
        raise ValueError("weights must contain positive mass")
    return float(np.sum(mass * residual * residual) / total)


def barycentric_projection_matrix(
    simplex_vertex_indices: np.ndarray,
    barycentric_weights: np.ndarray,
    n_target_vertices: int,
    atol: float = 1e-8,
) -> np.ndarray:
    """Build a dense reference form of patient-specific projection ``P_i``.

    Production code should store this operator sparsely.  It is deliberately
    distinct from the fixed binary assembly map ``A``.
    """

    indices = np.asarray(simplex_vertex_indices, dtype=np.int64)
    weights = np.asarray(barycentric_weights, dtype=np.float64)
    if n_target_vertices <= 0:
        raise ValueError("n_target_vertices must be positive")
    if indices.ndim != 2 or indices.shape != weights.shape or indices.shape[1] < 2:
        raise ValueError("indices and weights need equal shape (samples, simplex_vertices)")
    if indices.size == 0 or indices.min() < 0 or indices.max() >= n_target_vertices:
        raise ValueError("simplex indices are empty or out of range")
    if np.any(weights < -atol) or not np.all(np.isfinite(weights)):
        raise ValueError("barycentric weights must be finite and non-negative")
    if not np.allclose(np.sum(weights, axis=1), 1.0, atol=atol):
        raise ValueError("Each barycentric row must sum to one")
    projection = np.zeros((len(indices), n_target_vertices), dtype=np.float64)
    rows = np.arange(len(indices))
    for corner in range(indices.shape[1]):
        np.add.at(projection, (rows, indices[:, corner]), weights[:, corner])
    return projection
