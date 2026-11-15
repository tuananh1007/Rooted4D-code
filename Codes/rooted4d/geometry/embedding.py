"""Barycentric embedding for optional papillary-tip landmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BarycentricEmbedding:
    """Embed landmarks in a parent triangle/tetrahedral deformation field.

    Papillary tips should normally use four weights in a containing LV
    tetrahedron.  This makes them follow the myocardium without welding them to
    one endocardial vertex or allowing unrestricted free-floating motion.
    """

    simplex_vertex_indices: np.ndarray
    barycentric_weights: np.ndarray
    n_parent_vertices: int

    def __post_init__(self) -> None:
        indices = np.asarray(self.simplex_vertex_indices, dtype=np.int64)
        weights = np.asarray(self.barycentric_weights, dtype=np.float64)
        if self.n_parent_vertices <= 0:
            raise ValueError("n_parent_vertices must be positive")
        if indices.ndim != 2 or indices.shape != weights.shape or indices.shape[1] < 2:
            raise ValueError("indices and weights need equal shape (landmarks, simplex_vertices)")
        if indices.size == 0 or indices.min() < 0 or indices.max() >= self.n_parent_vertices:
            raise ValueError("simplex indices are empty or out of range")
        if np.any(weights < -1e-8) or not np.all(np.isfinite(weights)):
            raise ValueError("barycentric weights must be finite and non-negative")
        if not np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-8):
            raise ValueError("Each barycentric row must sum to one")
        object.__setattr__(self, "simplex_vertex_indices", indices)
        object.__setattr__(self, "barycentric_weights", weights)

    @property
    def n_landmarks(self) -> int:
        return len(self.simplex_vertex_indices)

    def evaluate(
        self,
        parent_vertices: np.ndarray,
        residual: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate embedded coordinates for arrays shaped (..., vertices, 3)."""

        parent = np.asarray(parent_vertices, dtype=np.float64)
        if parent.ndim < 2 or parent.shape[-2:] != (self.n_parent_vertices, 3):
            raise ValueError(
                "parent_vertices must have shape (..., n_parent_vertices, 3)"
            )
        gathered = parent[..., self.simplex_vertex_indices, :]
        lead_dimensions = parent.ndim - 2
        weight_shape = (1,) * lead_dimensions + self.barycentric_weights.shape + (1,)
        embedded = np.sum(
            gathered * self.barycentric_weights.reshape(weight_shape),
            axis=-2,
        )
        if residual is None:
            return embedded
        offset = np.asarray(residual, dtype=np.float64)
        if offset.ndim < 2 or offset.shape[-2:] != (self.n_landmarks, 3):
            raise ValueError("residual must have shape (..., n_landmarks, 3)")
        try:
            return embedded + offset
        except ValueError as error:
            raise ValueError("residual leading dimensions are not broadcast-compatible") from error


def bounded_residual(raw_residual: np.ndarray, maximum_norm: float) -> np.ndarray:
    """Smoothly bound landmark residual norms using radial tanh scaling."""

    raw = np.asarray(raw_residual, dtype=np.float64)
    if raw.ndim < 1 or raw.shape[-1] != 3:
        raise ValueError("raw_residual must have shape (..., 3)")
    if maximum_norm <= 0.0:
        raise ValueError("maximum_norm must be positive")
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    safe = np.maximum(norms, np.finfo(np.float64).eps)
    scale = maximum_norm * np.tanh(norms / maximum_norm) / safe
    return raw * scale


def fit_tetrahedral_embedding(
    points: np.ndarray,
    parent_vertices: np.ndarray,
    tetrahedra: np.ndarray,
    host_tetrahedron_indices: np.ndarray,
    *,
    require_inside: bool = True,
    tolerance: float = 1e-8,
) -> BarycentricEmbedding:
    """Fit fixed barycentric weights for points with declared host tetrahedra.

    Host selection is deliberately explicit: production code must determine it
    from the frozen LV tetrahedral mesh and then preserve it.  This function
    computes and validates weights; it does not perform an ambiguous global
    nearest-tetrahedron search.
    """

    samples = np.asarray(points, dtype=np.float64)
    parents = np.asarray(parent_vertices, dtype=np.float64)
    cells = np.asarray(tetrahedra, dtype=np.int64)
    hosts = np.asarray(host_tetrahedron_indices, dtype=np.int64)
    if samples.ndim != 2 or samples.shape[1] != 3 or not np.all(np.isfinite(samples)):
        raise ValueError("points must have finite shape (points, 3)")
    if parents.ndim != 2 or parents.shape[1] != 3 or not np.all(np.isfinite(parents)):
        raise ValueError("parent_vertices must have finite shape (vertices, 3)")
    if cells.ndim != 2 or cells.shape[1] != 4 or len(cells) == 0:
        raise ValueError("tetrahedra must have non-empty shape (tetrahedra, 4)")
    if cells.min() < 0 or cells.max() >= len(parents):
        raise ValueError("tetrahedra contain an out-of-range vertex index")
    invalid_hosts = len(hosts) and (
        hosts.min() < 0 or hosts.max() >= len(cells)
    )
    if hosts.shape != (len(samples),) or invalid_hosts:
        raise ValueError("host_tetrahedron_indices must select one valid host per point")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    simplex_indices = cells[hosts]
    simplex = parents[simplex_indices]
    matrices = np.stack(
        [
            simplex[:, 1] - simplex[:, 0],
            simplex[:, 2] - simplex[:, 0],
            simplex[:, 3] - simplex[:, 0],
        ],
        axis=-1,
    )
    determinants = np.linalg.det(matrices)
    if np.any(np.abs(determinants) <= tolerance):
        raise ValueError("a selected host tetrahedron is degenerate")
    coordinates = np.linalg.solve(matrices, (samples - simplex[:, 0])[..., None])[..., 0]
    weights = np.concatenate(
        [1.0 - np.sum(coordinates, axis=1, keepdims=True), coordinates],
        axis=1,
    )
    if require_inside and (
        np.any(weights < -tolerance) or np.any(weights > 1.0 + tolerance)
    ):
        raise ValueError("a point lies outside its declared host tetrahedron")
    if require_inside:
        weights = np.clip(weights, 0.0, 1.0)
        weights /= np.sum(weights, axis=1, keepdims=True)
    return BarycentricEmbedding(
        simplex_vertex_indices=simplex_indices,
        barycentric_weights=weights,
        n_parent_vertices=len(parents),
    )
