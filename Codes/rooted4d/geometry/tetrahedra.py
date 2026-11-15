"""Tetrahedral validity and deformation diagnostics.

Rooted4D uses a fixed tetrahedral LV myocardial shell.  These routines operate
on that production topology; they do not generate a new lattice or change
connectivity.  The barrier is an interior-point reference implementation: it
is finite only while every deformation Jacobian stays strictly above the
declared minimum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _mesh_arrays(
    vertices: np.ndarray,
    tetrahedra: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(vertices, dtype=np.float64)
    cells = np.asarray(tetrahedra, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("vertices must have finite shape (vertices, 3)")
    if cells.ndim != 2 or cells.shape[1] != 4 or len(cells) == 0:
        raise ValueError("tetrahedra must have non-empty shape (tetrahedra, 4)")
    if cells.min() < 0 or cells.max() >= len(points):
        raise ValueError("tetrahedra contain an out-of-range vertex index")
    return points, cells


def signed_tetrahedron_volumes(
    vertices: np.ndarray,
    tetrahedra: np.ndarray,
) -> np.ndarray:
    """Return signed physical volume for every tetrahedron."""

    points, cells = _mesh_arrays(vertices, tetrahedra)
    simplex = points[cells]
    six_volume = np.einsum(
        "ij,ij->i",
        simplex[:, 1] - simplex[:, 0],
        np.cross(simplex[:, 2] - simplex[:, 0], simplex[:, 3] - simplex[:, 0]),
    )
    return six_volume / 6.0


def tetrahedron_scaled_jacobians(
    vertices: np.ndarray,
    tetrahedra: np.ndarray,
    *,
    epsilon: float = 1e-15,
) -> np.ndarray:
    """Return the minimum corner-scaled Jacobian of each tetrahedron.

    A regular tetrahedron scores one, a degenerate tetrahedron scores zero,
    and an inverted tetrahedron has a negative score.  The corner order keeps
    all four triple products positive for a consistently oriented cell.
    """

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    points, cells = _mesh_arrays(vertices, tetrahedra)
    corner_orders = (
        (0, 1, 2, 3),
        (1, 0, 3, 2),
        (2, 0, 1, 3),
        (3, 0, 2, 1),
    )
    values = []
    for anchor, first, second, third in corner_orders:
        origin = points[cells[:, anchor]]
        edge_1 = points[cells[:, first]] - origin
        edge_2 = points[cells[:, second]] - origin
        edge_3 = points[cells[:, third]] - origin
        triple = np.einsum("ij,ij->i", np.cross(edge_1, edge_2), edge_3)
        denominator = (
            np.linalg.norm(edge_1, axis=1)
            * np.linalg.norm(edge_2, axis=1)
            * np.linalg.norm(edge_3, axis=1)
        )
        values.append(np.sqrt(2.0) * triple / np.maximum(denominator, epsilon))
    return np.min(np.stack(values, axis=1), axis=1)


def deformation_jacobian_determinants(
    reference_vertices: np.ndarray,
    deformed_vertices: np.ndarray,
    tetrahedra: np.ndarray,
    *,
    minimum_reference_volume: float = 1e-12,
) -> np.ndarray:
    """Return ``det(F)`` for the piecewise-affine tetrahedral deformation."""

    reference, cells = _mesh_arrays(reference_vertices, tetrahedra)
    deformed, deformed_cells = _mesh_arrays(deformed_vertices, tetrahedra)
    if len(reference) != len(deformed) or not np.array_equal(cells, deformed_cells):
        raise ValueError("reference and deformed meshes must share vertices and topology")
    if minimum_reference_volume <= 0.0:
        raise ValueError("minimum_reference_volume must be positive")
    reference_volume = signed_tetrahedron_volumes(reference, cells)
    if np.any(reference_volume <= minimum_reference_volume):
        raise ValueError("reference tetrahedra must be positively oriented and non-degenerate")
    deformed_volume = signed_tetrahedron_volumes(deformed, cells)
    return deformed_volume / reference_volume


def interior_log_barrier(
    values: np.ndarray,
    *,
    minimum: float,
    activation: float,
) -> float:
    r"""Evaluate a truncated singular interior barrier.

    For a scalar validity measure :math:`v`, let

    .. math::
       r=(v-v_{min})/(v_{act}-v_{min}).

    The contribution is ``-(r-1)^2 log(r)`` for ``0 < r < 1`` and zero
    above the activation value.  It tends to positive infinity as the
    feasibility boundary is approached and returns ``inf`` outside it.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("values must be non-empty and finite")
    if not np.isfinite(minimum) or not np.isfinite(activation) or activation <= minimum:
        raise ValueError("activation must be finite and strictly greater than minimum")
    gap = array - minimum
    if np.any(gap <= 0.0):
        return float("inf")
    ratio = gap / (activation - minimum)
    active = ratio < 1.0
    if not np.any(active):
        return 0.0
    selected = ratio[active]
    return float(np.sum(-((selected - 1.0) ** 2) * np.log(selected)))


def tetrahedron_jacobian_barrier(
    reference_vertices: np.ndarray,
    deformed_vertices: np.ndarray,
    tetrahedra: np.ndarray,
    *,
    minimum_determinant: float = 0.05,
    activation_determinant: float = 0.20,
) -> float:
    """Barrier against collapse or inversion of the reference tet mesh."""

    determinants = deformation_jacobian_determinants(
        reference_vertices,
        deformed_vertices,
        tetrahedra,
    )
    return interior_log_barrier(
        determinants,
        minimum=minimum_determinant,
        activation=activation_determinant,
    )


@dataclass(frozen=True)
class TetrahedronQualityReport:
    n_tetrahedra: int
    inverted_or_degenerate: int
    minimum_signed_volume: float
    total_signed_volume: float
    minimum_scaled_jacobian: float
    mean_scaled_jacobian: float
    fraction_scaled_jacobian_below_threshold: float
    scaled_jacobian_threshold: float

    @property
    def valid(self) -> bool:
        return self.inverted_or_degenerate == 0

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "n_tetrahedra": self.n_tetrahedra,
            "inverted_or_degenerate": self.inverted_or_degenerate,
            "minimum_signed_volume": self.minimum_signed_volume,
            "total_signed_volume": self.total_signed_volume,
            "minimum_scaled_jacobian": self.minimum_scaled_jacobian,
            "mean_scaled_jacobian": self.mean_scaled_jacobian,
            "fraction_scaled_jacobian_below_threshold": (
                self.fraction_scaled_jacobian_below_threshold
            ),
            "scaled_jacobian_threshold": self.scaled_jacobian_threshold,
            "valid": self.valid,
        }


def tetrahedron_quality_report(
    vertices: np.ndarray,
    tetrahedra: np.ndarray,
    *,
    scaled_jacobian_threshold: float = 0.10,
) -> TetrahedronQualityReport:
    """Summarize element orientation and shape quality without hiding failures."""

    if not 0.0 < scaled_jacobian_threshold < 1.0:
        raise ValueError("scaled_jacobian_threshold must lie in (0, 1)")
    volumes = signed_tetrahedron_volumes(vertices, tetrahedra)
    scaled = tetrahedron_scaled_jacobians(vertices, tetrahedra)
    return TetrahedronQualityReport(
        n_tetrahedra=len(volumes),
        inverted_or_degenerate=int(np.count_nonzero(volumes <= 0.0)),
        minimum_signed_volume=float(np.min(volumes)),
        total_signed_volume=float(np.sum(volumes)),
        minimum_scaled_jacobian=float(np.min(scaled)),
        mean_scaled_jacobian=float(np.mean(scaled)),
        fraction_scaled_jacobian_below_threshold=float(
            np.mean(scaled < scaled_jacobian_threshold)
        ),
        scaled_jacobian_threshold=float(scaled_jacobian_threshold),
    )
