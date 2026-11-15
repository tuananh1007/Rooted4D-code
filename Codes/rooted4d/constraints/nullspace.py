"""General linear equality constraints decoded exactly through a null space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class NullspaceParameterization:
    """Represent every feasible vector as ``particular + basis @ latent``."""

    constraint: np.ndarray
    target: np.ndarray
    particular: np.ndarray
    basis: np.ndarray
    rank: int

    @classmethod
    def from_constraints(
        cls,
        constraint: np.ndarray,
        target: np.ndarray | None = None,
        rtol: float | None = None,
        consistency_atol: float = 1e-9,
    ) -> "NullspaceParameterization":
        matrix = np.asarray(constraint, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("constraint must be a two-dimensional matrix")
        rows, columns = matrix.shape
        rhs = np.zeros(rows, dtype=np.float64) if target is None else np.asarray(target, dtype=np.float64)
        if rhs.shape != (rows,):
            raise ValueError(f"target must have shape ({rows},)")

        _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
        if singular_values.size == 0:
            rank = 0
        else:
            tolerance = (
                rtol * singular_values[0]
                if rtol is not None
                else max(matrix.shape) * np.finfo(matrix.dtype).eps * singular_values[0]
            )
            rank = int(np.sum(singular_values > tolerance))
        basis = vh[rank:].T.copy()
        particular, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
        residual = matrix @ particular - rhs
        if np.linalg.norm(residual, ord=np.inf) > consistency_atol:
            raise ValueError("The linear constraints are inconsistent")
        if basis.size and np.linalg.norm(matrix @ basis, ord=np.inf) > 100 * consistency_atol:
            raise RuntimeError("Computed null-space basis failed the constraint check")
        return cls(
            constraint=matrix,
            target=rhs,
            particular=particular,
            basis=basis,
            rank=rank,
        )

    @property
    def ambient_dimension(self) -> int:
        return self.constraint.shape[1]

    @property
    def latent_dimension(self) -> int:
        return self.basis.shape[1]

    def decode(self, latent: np.ndarray) -> np.ndarray:
        values = np.asarray(latent, dtype=np.float64)
        if values.shape[-1] != self.latent_dimension:
            raise ValueError(
                f"Last latent dimension must be {self.latent_dimension}; received {values.shape}"
            )
        return self.particular + values @ self.basis.T

    def encode(self, feasible: np.ndarray) -> np.ndarray:
        values = np.asarray(feasible, dtype=np.float64)
        if values.shape[-1] != self.ambient_dimension:
            raise ValueError(
                f"Last ambient dimension must be {self.ambient_dimension}; received {values.shape}"
            )
        return (values - self.particular) @ self.basis

    def project(self, values: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(values))

    def residual(self, feasible: np.ndarray) -> np.ndarray:
        values = np.asarray(feasible, dtype=np.float64)
        return values @ self.constraint.T - self.target


def constraint_matrix_from_equalities(
    n_vertices: int,
    equal_vertex_pairs: Iterable[tuple[int, int]],
) -> np.ndarray:
    """Build ``Cx=0`` for 3D equality of each duplicated vertex pair."""

    if n_vertices <= 0:
        raise ValueError("n_vertices must be positive")
    pairs = list(equal_vertex_pairs)
    matrix = np.zeros((3 * len(pairs), 3 * n_vertices), dtype=np.float64)
    for pair_index, (left, right) in enumerate(pairs):
        if left == right:
            raise ValueError("An equality pair must contain two distinct vertices")
        if min(left, right) < 0 or max(left, right) >= n_vertices:
            raise ValueError(f"Out-of-range equality pair: {(left, right)}")
        for coordinate in range(3):
            row = 3 * pair_index + coordinate
            matrix[row, 3 * left + coordinate] = 1.0
            matrix[row, 3 * right + coordinate] = -1.0
    return matrix

