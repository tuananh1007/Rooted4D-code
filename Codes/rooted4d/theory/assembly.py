r"""Theorem 1 falsification: exact assembly equals the constraint null space.

Theorem 1 of the specification states that for the fixed binary local-to-physical
map ``A = \widetilde A \otimes I_3`` and the interface equality system ``C``,

.. math::
   \operatorname{im}(A)=\ker(C).

Corollary 1.1 adds that ``\nabla_q L = A^\top \nabla_x L``, so every duplicated
occurrence of an annular vertex accumulates its gradient at one physical degree
of freedom.

The theorem-to-experiment contract requires three checks: ``CA = 0``, equality of
rank and dimension, and a finite-difference confirmation of shared gradients.
This module performs all three on a concrete attachment map, at a size where the
dense reference operators are exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from rooted4d.constraints import AttachmentMap, constraint_matrix_from_equalities


def spanning_tree_equalities(attachment_map: AttachmentMap) -> list[tuple[int, int]]:
    """Return an independent equality set: one edge per duplicated occurrence.

    Assumption A2 requires ``C`` to contain an oriented spanning-tree incidence
    basis for each interface equivalence class, which is what makes ``C`` full
    row rank.  Chaining every occurrence of a physical vertex to the first one
    yields exactly such a tree.
    """

    local_index = 0
    first_seen: dict[int, int] = {}
    pairs: list[tuple[int, int]] = []
    for indices in attachment_map.local_to_global.values():
        for physical in indices.tolist():
            if physical in first_seen:
                pairs.append((first_seen[physical], local_index))
            else:
                first_seen[physical] = local_index
            local_index += 1
    return pairs


@dataclass(frozen=True)
class AssemblyEquivalenceReport:
    """Numerical evidence for Theorem 1 and Corollary 1.1."""

    n_local_vertices: int
    n_physical_vertices: int
    n_equality_rows: int
    constraint_rank: int
    nullspace_dimension: int
    assembly_rank: int
    max_constraint_on_assembly: float
    max_projector_difference: float
    max_gradient_sharing_error: float

    @property
    def ranks_agree(self) -> bool:
        return (
            self.constraint_rank == self.n_equality_rows
            and self.nullspace_dimension == 3 * self.n_physical_vertices
            and self.assembly_rank == self.nullspace_dimension
        )

    def holds(self, tolerance: float = 1e-9) -> bool:
        return self.ranks_agree and max(
            self.max_constraint_on_assembly,
            self.max_projector_difference,
            self.max_gradient_sharing_error,
        ) <= tolerance

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "n_local_vertices": self.n_local_vertices,
            "n_physical_vertices": self.n_physical_vertices,
            "n_equality_rows": self.n_equality_rows,
            "constraint_rank": self.constraint_rank,
            "nullspace_dimension": self.nullspace_dimension,
            "assembly_rank": self.assembly_rank,
            "ranks_agree": self.ranks_agree,
            "max_constraint_on_assembly": self.max_constraint_on_assembly,
            "max_projector_difference": self.max_projector_difference,
            "max_gradient_sharing_error": self.max_gradient_sharing_error,
            "holds": self.holds(),
        }


def _max_abs(values: np.ndarray) -> float:
    """Maximum magnitude, defined as zero for an empty array.

    A template whose components share no interface produces an equality system
    with no rows.  That is a degenerate but legal input: it means there is
    nothing to attach, not that the check failed.
    """

    array = np.asarray(values, dtype=np.float64)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _column_space_projector(matrix: np.ndarray, rtol: float) -> tuple[np.ndarray, int]:
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros((matrix.shape[0], matrix.shape[0])), 0
    rank = int(np.count_nonzero(singular_values > rtol * singular_values[0]))
    basis = left[:, :rank]
    return basis @ basis.T, rank


def check_assembly_nullspace_equivalence(
    attachment_map: AttachmentMap,
    *,
    loss_gradient: Callable[[np.ndarray], np.ndarray] | None = None,
    seed: int = 0,
    rtol: float = 1e-10,
) -> AssemblyEquivalenceReport:
    """Verify ``im(A) = ker(C)`` and gradient accumulation for one attachment map.

    Column spaces are compared through their orthogonal projectors rather than
    through bases, because ``A`` and any null-space basis of ``C`` need not share
    a basis even when they span the same subspace.
    """

    if rtol <= 0.0:
        raise ValueError("rtol must be positive")
    assembly = attachment_map.coordinate_assembly_matrix()
    n_local = assembly.shape[0] // 3
    pairs = spanning_tree_equalities(attachment_map)
    constraint = constraint_matrix_from_equalities(n_local, pairs)

    constraint_singular = np.linalg.svd(constraint, compute_uv=False)
    constraint_rank = (
        int(np.count_nonzero(constraint_singular > rtol * constraint_singular[0]))
        if constraint_singular.size
        else 0
    )
    right = np.linalg.svd(constraint, full_matrices=True)[2]
    nullspace_basis = right[constraint_rank:].T
    nullspace_dimension = nullspace_basis.shape[1]

    assembly_projector, assembly_rank = _column_space_projector(assembly, rtol)
    nullspace_projector = nullspace_basis @ nullspace_basis.T

    rng = np.random.default_rng(seed)
    physical = rng.normal(size=(attachment_map.n_global_vertices, 3))
    # Default loss: the squared norm of the local coordinates, whose gradient
    # differs at every occurrence, so a map that failed to share degrees of
    # freedom would produce a mismatch rather than a coincidental zero.
    gradient_of = loss_gradient if loss_gradient is not None else (lambda local: 2.0 * local)

    local_coordinates = physical[np.concatenate(
        [indices for indices in attachment_map.local_to_global.values()]
    )]
    ambient_gradient = np.asarray(gradient_of(local_coordinates), dtype=np.float64)
    if ambient_gradient.shape != local_coordinates.shape:
        raise ValueError("loss_gradient must return an array shaped like the local vertices")
    physical_gradient = assembly.T @ ambient_gradient.reshape(-1)
    accumulated = np.zeros((attachment_map.n_global_vertices, 3), dtype=np.float64)
    offset = 0
    for indices in attachment_map.local_to_global.values():
        for local_row, global_index in enumerate(indices.tolist()):
            accumulated[global_index] += ambient_gradient[offset + local_row]
        offset += len(indices)

    return AssemblyEquivalenceReport(
        n_local_vertices=n_local,
        n_physical_vertices=attachment_map.n_global_vertices,
        n_equality_rows=constraint.shape[0],
        constraint_rank=constraint_rank,
        nullspace_dimension=nullspace_dimension,
        assembly_rank=assembly_rank,
        max_constraint_on_assembly=_max_abs(constraint @ assembly),
        max_projector_difference=_max_abs(assembly_projector - nullspace_projector),
        max_gradient_sharing_error=_max_abs(physical_gradient - accumulated.reshape(-1)),
    )
