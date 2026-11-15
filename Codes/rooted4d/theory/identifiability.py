r"""Theorems 5 and 6: constrained identifiability and local noise stability.

Theorem 5 states that the Rooted4D variables are first-order locally identifiable
at ``theta*`` if and only if

.. math::
   \ker(J_C)\cap\ker(J_E)=\{0\},

equivalently that the restricted Gauss-Newton matrix
``H_T = B^T J_E^T W J_E B`` is positive definite for an orthonormal basis ``B``
of the feasible tangent space ``ker(J_C)``.

Theorem 6 then bounds the local estimation error by

.. math::
   \lVert\widehat\theta-\theta^\star\rVert_2
   \le\frac{4\lVert\varepsilon\rVert_W}{\sqrt{\alpha(1-c_{nl})}},
   \qquad \alpha=\lambda_{\min}(H_T),

with a regularized-MAP variant that exposes prior-induced bias rather than
hiding it.

Both results are local.  The smallest tangent eigenvalue is reported together
with the ambient direction attaining it, because the specification requires
"null vector anatomy" — which latent block a weakly determined direction lives
in — not only a scalar condition number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


def _tangent_basis(constraint_jacobian: np.ndarray | None, columns: int, rtol: float) -> np.ndarray:
    if constraint_jacobian is None:
        return np.eye(columns, dtype=np.float64)
    matrix = np.asarray(constraint_jacobian, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != columns:
        raise ValueError(f"constraint_jacobian must have shape (rows, {columns})")
    if matrix.shape[0] == 0:
        return np.eye(columns, dtype=np.float64)
    singular_values, right = np.linalg.svd(matrix, full_matrices=True)[1:]
    rank = (
        int(np.count_nonzero(singular_values > rtol * singular_values[0]))
        if singular_values.size
        else 0
    )
    return right[rank:].T.copy()


@dataclass(frozen=True)
class ConstrainedIdentifiability:
    """Restricted Fisher/Gauss-Newton diagnostics on the feasible tangent space."""

    n_parameters: int
    tangent_dimension: int
    smallest_tangent_eigenvalue: float
    largest_tangent_eigenvalue: float
    tangent_condition_number: float
    weakest_direction: np.ndarray
    block_sensitivity: dict[str, float]

    @property
    def identifiable(self) -> bool:
        return self.tangent_dimension > 0 and self.smallest_tangent_eigenvalue > 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "n_parameters": self.n_parameters,
            "tangent_dimension": self.tangent_dimension,
            "smallest_tangent_eigenvalue": self.smallest_tangent_eigenvalue,
            "largest_tangent_eigenvalue": self.largest_tangent_eigenvalue,
            "tangent_condition_number": self.tangent_condition_number,
            "identifiable": self.identifiable,
            "block_sensitivity": dict(self.block_sensitivity),
        }


def constrained_identifiability(
    evidence_jacobian: np.ndarray,
    *,
    constraint_jacobian: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    parameter_blocks: Mapping[str, slice] | None = None,
    rtol: float = 1e-10,
) -> ConstrainedIdentifiability:
    """Evaluate Theorem 5 on one linearization.

    ``weights`` may be a one-dimensional vector of per-observation weights or a
    full symmetric positive-semidefinite matrix.  ``parameter_blocks`` names
    contiguous latent blocks (pose, phase, identity, motion, session) so the
    weakest tangent direction can be attributed to the block it occupies.
    """

    jacobian = np.asarray(evidence_jacobian, dtype=np.float64)
    if jacobian.ndim != 2 or jacobian.size == 0:
        raise ValueError("evidence_jacobian must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("evidence_jacobian must be finite")
    if rtol <= 0.0:
        raise ValueError("rtol must be positive")
    rows, columns = jacobian.shape

    if weights is None:
        weighted = jacobian
    else:
        weight_array = np.asarray(weights, dtype=np.float64)
        if weight_array.ndim == 1:
            if weight_array.shape != (rows,):
                raise ValueError(f"weight vector must have shape ({rows},)")
            if np.any(weight_array < 0.0):
                raise ValueError("observation weights must be non-negative")
            weighted = np.sqrt(weight_array)[:, None] * jacobian
        elif weight_array.ndim == 2:
            if weight_array.shape != (rows, rows):
                raise ValueError(f"weight matrix must have shape ({rows}, {rows})")
            eigenvalues, vectors = np.linalg.eigh(
                0.5 * (weight_array + weight_array.T)
            )
            if float(eigenvalues[0]) < -1e-10 * max(float(eigenvalues[-1]), 1.0):
                raise ValueError("weight matrix must be positive semidefinite")
            root = vectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None))) @ vectors.T
            weighted = root @ jacobian
        else:
            raise ValueError("weights must be one- or two-dimensional")

    basis = _tangent_basis(constraint_jacobian, columns, rtol)
    tangent_dimension = basis.shape[1]
    if tangent_dimension == 0:
        return ConstrainedIdentifiability(
            n_parameters=columns,
            tangent_dimension=0,
            smallest_tangent_eigenvalue=float("inf"),
            largest_tangent_eigenvalue=float("inf"),
            tangent_condition_number=1.0,
            weakest_direction=np.zeros(columns),
            block_sensitivity={},
        )

    restricted = weighted @ basis
    hessian = restricted.T @ restricted
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (hessian + hessian.T))
    smallest = float(max(eigenvalues[0], 0.0))
    largest = float(eigenvalues[-1])
    condition = float(largest / smallest) if smallest > 0.0 else float("inf")
    weakest = basis @ eigenvectors[:, 0]
    norm = float(np.linalg.norm(weakest))
    if norm > 0.0:
        weakest = weakest / norm

    sensitivity: dict[str, float] = {}
    if parameter_blocks:
        for name, block in parameter_blocks.items():
            sensitivity[name] = float(np.linalg.norm(weakest[block]))
    return ConstrainedIdentifiability(
        n_parameters=columns,
        tangent_dimension=tangent_dimension,
        smallest_tangent_eigenvalue=smallest,
        largest_tangent_eigenvalue=largest,
        tangent_condition_number=condition,
        weakest_direction=weakest,
        block_sensitivity=sensitivity,
    )


@dataclass(frozen=True)
class NoiseStabilityBound:
    """The Theorem 6 error bound and its regularized-MAP extension."""

    smallest_tangent_eigenvalue: float
    weighted_noise_norm: float
    nonlinearity: float
    bound: float
    regularized_bound: float

    @property
    def informative(self) -> bool:
        """A vanishing ``alpha`` makes the bound useless; that must be visible."""

        return bool(np.isfinite(self.bound))

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "smallest_tangent_eigenvalue": self.smallest_tangent_eigenvalue,
            "weighted_noise_norm": self.weighted_noise_norm,
            "nonlinearity": self.nonlinearity,
            "bound": self.bound,
            "regularized_bound": self.regularized_bound,
            "informative": self.informative,
        }


def noise_stability_bound(
    smallest_tangent_eigenvalue: float,
    weighted_noise_norm: float,
    *,
    nonlinearity: float = 0.0,
    prior_penalty: float = 0.0,
) -> NoiseStabilityBound:
    """Evaluate the Theorem 6 bounds.

    ``prior_penalty`` is ``lambda * R(theta*)`` from the regularized-MAP variant.
    A non-zero value widens the bound rather than tightening it, which is the
    intended behaviour: prior-induced bias is exposed, not absorbed.
    """

    alpha = float(smallest_tangent_eigenvalue)
    noise = float(weighted_noise_norm)
    curvature = float(nonlinearity)
    penalty = float(prior_penalty)
    if alpha < 0.0:
        raise ValueError("smallest_tangent_eigenvalue must be non-negative")
    if noise < 0.0:
        raise ValueError("weighted_noise_norm must be non-negative")
    if not 0.0 <= curvature < 1.0:
        raise ValueError("nonlinearity must lie in [0, 1)")
    if penalty < 0.0:
        raise ValueError("prior_penalty must be non-negative")
    denominator = np.sqrt(alpha * (1.0 - curvature))
    if denominator <= 0.0:
        return NoiseStabilityBound(
            smallest_tangent_eigenvalue=alpha,
            weighted_noise_norm=noise,
            nonlinearity=curvature,
            bound=float("inf"),
            regularized_bound=float("inf"),
        )
    return NoiseStabilityBound(
        smallest_tangent_eigenvalue=alpha,
        weighted_noise_norm=noise,
        nonlinearity=curvature,
        bound=float(4.0 * noise / denominator),
        regularized_bound=float(
            2.0 * (noise + np.sqrt(noise * noise + penalty)) / denominator
        ),
    )
