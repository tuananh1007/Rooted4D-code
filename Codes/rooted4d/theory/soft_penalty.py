r"""Theorem 2 falsification: finite soft attachment leaves ``O(1/lambda)`` drift.

For the soft alternative

.. math::
   F_\lambda(x)=f(x)+\tfrac{\lambda}{2}\lVert Cx\rVert_2^2 ,

Theorem 2 shows that every stationary point satisfies

.. math::
   Cx_\lambda=-\frac{1}{\lambda}(CC^\top)^{-1}C\nabla f(x_\lambda),

so no finite penalty weight attains exact attachment unless the projected
data/prior gradient vanishes.  The theorem-to-experiment contract asks for a
prespecified weight sweep whose attachment residual has an approximately ``-1``
log-log slope, contrasted against the assembly parameterization, which stays at
floating-point tolerance without tuning.

The reference data term here is the quadratic ``f(x) = 1/2 (x - b)^T M (x - b)``
with symmetric positive-definite ``M``.  A quadratic makes the stationary point
available in closed form, so the sweep measures the penalty's behaviour rather
than an optimizer's convergence.  Section 4.6's full objective is not quadratic;
this module isolates the algebraic claim, which is the part Theorem 2 makes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate_system(
    constraint: np.ndarray,
    metric: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(constraint, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("constraint must be a non-empty two-dimensional matrix")
    weight = np.asarray(metric, dtype=np.float64)
    columns = matrix.shape[1]
    if weight.shape != (columns, columns):
        raise ValueError(f"metric must have shape ({columns}, {columns})")
    if not np.allclose(weight, weight.T, atol=1e-12):
        raise ValueError("metric must be symmetric")
    eigenvalues = np.linalg.eigvalsh(weight)
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError("metric must be positive definite")
    centre = np.asarray(target, dtype=np.float64)
    if centre.shape != (columns,):
        raise ValueError(f"target must have shape ({columns},)")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values[-1] <= 1e-12 * singular_values[0]:
        raise ValueError(
            "constraint must have full row rank; use an independent spanning-tree "
            "equality set as required by assumption A2"
        )
    return matrix, weight, centre


@dataclass(frozen=True)
class SoftPenaltyPoint:
    """One stationary point of the soft-attachment surrogate."""

    penalty_weight: float
    attachment_residual: float
    predicted_residual: float
    condition_number: float

    def as_dict(self) -> dict[str, float]:
        return {
            "penalty_weight": self.penalty_weight,
            "attachment_residual": self.attachment_residual,
            "predicted_residual": self.predicted_residual,
            "condition_number": self.condition_number,
        }


@dataclass(frozen=True)
class SoftPenaltySweep:
    """A prespecified penalty sweep and its fitted log-log slope."""

    points: tuple[SoftPenaltyPoint, ...]
    slope: float
    max_identity_error: float

    @property
    def penalty_weights(self) -> np.ndarray:
        return np.array([point.penalty_weight for point in self.points])

    @property
    def attachment_residuals(self) -> np.ndarray:
        return np.array([point.attachment_residual for point in self.points])

    def as_dict(self) -> dict[str, object]:
        return {
            "slope": self.slope,
            "max_identity_error": self.max_identity_error,
            "points": [point.as_dict() for point in self.points],
        }


def soft_attachment_stationary_point(
    constraint: np.ndarray,
    metric: np.ndarray,
    target: np.ndarray,
    penalty_weight: float,
) -> SoftPenaltyPoint:
    """Solve ``(M + lambda C^T C) x = M b`` and compare against Theorem 2's identity."""

    matrix, weight, centre = _validate_system(constraint, metric, target)
    if not np.isfinite(penalty_weight) or penalty_weight <= 0.0:
        raise ValueError("penalty_weight must be positive and finite")
    system = weight + penalty_weight * (matrix.T @ matrix)
    solution = np.linalg.solve(system, weight @ centre)
    residual = matrix @ solution
    gradient = weight @ (solution - centre)
    predicted = -np.linalg.solve(matrix @ matrix.T, matrix @ gradient) / penalty_weight
    return SoftPenaltyPoint(
        penalty_weight=float(penalty_weight),
        attachment_residual=float(np.linalg.norm(residual)),
        predicted_residual=float(np.linalg.norm(predicted)),
        condition_number=float(np.linalg.cond(system)),
    )


def soft_attachment_sweep(
    constraint: np.ndarray,
    metric: np.ndarray,
    target: np.ndarray,
    penalty_weights: np.ndarray,
) -> SoftPenaltySweep:
    """Run the sweep and fit ``log residual`` against ``log lambda``.

    Theorem 2 predicts a slope near ``-1``.  The sweep also reports the largest
    disagreement with the closed-form stationarity identity, which is the check
    that distinguishes a genuine confirmation from a coincidental trend.
    """

    weights = np.asarray(penalty_weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size < 2:
        raise ValueError("penalty_weights must contain at least two values")
    if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("penalty_weights must be positive and finite")
    points = tuple(
        soft_attachment_stationary_point(constraint, metric, target, float(value))
        for value in weights
    )
    residuals = np.array([point.attachment_residual for point in points])
    if np.any(residuals <= 0.0):
        raise ValueError(
            "a stationary residual reached exactly zero; the projected gradient "
            "vanishes and this system cannot falsify Theorem 2"
        )
    slope = float(
        np.polyfit(np.log(weights), np.log(residuals), deg=1)[0]
    )
    identity_error = max(
        abs(point.attachment_residual - point.predicted_residual)
        / max(point.attachment_residual, 1e-300)
        for point in points
    )
    return SoftPenaltySweep(
        points=points,
        slope=slope,
        max_identity_error=float(identity_error),
    )
