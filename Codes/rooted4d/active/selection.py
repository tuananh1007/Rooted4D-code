r"""Expected uncertainty reduction per unit acquisition cost, over imaging planes.

This is the ActivePlaneTwin criterion. For a Gaussian posterior and a linearized
forward model, the posterior covariance after adding a plane depends only on that
plane's Jacobian, **not on the values it would measure**. Expected uncertainty
reduction is therefore computable before acquiring anything, which is what makes
sequential planning possible at the scanner rather than in hindsight.

Concretely, with prior precision ``Lambda`` and a candidate plane contributing
rows ``J_a``:

.. math::
   \\Lambda' = \\Lambda + J_a^\\top J_a / \\sigma^2,
   \\qquad
   \\operatorname{Var}(q) \\approx \\nabla q^\\top \\Lambda'^{-1} \\nabla q .

The utility of a plane is the drop in the variance of the clinical quantities
divided by what the plane costs to acquire. This is Bayesian optimal experimental
design restricted to the quantities of interest rather than to the parameters,
which matters here: a plane that pins down parameters nobody reports is not worth
scan time.

The distinction from the two closest works is deliberate. AcquisitionFocus
optimizes plane orientation against reconstruction loss, once, offline, for two
views. CUTE-MRI adapts sampling density under fixed geometry. Selecting geometry
sequentially, against uncertainty in a reported quantity, against cost, is
neither.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .planes import PlaneCandidate, plane_contour_points


@dataclass(frozen=True)
class PlaneUtility:
    """What one candidate plane is expected to buy, and at what cost."""

    name: str
    variance_before: float
    variance_after: float
    reduction: float
    cost: float
    utility: float
    n_evidence_rows: int

    def as_dict(self) -> dict[str, float | str | int]:
        return {
            "name": self.name,
            "variance_before": self.variance_before,
            "variance_after": self.variance_after,
            "reduction": self.reduction,
            "cost": self.cost,
            "utility": self.utility,
            "n_evidence_rows": self.n_evidence_rows,
        }


def plane_jacobian(
    geometry_of: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    parameters: np.ndarray,
    candidate: PlaneCandidate,
    *,
    step: float = 1e-5,
    samples_per_segment: int = 2,
) -> np.ndarray:
    """Central-difference Jacobian of a plane's contour evidence.

    A contour can change its number of samples when the parameters move, because
    the plane cuts a different set of triangles. Rows are therefore truncated to
    the smallest count seen across the perturbations; a candidate whose sample
    count is unstable contributes fewer rows rather than a mis-aligned Jacobian.
    """

    theta = np.asarray(parameters, dtype=np.float64)
    columns = []
    lengths = []
    for index in range(theta.size):
        offset = np.zeros_like(theta)
        offset[index] = step
        forward_v, forward_f = geometry_of(theta + offset)
        backward_v, backward_f = geometry_of(theta - offset)
        forward = plane_contour_points(
            forward_v, forward_f, candidate, samples_per_segment=samples_per_segment
        )
        backward = plane_contour_points(
            backward_v, backward_f, candidate, samples_per_segment=samples_per_segment
        )
        count = min(len(forward), len(backward))
        lengths.append(count)
        columns.append(((forward[:count] - backward[:count]) / (2.0 * step)).reshape(-1))
    usable = min(lengths) if lengths else 0
    if usable == 0:
        return np.zeros((0, theta.size))
    trimmed = [column[: 2 * usable] for column in columns]
    return np.stack(trimmed, axis=1)


def whiten_correlated_contour(
    jacobian: np.ndarray,
    *,
    correlation_length: float,
    n_coordinates: int = 2,
) -> np.ndarray:
    """Whiten a contour Jacobian under spatially correlated segmentation error.

    Treating every contour sample as an independent measurement is not a
    conservative simplification; it is the difference between an experiment that
    can discriminate and one that cannot. A hundred samples along one contour,
    taken as independent, carry enough information that a single plane saturates
    any reasonable precision target, every acquisition strategy ties at one
    plane, and the comparison becomes vacuous.

    Real segmentation error is strongly correlated along a contour: a boundary
    that is biased outward tends to be biased outward for its whole neighbourhood.
    An exponential kernel in sample index captures that, and whitening by its
    Cholesky factor converts the Jacobian into one whose rows really can be
    treated as independent.

    ``correlation_length`` is in samples. Zero recovers the independent case.
    """

    matrix = np.asarray(jacobian, dtype=np.float64)
    if matrix.size == 0:
        return matrix
    if correlation_length <= 0.0:
        return matrix
    if matrix.shape[0] % n_coordinates:
        raise ValueError("row count must be a multiple of n_coordinates")
    n_points = matrix.shape[0] // n_coordinates
    index = np.arange(n_points, dtype=np.float64)
    kernel = np.exp(-np.abs(index[:, None] - index[None, :]) / correlation_length)
    kernel = kernel + 1e-8 * np.eye(n_points)
    factor = np.linalg.cholesky(kernel)
    reshaped = matrix.reshape(n_points, n_coordinates, matrix.shape[1])
    whitened = np.linalg.solve(
        factor, reshaped.reshape(n_points, -1)
    ).reshape(n_points, n_coordinates, matrix.shape[1])
    return whitened.reshape(matrix.shape)


def posterior_precision(
    prior_precision: np.ndarray,
    jacobians: Sequence[np.ndarray],
    noise_variance: float,
) -> np.ndarray:
    """Accumulate evidence into the precision matrix."""

    if noise_variance <= 0.0:
        raise ValueError("noise_variance must be positive")
    precision = np.array(prior_precision, dtype=np.float64, copy=True)
    for jacobian in jacobians:
        if jacobian.size:
            precision = precision + (jacobian.T @ jacobian) / noise_variance
    return precision


def quantity_variance(
    precision: np.ndarray,
    gradients: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Total posterior variance of the reported quantities.

    ``gradients`` holds one row per quantity. Weighting exists so quantities on
    different scales, a volume in millilitres and a dimensionless fraction, can
    be combined without one of them dominating by units alone.
    """

    matrix = np.asarray(gradients, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("gradients must have shape (quantities, parameters)")
    covariance = np.linalg.inv(precision)
    variances = np.einsum("qi,ij,qj->q", matrix, covariance, matrix)
    if weights is None:
        return float(np.sum(np.clip(variances, 0.0, None)))
    weight = np.asarray(weights, dtype=np.float64)
    if weight.shape != (matrix.shape[0],):
        raise ValueError("weights must have one entry per quantity")
    return float(np.sum(weight * np.clip(variances, 0.0, None)))


def score_candidates(
    candidates: Sequence[PlaneCandidate],
    jacobians: Sequence[np.ndarray],
    current_precision: np.ndarray,
    gradients: np.ndarray,
    noise_variance: float,
    *,
    weights: np.ndarray | None = None,
) -> list[PlaneUtility]:
    """Rank candidates by expected variance reduction per unit cost."""

    if len(candidates) != len(jacobians):
        raise ValueError("one Jacobian is required per candidate")
    before = quantity_variance(current_precision, gradients, weights)
    scored = []
    for candidate, jacobian in zip(candidates, jacobians):
        updated = posterior_precision(current_precision, [jacobian], noise_variance)
        after = quantity_variance(updated, gradients, weights)
        reduction = before - after
        scored.append(
            PlaneUtility(
                name=candidate.name,
                variance_before=before,
                variance_after=after,
                reduction=reduction,
                cost=candidate.cost,
                utility=reduction / candidate.cost,
                n_evidence_rows=int(jacobian.shape[0]),
            )
        )
    return scored


@dataclass(frozen=True)
class SelectionStep:
    """One decision in a sequential acquisition plan."""

    index: int
    chosen: str
    utility: float
    cumulative_cost: float
    variance_after: float
    n_candidates: int

    def as_dict(self) -> dict[str, float | str | int]:
        return {
            "index": self.index,
            "chosen": self.chosen,
            "utility": self.utility,
            "cumulative_cost": self.cumulative_cost,
            "variance_after": self.variance_after,
            "n_candidates": self.n_candidates,
        }


def greedy_plan(
    candidates: Sequence[PlaneCandidate],
    jacobians: Sequence[np.ndarray],
    prior_precision: np.ndarray,
    gradients: np.ndarray,
    noise_variance: float,
    *,
    n_planes: int,
    weights: np.ndarray | None = None,
    variance_target: float | None = None,
) -> list[SelectionStep]:
    """Select planes one at a time, each conditioned on those already chosen.

    Greedy is the honest choice here rather than a convenience. The decision is
    made at the scanner with the previous planes already acquired, so the
    sequential structure is the problem's, not an approximation of a batch
    optimum. ``variance_target`` provides the stopping rule: acquisition ends
    once the reported quantities are precise enough, which is the same contract
    CUTE-MRI applies to sampling density.
    """

    if n_planes < 1:
        raise ValueError("n_planes must be at least one")
    precision = np.array(prior_precision, dtype=np.float64, copy=True)
    remaining = list(range(len(candidates)))
    steps: list[SelectionStep] = []
    cost = 0.0

    for index in range(n_planes):
        if not remaining:
            break
        scored = score_candidates(
            [candidates[i] for i in remaining],
            [jacobians[i] for i in remaining],
            precision,
            gradients,
            noise_variance,
            weights=weights,
        )
        best = int(np.argmax([entry.utility for entry in scored]))
        chosen = remaining[best]
        precision = posterior_precision(precision, [jacobians[chosen]], noise_variance)
        cost += candidates[chosen].cost
        variance = quantity_variance(precision, gradients, weights)
        steps.append(
            SelectionStep(
                index=index,
                chosen=candidates[chosen].name,
                utility=scored[best].utility,
                cumulative_cost=cost,
                variance_after=variance,
                n_candidates=len(remaining),
            )
        )
        remaining.pop(best)
        if variance_target is not None and variance <= variance_target:
            break
    return steps


def fixed_order_plan(
    candidates: Sequence[PlaneCandidate],
    jacobians: Sequence[np.ndarray],
    prior_precision: np.ndarray,
    gradients: np.ndarray,
    noise_variance: float,
    *,
    order: Sequence[int],
    weights: np.ndarray | None = None,
) -> list[SelectionStep]:
    """Follow a prescribed order, for the protocol and random baselines."""

    precision = np.array(prior_precision, dtype=np.float64, copy=True)
    steps: list[SelectionStep] = []
    cost = 0.0
    for index, choice in enumerate(order):
        precision = posterior_precision(precision, [jacobians[choice]], noise_variance)
        cost += candidates[choice].cost
        steps.append(
            SelectionStep(
                index=index,
                chosen=candidates[choice].name,
                utility=float("nan"),
                cumulative_cost=cost,
                variance_after=quantity_variance(precision, gradients, weights),
                n_candidates=len(candidates),
            )
        )
    return steps
