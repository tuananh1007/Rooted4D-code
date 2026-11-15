r"""Laplace posterior, measurement propagation, and split-conformal calibration.

Work package 6. Section 4.7 prescribes the practical first version: amortized
initialization, MAP optimization, a low-rank Laplace or variational Gaussian
around the optimum, posterior samples propagated through the clinical
measurements, and split-conformal calibration of scalar intervals on a
validation cohort.

The gauge directions matter here more than anywhere else. A Laplace
approximation inverts the Gauss-Newton matrix, which is singular along every
gauge, so a naive inverse produces infinite variance in directions that are pure
reparameterization and carry no uncertainty at all. The covariance is therefore
built on the non-gauge tangent space and mapped back, which is the same
restriction Theorem 5 uses.

Section 4.7 also insists the uncertainty claim cover clinical quantities and
pose/phase ambiguity, not merely per-vertex variance, so measurements are
propagated by sampling rather than by linearizing the geometry alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .joint import SceneFit, _unpack
from .scene import SceneParameters


@dataclass(frozen=True)
class LaplacePosterior:
    """Gaussian posterior on the non-gauge tangent space at a MAP solution."""

    mean: np.ndarray
    tangent_basis: np.ndarray
    tangent_covariance: np.ndarray
    noise_variance: float
    gauge_dimension: int

    @property
    def n_parameters(self) -> int:
        return self.mean.size

    @property
    def tangent_dimension(self) -> int:
        return self.tangent_basis.shape[1]

    def covariance(self) -> np.ndarray:
        """Ambient covariance, exactly zero along every gauge direction."""

        return self.tangent_basis @ self.tangent_covariance @ self.tangent_basis.T

    def marginal_standard_deviations(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.covariance()), 0.0, None))

    def sample(self, n_samples: int, seed: int = 0) -> np.ndarray:
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        rng = np.random.default_rng(seed)
        eigenvalues, vectors = np.linalg.eigh(self.tangent_covariance)
        root = vectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None)))
        draws = rng.normal(size=(n_samples, self.tangent_dimension)) @ root.T
        return self.mean + draws @ self.tangent_basis.T

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_parameters": self.n_parameters,
            "tangent_dimension": self.tangent_dimension,
            "gauge_dimension": self.gauge_dimension,
            "noise_variance": self.noise_variance,
            "max_marginal_sd": float(np.max(self.marginal_standard_deviations())),
        }


def fit_laplace_posterior(
    fit: SceneFit,
    *,
    ridge: float = 1e-10,
) -> LaplacePosterior:
    """Gauss-Newton Laplace approximation restricted to the non-gauge directions.

    The residual noise scale is estimated from the fit itself as
    ``cost / (n_residuals - n_free)``, so an over-parameterized fit widens rather
    than narrows its own intervals.
    """

    # The *objective* Jacobian, deliberately: for a posterior the prior is part
    # of the model, so its curvature belongs here. This used to call
    # evidence_jacobian(), which returned the same thing under a name that
    # promised observations only; the two are now distinct and this call is
    # explicit about which it wants. Behaviour is unchanged.
    #
    # Note that ``fit.cost`` below is the total penalized cost and
    # ``n_residuals`` the augmented count, so the noise scale is estimated from
    # the penalized problem too. That is self-consistent but has not been
    # reviewed as a statistical procedure, and it should not be changed
    # piecemeal by swapping in observation-only curvature while leaving the
    # scale as it is.
    jacobian = fit.objective_jacobian()
    gauges = fit.gauge_directions()
    n_residuals, n_parameters = jacobian.shape

    if gauges.size:
        # Orthonormal complement of the gauge subspace.
        _, _, right = np.linalg.svd(gauges, full_matrices=True)
        basis = right[gauges.shape[0] :].T.copy()
    else:
        basis = np.eye(n_parameters)

    restricted = jacobian @ basis
    hessian = restricted.T @ restricted
    scale = float(np.trace(hessian)) / max(hessian.shape[0], 1)
    hessian = hessian + max(ridge * max(scale, 1.0), 1e-12) * np.eye(hessian.shape[0])

    degrees_of_freedom = max(n_residuals - basis.shape[1], 1)
    noise_variance = float(fit.cost) / degrees_of_freedom
    covariance = noise_variance * np.linalg.inv(hessian)
    covariance = 0.5 * (covariance + covariance.T)
    return LaplacePosterior(
        mean=fit.packed.copy(),
        tangent_basis=basis,
        tangent_covariance=covariance,
        noise_variance=noise_variance,
        gauge_dimension=int(gauges.shape[0]),
    )


def propagate_measurements(
    posterior: LaplacePosterior,
    fit: SceneFit,
    measurements: Mapping[str, Callable[[SceneParameters], float]],
    *,
    n_samples: int = 200,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Push posterior samples through scalar clinical measurements."""

    if not measurements:
        raise ValueError("at least one measurement is required")
    draws = posterior.sample(n_samples, seed=seed)
    results: dict[str, list[float]] = {name: [] for name in measurements}
    for vector in draws:
        parameters = _unpack(vector, fit.parameters, fit.model, fit.config)
        for name, function in measurements.items():
            results[name].append(float(function(parameters)))
    return {name: np.asarray(values, dtype=np.float64) for name, values in results.items()}


@dataclass(frozen=True)
class ConformalInterval:
    """A split-conformal interval calibrated on held-out cases only."""

    name: str
    coverage_target: float
    quantile: float
    n_calibration: int
    #: Circular quantities need wrapped residuals. Cardiac phase lives on the
    #: circle, so a plain absolute residual scores 0.9 for a pair 0.1 apart and
    #: inflates every interval it touches.
    circular: bool = False

    def interval(self, point_estimate: float) -> tuple[float, float]:
        return (point_estimate - self.quantile, point_estimate + self.quantile)

    def as_dict(self) -> dict[str, float | str | int | bool]:
        return {
            "name": self.name,
            "coverage_target": self.coverage_target,
            "quantile": self.quantile,
            "n_calibration": self.n_calibration,
            "circular": self.circular,
            "attainable": bool(np.isfinite(self.quantile)),
        }


def _residuals(
    predictions: np.ndarray, truths: np.ndarray, circular: bool
) -> np.ndarray:
    if circular:
        from rooted4d.motion import circular_distance

        return circular_distance(predictions, truths)
    return np.abs(predictions - truths)


def minimum_calibration_size(coverage_target: float) -> int:
    """Smallest calibration set for which the target is attainable.

    Split conformal needs ``ceil((n+1)q) <= n``. Below that the honest interval
    is infinite, so this is worth checking before designing a study rather than
    discovering it in the results.

    This answers attainability, not reliability, and the gap between the two is
    large. The conformal guarantee is *marginal over the draw of the calibration
    set*, so a single realization can miss the target badly while the theory
    holds. On this benchmark the function returns 9 for a 0.90 target, and 30
    calibration identities still under-covered annulus diameter at 0.733,
    because that split's validation and test cohorts happened to differ in the
    anatomy itself: mean true diameter 51.5 against 47.4 mm. Pose and phase were
    unaffected, their error being driven by imaging geometry rather than by which
    people landed in which split. Treat this number as a floor to clear, never as
    a calibration set size to design to.
    """

    if not 0.0 < coverage_target < 1.0:
        raise ValueError("coverage_target must lie in (0, 1)")
    # The closed form ceil(q/(1-q)) is off by one wherever the division rounds
    # up: 0.9/0.1 evaluates to 9.000000000000002, which would demand ten cases
    # when nine satisfy the rule. Search the actual condition instead, starting
    # from the closed form so the loop is short.
    candidate = max(1, int(coverage_target / (1.0 - coverage_target)) - 1)
    while int(np.ceil((candidate + 1) * coverage_target)) > candidate:
        candidate += 1
    return candidate


def calibrate_split_conformal(
    name: str,
    predictions: np.ndarray,
    truths: np.ndarray,
    *,
    coverage_target: float = 0.90,
    circular: bool = False,
) -> ConformalInterval:
    """Fit the residual quantile on calibration cases.

    Uses the finite-sample corrected rank so the guarantee is the usual
    distribution-free one rather than an asymptotic approximation. The protocol
    requires this to be fitted on validation subjects only; passing test cases
    here would invalidate the coverage claim, which is why the caller must do the
    splitting and this function never sees a split definition.
    """

    predicted = np.asarray(predictions, dtype=np.float64)
    actual = np.asarray(truths, dtype=np.float64)
    if predicted.shape != actual.shape or predicted.ndim != 1:
        raise ValueError("predictions and truths must be paired one-dimensional arrays")
    if predicted.size < 2:
        raise ValueError("at least two calibration cases are required")
    if not 0.0 < coverage_target < 1.0:
        raise ValueError("coverage_target must lie in (0, 1)")

    residuals = _residuals(predicted, actual, circular)
    n = residuals.size
    rank = int(np.ceil((n + 1) * coverage_target))
    if rank > n:
        # The requested coverage is not attainable at this calibration size; the
        # honest response is an infinite interval, not a silently capped one.
        return ConformalInterval(name, coverage_target, float("inf"), n, circular)
    quantile = float(np.sort(residuals)[rank - 1])
    return ConformalInterval(name, coverage_target, quantile, n, circular)


def empirical_coverage(
    interval: ConformalInterval,
    predictions: np.ndarray,
    truths: np.ndarray,
) -> dict[str, float]:
    """Coverage and mean width on evaluation cases the calibration never saw."""

    predicted = np.asarray(predictions, dtype=np.float64)
    actual = np.asarray(truths, dtype=np.float64)
    if predicted.shape != actual.shape or predicted.ndim != 1:
        raise ValueError("predictions and truths must be paired one-dimensional arrays")
    covered = _residuals(predicted, actual, interval.circular) <= interval.quantile
    observed = float(np.mean(covered))
    target = interval.coverage_target
    # Binomial standard error of the coverage estimate. Without it a reader
    # cannot tell a miscalibrated interval from a small evaluation set: at
    # twenty-odd cases the error on a 0.90 target is already about six points,
    # so a "within five points" criterion is below the resolution of the sample.
    standard_error = float(np.sqrt(target * (1.0 - target) / max(predicted.size, 1)))
    return {
        "coverage": observed,
        "target": target,
        "coverage_error": observed - target,
        "coverage_standard_error": standard_error,
        "within_one_standard_error": bool(abs(observed - target) <= standard_error),
        # Split conformal is one-sided: it guarantees coverage at least the
        # target in finite samples, so over-coverage is correct conservative
        # behaviour and only under-coverage is a failure. A symmetric criterion
        # would mark a valid interval as miscalibrated.
        "meets_guarantee": bool(observed >= target - standard_error),
        "under_covers": bool(observed < target - standard_error),
        "mean_width": float(2.0 * interval.quantile),
        "n_evaluation": int(predicted.size),
    }
