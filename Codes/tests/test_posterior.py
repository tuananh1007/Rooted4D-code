"""Tests for the Laplace posterior and split-conformal calibration."""

import unittest

import numpy as np

from rooted4d.inference import SceneModel, SolverConfig, fit_joint_scene
from rooted4d.inference.posterior import (
    calibrate_split_conformal,
    empirical_coverage,
    fit_laplace_posterior,
    propagate_measurements,
)
from rooted4d.inference.problem import make_joint_problem


def _fit(seed: int = 0):
    model = SceneModel.procedural(n_modes=10, n_motion=3, n_session=2, n_phase_nodes=6)
    config = SolverConfig(fourier_order=2, session_order=1, max_iterations=25)
    problem = make_joint_problem(model, config, seed=seed)
    return fit_joint_scene(problem.observations, model, config), problem


class LaplacePosteriorTest(unittest.TestCase):
    def test_gauge_directions_carry_no_variance(self):
        """A naive inverse would report infinite variance along the gauges."""

        fit, _ = _fit()
        posterior = fit_laplace_posterior(fit)
        gauges = fit.gauge_directions()
        projected = gauges @ posterior.covariance() @ gauges.T
        self.assertLess(float(np.max(np.abs(projected))), 1e-12)

    def test_tangent_dimension_excludes_the_gauges(self):
        fit, _ = _fit()
        posterior = fit_laplace_posterior(fit)
        self.assertEqual(
            posterior.tangent_dimension,
            posterior.n_parameters - posterior.gauge_dimension,
        )

    def test_covariance_is_symmetric_positive_semidefinite(self):
        fit, _ = _fit()
        covariance = fit_laplace_posterior(fit).covariance()
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
        eigenvalues = np.linalg.eigvalsh(covariance)
        self.assertGreater(float(eigenvalues[0]), -1e-10)

    def test_samples_have_the_requested_shape_and_are_deterministic(self):
        fit, _ = _fit()
        posterior = fit_laplace_posterior(fit)
        first = posterior.sample(16, seed=3)
        second = posterior.sample(16, seed=3)
        self.assertEqual(first.shape, (16, posterior.n_parameters))
        np.testing.assert_array_equal(first, second)

    def test_measurements_propagate_with_positive_spread(self):
        fit, _ = _fit()
        posterior = fit_laplace_posterior(fit)
        samples = propagate_measurements(
            posterior,
            fit,
            {"identity_norm": lambda p: float(np.linalg.norm(p.identity))},
            n_samples=40,
        )
        self.assertEqual(samples["identity_norm"].shape, (40,))
        self.assertGreater(float(np.std(samples["identity_norm"])), 0.0)


class ConformalCalibrationTest(unittest.TestCase):
    def test_interval_attains_near_nominal_coverage(self):
        rng = np.random.default_rng(0)
        truth = rng.normal(size=400)
        predicted = truth + rng.normal(scale=0.5, size=400)
        interval = calibrate_split_conformal(
            "x", predicted[:200], truth[:200], coverage_target=0.90
        )
        coverage = empirical_coverage(interval, predicted[200:], truth[200:])
        self.assertLess(abs(coverage["coverage"] - 0.90), 0.05)

    def test_higher_target_gives_a_wider_interval(self):
        rng = np.random.default_rng(1)
        truth = rng.normal(size=200)
        predicted = truth + rng.normal(scale=0.4, size=200)
        narrow = calibrate_split_conformal("x", predicted, truth, coverage_target=0.80)
        wide = calibrate_split_conformal("x", predicted, truth, coverage_target=0.95)
        self.assertGreater(wide.quantile, narrow.quantile)

    def test_unattainable_coverage_returns_an_infinite_interval(self):
        """Honest failure beats a silently capped quantile."""

        rng = np.random.default_rng(2)
        truth = rng.normal(size=3)
        interval = calibrate_split_conformal(
            "x", truth + 0.1, truth, coverage_target=0.99
        )
        self.assertEqual(interval.quantile, float("inf"))

    def test_too_few_calibration_cases_are_rejected(self):
        with self.assertRaises(ValueError):
            calibrate_split_conformal("x", np.array([1.0]), np.array([1.0]))


if __name__ == "__main__":
    unittest.main()


class CircularCalibrationTest(unittest.TestCase):
    """Cardiac phase is circular; absolute residuals wrap and inflate intervals."""

    def test_absolute_residuals_wrap_but_circular_ones_do_not(self):
        predicted = np.array([0.02, 0.05, 0.98, 0.01])
        truth = np.array([0.98, 0.99, 0.02, 0.97])
        plain = calibrate_split_conformal("phase", predicted, truth, coverage_target=0.8)
        wrapped = calibrate_split_conformal(
            "phase", predicted, truth, coverage_target=0.8, circular=True
        )
        self.assertGreater(plain.quantile, 0.5)
        self.assertLess(wrapped.quantile, 0.1)

    def test_circular_coverage_uses_wrapped_distance(self):
        predicted = np.array([0.02, 0.05, 0.98, 0.01])
        truth = np.array([0.98, 0.99, 0.02, 0.97])
        interval = calibrate_split_conformal(
            "phase", predicted, truth, coverage_target=0.8, circular=True
        )
        coverage = empirical_coverage(interval, predicted, truth)
        self.assertGreaterEqual(coverage["coverage"], 0.75)

    def test_minimum_calibration_size_matches_the_attainability_rule(self):
        from rooted4d.inference.posterior import minimum_calibration_size

        for target in (0.80, 0.90, 0.95):
            n = minimum_calibration_size(target)
            rng = np.random.default_rng(0)
            truth = rng.normal(size=n)
            enough = calibrate_split_conformal(
                "x", truth + 0.1, truth, coverage_target=target
            )
            self.assertTrue(np.isfinite(enough.quantile))
            short_truth = truth[: n - 1]
            too_few = calibrate_split_conformal(
                "x", short_truth + 0.1, short_truth, coverage_target=target
            )
            self.assertEqual(too_few.quantile, float("inf"))
