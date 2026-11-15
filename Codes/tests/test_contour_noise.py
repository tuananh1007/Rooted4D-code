"""Tests for the contour-residual correlation estimator."""

import unittest

import numpy as np

from rooted4d.active.contour_noise import (
    circular_autocorrelation,
    contour_residuals,
    contour_sample_spacing_mm,
    estimate_correlation_length,
    mask_boundary_contour,
    outward_normals,
    resample_closed_contour,
    residuals_from_mask_pairs,
    synthetic_correlated_residuals,
)


def _circle(radius: float, n: int = 90) -> np.ndarray:
    angle = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)


class ResamplingTest(unittest.TestCase):
    def test_resampling_preserves_a_circle(self):
        resampled = resample_closed_contour(_circle(10.0), 64)
        radii = np.linalg.norm(resampled, axis=1)
        self.assertTrue(np.allclose(radii, 10.0, atol=0.05))

    def test_resampling_is_uniform_in_arclength(self):
        resampled = resample_closed_contour(_circle(10.0, 37), 64)
        steps = np.linalg.norm(np.diff(resampled, axis=0), axis=1)
        self.assertLess(float(np.std(steps) / np.mean(steps)), 0.05)

    def test_degenerate_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            resample_closed_contour(np.zeros((2, 2)), 16)
        with self.assertRaises(ValueError):
            resample_closed_contour(_circle(1.0), 2)

    def test_normals_point_outward(self):
        contour = resample_closed_contour(_circle(5.0), 32)
        normals = outward_normals(contour)
        radial = contour / np.linalg.norm(contour, axis=1, keepdims=True)
        self.assertGreater(float(np.mean(np.sum(normals * radial, axis=1))), 0.95)


class ResidualTest(unittest.TestCase):
    def test_uniform_dilation_gives_a_constant_positive_residual(self):
        residuals = contour_residuals(_circle(10.0), _circle(11.0), n_samples=64)
        self.assertGreater(float(np.mean(residuals)), 0.8)
        self.assertLess(float(np.std(residuals)), 0.15)

    def test_identical_contours_give_near_zero_residuals(self):
        residuals = contour_residuals(_circle(10.0), _circle(10.0), n_samples=64)
        self.assertLess(float(np.max(np.abs(residuals))), 0.05)

    def test_autocorrelation_starts_at_one(self):
        rng = np.random.default_rng(0)
        curve = circular_autocorrelation(rng.normal(size=64))
        self.assertAlmostEqual(float(curve[0]), 1.0, places=10)

    def test_constant_residuals_are_rejected_by_autocorrelation(self):
        with self.assertRaises(ValueError):
            circular_autocorrelation(np.ones(32))


class CorrelationEstimatorTest(unittest.TestCase):
    """The estimator is validated against ground truth before use on real data."""

    def test_recovers_a_known_short_correlation_length(self):
        rng = np.random.default_rng(0)
        sets = [synthetic_correlated_residuals(128, 2.0, 1.5, rng) for _ in range(40)]
        estimate = estimate_correlation_length(sets)
        self.assertLess(abs(estimate.correlation_length_samples - 2.0), 0.6)

    def test_recovers_a_known_medium_correlation_length(self):
        rng = np.random.default_rng(1)
        sets = [synthetic_correlated_residuals(128, 5.0, 1.5, rng) for _ in range(40)]
        estimate = estimate_correlation_length(sets)
        self.assertLess(abs(estimate.correlation_length_samples - 5.0) / 5.0, 0.25)

    def test_independent_residuals_give_a_short_length(self):
        rng = np.random.default_rng(2)
        sets = [1.2 * rng.normal(size=128) for _ in range(40)]
        estimate = estimate_correlation_length(sets)
        self.assertLess(estimate.correlation_length_samples, 1.0)

    def test_effective_sample_count_falls_as_correlation_rises(self):
        rng = np.random.default_rng(3)
        short = estimate_correlation_length(
            [synthetic_correlated_residuals(128, 1.0, 1.0, rng) for _ in range(20)]
        )
        long = estimate_correlation_length(
            [synthetic_correlated_residuals(128, 12.0, 1.0, rng) for _ in range(20)]
        )
        self.assertGreater(
            short.effective_independent_samples, long.effective_independent_samples
        )

    def test_residual_standard_deviation_is_reported(self):
        rng = np.random.default_rng(4)
        sets = [synthetic_correlated_residuals(128, 3.0, 2.0, rng) for _ in range(20)]
        self.assertGreater(estimate_correlation_length(sets).residual_sd, 0.5)

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            estimate_correlation_length([np.zeros(32), np.zeros(64)])

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            estimate_correlation_length([])


class MaskPipelineTest(unittest.TestCase):
    def _disc(self, radius: float, size: int = 48) -> np.ndarray:
        rows, columns = np.mgrid[0:size, 0:size]
        return np.hypot(rows - size / 2, columns - size / 2) < radius

    def test_boundary_contour_is_ordered_around_the_centroid(self):
        contour = mask_boundary_contour(self._disc(15.0))
        centre = contour.mean(axis=0)
        angles = np.arctan2(contour[:, 0] - centre[0], contour[:, 1] - centre[1])
        self.assertTrue(np.all(np.diff(angles) >= -1e-9))

    def test_empty_mask_is_rejected(self):
        with self.assertRaises(ValueError):
            mask_boundary_contour(np.zeros((16, 16), dtype=bool))

    def test_a_localized_bias_yields_long_correlation(self):
        """A bias over one arc is exactly the error that costs a contour its samples."""

        size = 64
        rows, columns = np.mgrid[0:size, 0:size]
        radius = np.hypot(rows - 32, columns - 32)
        angle = np.arctan2(rows - 32, columns - 32)
        pairs = [(radius < 20, radius < (20 + 2.0 * (np.abs(angle) < 0.8)))]
        residuals = residuals_from_mask_pairs(pairs, spacing_mm=(1.5, 1.5))
        estimate = estimate_correlation_length(residuals)
        self.assertGreater(estimate.correlation_length_samples, 5.0)
        self.assertLess(estimate.effective_independent_samples, 20.0)

    def test_sample_spacing_converts_samples_to_millimetres(self):
        contour = mask_boundary_contour(self._disc(15.0), spacing_mm=(2.0, 2.0))
        spacing = contour_sample_spacing_mm(contour, 128)
        self.assertGreater(spacing, 0.0)
        self.assertLess(spacing, 5.0)


if __name__ == "__main__":
    unittest.main()
