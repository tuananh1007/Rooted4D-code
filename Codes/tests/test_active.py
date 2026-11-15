"""Tests for ActivePlaneTwin plane candidates and the selection criterion."""

import unittest

import numpy as np

from rooted4d.active import (
    PlaneCandidate,
    greedy_plan,
    long_axis_candidates,
    oblique_candidates,
    plane_contour_points,
    posterior_precision,
    quantity_variance,
    score_candidates,
    short_axis_stack,
)
from rooted4d.geometry import make_toy_complex


class PlaneCandidateTest(unittest.TestCase):
    def setUp(self):
        self.complex = make_toy_complex(24)

    def test_normal_is_normalized(self):
        candidate = PlaneCandidate("x", np.zeros(3), np.array([0.0, 0.0, 5.0]))
        self.assertAlmostEqual(float(np.linalg.norm(candidate.normal)), 1.0)

    def test_degenerate_normal_and_cost_are_rejected(self):
        with self.assertRaises(ValueError):
            PlaneCandidate("x", np.zeros(3), np.zeros(3))
        with self.assertRaises(ValueError):
            PlaneCandidate("x", np.zeros(3), np.array([0.0, 0.0, 1.0]), cost=0.0)

    def test_plane_frame_is_orthonormal(self):
        for candidate in short_axis_stack(self.complex.global_vertices, n_planes=3):
            plane = candidate.to_plane()
            self.assertAlmostEqual(
                float(plane.row_direction @ plane.column_direction), 0.0, places=10
            )

    def test_short_axis_planes_cut_the_chamber(self):
        for candidate in short_axis_stack(self.complex.global_vertices, n_planes=5)[1:-1]:
            points = plane_contour_points(
                self.complex.global_vertices, self.complex.faces["chamber"], candidate
            )
            self.assertGreater(len(points), 0, candidate.name)

    def test_a_plane_that_misses_returns_no_evidence(self):
        """A miss is a legitimate outcome the selector must score, not a crash."""

        far = PlaneCandidate(
            "far", np.array([0.0, 0.0, 1e4]), np.array([0.0, 0.0, 1.0])
        )
        points = plane_contour_points(
            self.complex.global_vertices, self.complex.faces["chamber"], far
        )
        self.assertEqual(len(points), 0)

    def test_oblique_planes_are_not_axis_aligned(self):
        for candidate in oblique_candidates(self.complex.global_vertices, n_rotations=3):
            self.assertLess(abs(float(candidate.normal[2])), 0.999)


class SelectionCriterionTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.dimension = 6
        self.prior = np.eye(self.dimension) * 1e-2
        self.gradients = rng.normal(size=(2, self.dimension))
        self.jacobians = [rng.normal(size=(8, self.dimension)) for _ in range(4)]
        self.candidates = [
            PlaneCandidate(f"P{i}", np.zeros(3), np.array([0.0, 0.0, 1.0]), cost=1.0)
            for i in range(4)
        ]

    def test_evidence_never_increases_posterior_variance(self):
        """Adding a measurement cannot make a Gaussian posterior less certain."""

        before = quantity_variance(self.prior, self.gradients)
        for jacobian in self.jacobians:
            after = quantity_variance(
                posterior_precision(self.prior, [jacobian], 1.0), self.gradients
            )
            self.assertLessEqual(after, before + 1e-12)

    def test_utility_is_reduction_divided_by_cost(self):
        expensive = [
            PlaneCandidate(f"P{i}", np.zeros(3), np.array([0.0, 0.0, 1.0]), cost=2.0)
            for i in range(4)
        ]
        cheap = score_candidates(
            self.candidates, self.jacobians, self.prior, self.gradients, 1.0
        )
        costly = score_candidates(
            expensive, self.jacobians, self.prior, self.gradients, 1.0
        )
        for a, b in zip(cheap, costly):
            self.assertAlmostEqual(a.reduction, b.reduction, places=12)
            self.assertAlmostEqual(a.utility, 2.0 * b.utility, places=12)

    def test_an_uninformative_plane_has_zero_utility(self):
        empty = np.zeros((0, self.dimension))
        scored = score_candidates(
            self.candidates[:1], [empty], self.prior, self.gradients, 1.0
        )
        self.assertAlmostEqual(scored[0].reduction, 0.0, places=12)

    def test_greedy_never_repeats_a_plane(self):
        steps = greedy_plan(
            self.candidates, self.jacobians, self.prior, self.gradients, 1.0, n_planes=4
        )
        names = [step.chosen for step in steps]
        self.assertEqual(len(names), len(set(names)))

    def test_greedy_variance_decreases_monotonically(self):
        steps = greedy_plan(
            self.candidates, self.jacobians, self.prior, self.gradients, 1.0, n_planes=4
        )
        variances = [step.variance_after for step in steps]
        for earlier, later in zip(variances, variances[1:]):
            self.assertLessEqual(later, earlier + 1e-12)

    def test_greedy_first_pick_maximizes_utility(self):
        scored = score_candidates(
            self.candidates, self.jacobians, self.prior, self.gradients, 1.0
        )
        best = max(scored, key=lambda entry: entry.utility).name
        steps = greedy_plan(
            self.candidates, self.jacobians, self.prior, self.gradients, 1.0, n_planes=1
        )
        self.assertEqual(steps[0].chosen, best)

    def test_stopping_rule_ends_acquisition_early(self):
        steps = greedy_plan(
            self.candidates, self.jacobians, self.prior, self.gradients, 1.0,
            n_planes=4, variance_target=float("inf"),
        )
        self.assertEqual(len(steps), 1)

    def test_mismatched_candidate_and_jacobian_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            score_candidates(
                self.candidates, self.jacobians[:2], self.prior, self.gradients, 1.0
            )


if __name__ == "__main__":
    unittest.main()
