"""Executable falsification checks for the section 4.9 theorem stack."""

import unittest

import numpy as np

from rooted4d.constraints import AttachmentMap, constraint_matrix_from_equalities
from rooted4d.geometry import make_toy_complex
from rooted4d.motion import FourierMotion, MonotoneCircularPhaseWarp
from rooted4d.theory import (
    check_assembly_nullspace_equivalence,
    constrained_identifiability,
    noise_stability_bound,
    soft_attachment_stationary_point,
    soft_attachment_sweep,
    spanning_tree_equalities,
)


class Theorem1AssemblyEquivalenceTest(unittest.TestCase):
    """im(A) = ker(C), with matching ranks and shared interface gradients."""

    def setUp(self):
        self.attachment_map = make_toy_complex(n_annulus=12).attachment_map

    def test_assembly_spans_exactly_the_constraint_nullspace(self):
        report = check_assembly_nullspace_equivalence(self.attachment_map)
        self.assertTrue(report.holds(), report.as_dict())
        self.assertLess(report.max_constraint_on_assembly, 1e-12)
        self.assertLess(report.max_projector_difference, 1e-10)

    def test_spanning_tree_equalities_are_independent(self):
        pairs = spanning_tree_equalities(self.attachment_map)
        n_local = sum(len(v) for v in self.attachment_map.local_to_global.values())
        constraint = constraint_matrix_from_equalities(n_local, pairs)
        rank = np.linalg.matrix_rank(constraint)
        self.assertEqual(rank, constraint.shape[0], "assumption A2 requires full row rank")
        self.assertEqual(
            constraint.shape[0],
            3 * (n_local - self.attachment_map.n_global_vertices),
        )

    def test_gradients_accumulate_at_shared_physical_vertices(self):
        report = check_assembly_nullspace_equivalence(self.attachment_map)
        self.assertLess(report.max_gradient_sharing_error, 1e-12)

    def test_a_detached_map_does_not_span_the_same_space(self):
        """A map without duplication has no interface constraints to satisfy."""

        detached = AttachmentMap(
            local_to_global={"only": np.arange(9, dtype=np.int64)},
            n_global_vertices=9,
        )
        report = check_assembly_nullspace_equivalence(detached)
        self.assertEqual(report.n_equality_rows, 0)
        self.assertEqual(report.nullspace_dimension, 27)


class Theorem2SoftPenaltyTest(unittest.TestCase):
    """A finite penalty weight leaves residual drift that decays only as 1/lambda."""

    def setUp(self):
        rng = np.random.default_rng(4)
        self.dimension = 10
        root = rng.normal(size=(self.dimension, self.dimension))
        self.metric = root @ root.T + self.dimension * np.eye(self.dimension)
        self.target = rng.normal(size=self.dimension)
        self.constraint = rng.normal(size=(3, self.dimension))

    def test_stationarity_identity_matches_the_closed_form(self):
        point = soft_attachment_stationary_point(
            self.constraint, self.metric, self.target, 1e3
        )
        self.assertAlmostEqual(
            point.attachment_residual, point.predicted_residual, delta=1e-9
        )

    def test_residual_decays_with_slope_near_minus_one(self):
        sweep = soft_attachment_sweep(
            self.constraint,
            self.metric,
            self.target,
            np.logspace(1.0, 6.0, 11),
        )
        self.assertAlmostEqual(sweep.slope, -1.0, delta=0.05)
        self.assertLess(sweep.max_identity_error, 1e-6)

    def test_no_finite_weight_reaches_the_assembly_tolerance(self):
        """The contrast the paper must report: soft drift versus exact assembly."""

        sweep = soft_attachment_sweep(
            self.constraint,
            self.metric,
            self.target,
            np.logspace(1.0, 8.0, 8),
        )
        self.assertGreater(float(sweep.attachment_residuals.min()), 1e-12)

        exact = check_assembly_nullspace_equivalence(
            make_toy_complex(n_annulus=12).attachment_map
        )
        self.assertLess(exact.max_constraint_on_assembly, 1e-12)

    def test_rank_deficient_constraint_is_rejected(self):
        duplicated = np.vstack([self.constraint, self.constraint[0]])
        with self.assertRaises(ValueError):
            soft_attachment_stationary_point(duplicated, self.metric, self.target, 10.0)


class Proposition4PeriodicityTest(unittest.TestCase):
    """Periodicity and degree-one phase ordering hold by construction."""

    def test_fourier_motion_closes_the_cycle_and_has_zero_mean(self):
        rng = np.random.default_rng(9)
        base = rng.normal(size=(20, 3))
        motion = FourierMotion(base=base, coefficients=rng.normal(size=(6, 20, 3)))
        self.assertLess(motion.seam_error(), 1e-12)
        phases = np.arange(64) / 64
        cycle_mean = motion.evaluate(phases).mean(axis=0) - base
        self.assertLess(float(np.max(np.abs(cycle_mean))), 1e-12)

    def test_phase_warp_is_strictly_increasing_over_exactly_one_turn(self):
        warp = MonotoneCircularPhaseWarp(np.linspace(-1.0, 1.0, 9), offset=0.37)
        times = np.linspace(0.0, 1.0, 200)
        lifted = warp.evaluate_unwrapped(times)
        self.assertTrue(np.all(np.diff(lifted) > 0.0))
        self.assertGreater(warp.minimum_increment(), 0.0)
        self.assertAlmostEqual(
            float(warp.evaluate_unwrapped(1.0) - warp.evaluate_unwrapped(0.0)),
            1.0,
            places=12,
        )


class Theorem5IdentifiabilityTest(unittest.TestCase):
    """ker(J_C) ∩ ker(J_E) = {0} iff the restricted Gauss-Newton matrix is positive definite."""

    def test_full_rank_evidence_is_identifiable(self):
        rng = np.random.default_rng(1)
        jacobian = rng.normal(size=(30, 6))
        result = constrained_identifiability(jacobian)
        self.assertTrue(result.identifiable)
        self.assertEqual(result.tangent_dimension, 6)
        self.assertGreater(result.smallest_tangent_eigenvalue, 0.0)

    def test_an_invisible_direction_is_detected(self):
        rng = np.random.default_rng(2)
        jacobian = rng.normal(size=(30, 6))
        jacobian[:, 4] = 0.0  # a parameter the evidence cannot see at all
        result = constrained_identifiability(jacobian)
        self.assertFalse(result.identifiable)
        self.assertAlmostEqual(abs(float(result.weakest_direction[4])), 1.0, places=8)

    def test_a_constraint_can_restore_identifiability(self):
        """Removing the blind direction from the tangent space makes H_T positive definite."""

        rng = np.random.default_rng(2)
        jacobian = rng.normal(size=(30, 6))
        jacobian[:, 4] = 0.0
        gauge = np.zeros((1, 6))
        gauge[0, 4] = 1.0
        result = constrained_identifiability(jacobian, constraint_jacobian=gauge)
        self.assertEqual(result.tangent_dimension, 5)
        self.assertTrue(result.identifiable)

    def test_weakest_direction_is_attributed_to_a_named_block(self):
        rng = np.random.default_rng(7)
        jacobian = rng.normal(size=(40, 9))
        jacobian[:, 6:] *= 1e-6  # a nearly unobserved session block
        result = constrained_identifiability(
            jacobian,
            parameter_blocks={
                "pose": slice(0, 3),
                "phase": slice(3, 6),
                "session": slice(6, 9),
            },
        )
        dominant = max(result.block_sensitivity, key=result.block_sensitivity.get)
        self.assertEqual(dominant, "session")

    def test_observation_weights_scale_the_restricted_matrix(self):
        rng = np.random.default_rng(3)
        jacobian = rng.normal(size=(20, 4))
        plain = constrained_identifiability(jacobian)
        scaled = constrained_identifiability(jacobian, weights=np.full(20, 4.0))
        self.assertAlmostEqual(
            scaled.smallest_tangent_eigenvalue,
            4.0 * plain.smallest_tangent_eigenvalue,
            places=8,
        )


class Theorem6NoiseStabilityTest(unittest.TestCase):
    """The local error bound scales as 1/sqrt(alpha) and degrades honestly."""

    def test_bound_scales_with_inverse_root_alpha(self):
        first = noise_stability_bound(1.0, 0.5)
        second = noise_stability_bound(4.0, 0.5)
        self.assertAlmostEqual(first.bound / second.bound, 2.0, places=10)

    def test_vanishing_alpha_makes_the_bound_uninformative(self):
        result = noise_stability_bound(0.0, 0.5)
        self.assertFalse(result.informative)
        self.assertEqual(result.bound, float("inf"))

    def test_prior_penalty_widens_rather_than_tightens_the_bound(self):
        without = noise_stability_bound(2.0, 0.3)
        with_prior = noise_stability_bound(2.0, 0.3, prior_penalty=1.0)
        self.assertGreater(with_prior.regularized_bound, without.regularized_bound)

    def test_empirical_error_respects_the_bound_on_a_linear_problem(self):
        """A linear model satisfies the secant assumption with c_nl = 0 exactly."""

        rng = np.random.default_rng(12)
        jacobian = rng.normal(size=(60, 5))
        truth = rng.normal(size=5)
        diagnostics = constrained_identifiability(jacobian)
        for trial in range(20):
            noise = np.random.default_rng(100 + trial).normal(scale=0.05, size=60)
            observed = jacobian @ truth + noise
            estimate = np.linalg.lstsq(jacobian, observed, rcond=None)[0]
            bound = noise_stability_bound(
                diagnostics.smallest_tangent_eigenvalue,
                float(np.linalg.norm(noise)),
            )
            self.assertLessEqual(float(np.linalg.norm(estimate - truth)), bound.bound)

    def test_invalid_nonlinearity_is_rejected(self):
        with self.assertRaises(ValueError):
            noise_stability_bound(1.0, 1.0, nonlinearity=1.0)


if __name__ == "__main__":
    unittest.main()
