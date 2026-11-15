"""The fusion designs built from recent methods for E115.

What the comparison depends on: the curve gap sees a ring that slides along
itself as attached and a ring that is offset as detached; the BCPD port leaves
identical sets in place, pulls a deformed ring onto its target, and its field
at the source is its displacement; NDP's landmark mode moves landmarks onto
their targets; the attachment arms share late fusion's fits, timing and pose,
so before the attachment they score exactly as late fusion does; GPMM fusion
has one model, so no gap.
"""

import importlib.util
import unittest

import numpy as np

from rooted4d.inference import SceneModel, SolverConfig
from rooted4d.inference.baselines import run_arm
from rooted4d.inference.initialization import input_derived_alignment
from rooted4d.inference.problem import make_joint_problem
from rooted4d.inference.recent_fusion import (
    bcpd_nonrigid,
    circular_interpolate,
    curve_gap,
    fit_gpmm_fusion,
    gpmm_kernel,
    ring_order,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _ring(count=96, radius=26.0, phase=0.0):
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False) + phase
    return np.stack([radius * np.cos(angles), radius * np.sin(angles), 3.0 * np.cos(2 * angles)], axis=1)


class CurvesTest(unittest.TestCase):
    def test_ring_order_recovers_the_cycle(self):
        ring = _ring()
        shuffled = np.random.default_rng(0).permutation(len(ring))
        ordered = ring[shuffled][ring_order(ring[shuffled])]
        steps = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
        self.assertLess(steps.max(), 1.2 * np.median(steps))

    def test_a_sliding_ring_is_attached_and_an_offset_ring_is_not(self):
        ring = _ring()
        slid = _ring(phase=np.pi / 96)  # half a node spacing along the ring
        self.assertGreater(np.linalg.norm(slid - ring, axis=1).max(), 0.8)
        self.assertLess(curve_gap(slid, ring), 0.05)
        self.assertAlmostEqual(curve_gap(ring + np.array([0.0, 0.0, 1.0]), ring), 1.0, places=6)


class BcpdTest(unittest.TestCase):
    def test_identical_sets_stay_in_place(self):
        ring = _ring()
        fit = bcpd_nonrigid(ring, ring, beta=0.5, lam=1.0)
        np.testing.assert_allclose(fit["deformed"], ring, atol=1e-3)

    def test_a_deformed_ring_is_pulled_onto_its_target(self):
        target = _ring()
        angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        source = target + np.stack([1.5 * np.cos(angles) + 0.5, np.sin(2 * angles), 0.8 * np.cos(3 * angles)], 1)
        before = curve_gap(source, target)
        fit = bcpd_nonrigid(source, target, beta=0.5, lam=0.01)
        self.assertLess(curve_gap(fit["deformed"], target), 0.2 * before)
        np.testing.assert_allclose(source + fit["field"](source), fit["deformed"], atol=1e-9)
        stiff = bcpd_nonrigid(source, target, beta=0.5, lam=1e4)
        self.assertGreater(curve_gap(stiff["deformed"], target), curve_gap(fit["deformed"], target))


class InterpolationTest(unittest.TestCase):
    def test_circular_interpolation_hits_its_knots_and_wraps(self):
        known = np.array([0.1, 0.4, 0.9])
        values = np.array([[1.0], [4.0], [9.0]])
        np.testing.assert_allclose(circular_interpolate(known, values, known), values)
        # halfway from 0.9 to 1.1 across the wrap
        np.testing.assert_allclose(circular_interpolate(known, values, np.array([0.0])), [[5.0]])
        np.testing.assert_allclose(circular_interpolate(known, values, np.array([0.25])), [[2.5]])


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class NdpTest(unittest.TestCase):
    def test_landmarks_move_onto_their_targets(self):
        from rooted4d.inference.ndp import landmark_pyramid

        source = _ring()
        angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        target = source + np.stack([1.5 * np.cos(angles), np.sin(2 * angles), 0.8 * np.cos(3 * angles)], 1)
        levels = landmark_pyramid(source, target, [source], levels=9, iterations=200)
        errors = [np.linalg.norm(level[0] - target, axis=1).mean() for level in levels]
        self.assertLess(errors[-1], 0.5 * np.linalg.norm(source - target, axis=1).mean())
        self.assertLessEqual(errors[-1], errors[0] + 1e-6)


class ArmsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SolverConfig(max_iterations=25)
        cls.model = SceneModel.procedural(n_modes=10, n_motion=3, n_session=2, n_phase_nodes=6)
        cls.problem = make_joint_problem(cls.model, cls.config, seed=5)
        cls.alignment = input_derived_alignment(cls.problem.observations)
        cls.late = run_arm("late_fusion_trajectory", cls.problem, cls.config, alignment=cls.alignment)

    def _shares_late_fusion(self, ours):
        self.assertTrue(ours.succeeded)
        self.assertAlmostEqual(ours.heldout_cmr_mm, self.late.heldout_cmr_mm, places=9)
        self.assertAlmostEqual(ours.extras["undeformed_heldout_tee_mm"], self.late.heldout_tee_mm, places=9)
        self.assertAlmostEqual(ours.rotation_deg, self.late.rotation_deg, places=9)
        self.assertAlmostEqual(ours.phase_map_cycles, self.late.phase_map_cycles, places=12)
        self.assertAlmostEqual(ours.extras["late_fusion_gap_mm"], self.late.attachment_residual_mm, places=9)

    def test_bcpd_attachment_shares_late_fusion_and_closes_the_curve_gap(self):
        ours = run_arm("late_fusion_bcpd", self.problem, self.config, alignment=self.alignment,
                       options={"grid": [[0.5, 0.01], [0.5, 1e4]]})
        self._shares_late_fusion(ours)
        loose, stiff = (ours.extras["by_config"][k] for k in ("0.5,0.01", "0.5,10000"))
        self.assertLess(loose["curve_gap_mm"], ours.extras["late_fusion_curve_gap_mm"])
        self.assertLess(loose["curve_gap_mm"], stiff["curve_gap_mm"])
        self.assertTrue(np.isfinite(loose["hinge_shear_mm"]))

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_ndp_attachment_shares_late_fusion_and_closes_the_gap(self):
        ours = run_arm("late_fusion_ndp", self.problem, self.config, alignment=self.alignment,
                       options={"levels": 2, "iterations": 100})
        self._shares_late_fusion(ours)
        self.assertEqual(set(ours.extras["by_config"]), {"1", "2"})
        self.assertLess(ours.extras["by_config"]["2"]["attachment_residual_mm"], self.late.attachment_residual_mm)

    def test_gpmm_fusion_has_one_model_and_scores_every_endpoint(self):
        result = run_arm("gpmm_fusion", self.problem, self.config, alignment=self.alignment,
                         options={"grid": [[0.0, 0.0, 1.0], [2.0, 20.0, 0.5]], "iterations": 5})
        self.assertTrue(result.succeeded)
        self.assertLess(result.attachment_residual_mm, 1e-9)
        for key in ("heldout_cmr_mm", "heldout_tee_mm", "rotation_deg", "phase_map_cycles"):
            self.assertTrue(np.isfinite(getattr(result, key)), key)
        self.assertEqual(set(result.extras["by_config"]), {"0,0,1", "2,20,0.5"})

    def test_the_gpmm_kernel_is_a_covariance(self):
        kernel = gpmm_kernel(self.model, 2.0, 20.0, 0.005)
        np.testing.assert_allclose(kernel, kernel.T, atol=1e-9)
        self.assertGreater(np.linalg.eigvalsh(kernel).min(), -1e-8 * np.abs(kernel).max())

    def test_a_flexible_gpmm_follows_the_cmr_evidence(self):
        observations = self.problem.observations
        fit = fit_gpmm_fusion(self.model, observations, self.alignment, gaussian_scale=4.0,
                              gaussian_width=20.0, noise=0.05, iterations=3)
        rows = self.model.observation_rows(np.asarray(observations.cmr_vertices))
        order = np.argsort(np.mod(np.asarray(observations.cmr_times), 1.0))
        fitted = self.model.template_physical[rows][None] + fit["deformations"][:, rows]
        observed = np.asarray(observations.cmr_points)[order]
        self.assertLess(np.linalg.norm(fitted - observed, axis=-1).mean(), 0.5)


if __name__ == "__main__":
    unittest.main()
