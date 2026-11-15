"""The two published fusion designs re-implemented for E114.

What the comparison depends on: the two-landmark timing is a monotone circular
map that its inverse undoes; the radial field passes through the displacements
it was given; spatio-temporal registration shares late fusion's fits, so before
its deformation it scores exactly as late fusion does, and its deformation
closes the ring; model-based fusion has one model, so no gap, and its inverse
transform undoes its forward transform.
"""

import unittest

import numpy as np

from rooted4d.geometry.rigid import RigidTransform
from rooted4d.inference import SceneModel, SolverConfig
from rooted4d.inference.baselines import run_arm
from rooted4d.inference.initialization import input_derived_alignment
from rooted4d.inference.problem import make_joint_problem
from rooted4d.inference.published_fusion import (
    RadialField,
    _to_tee_frame,
    _gaussian,
    farthest_points,
    fit_model_based_fusion,
    piecewise_circular_map,
    ring_area,
)


class HelpersTest(unittest.TestCase):
    def test_the_timing_map_hits_its_landmarks_and_inverts(self):
        source, target = (0.10, 0.45), (0.92, 0.30)
        self.assertAlmostEqual(float(piecewise_circular_map(0.10, source, target)), 0.92)
        self.assertAlmostEqual(float(piecewise_circular_map(0.45, source, target)), 0.30)
        x = np.linspace(0.0, 1.0, 101, endpoint=False)
        back = piecewise_circular_map(piecewise_circular_map(x, source, target), target, source)
        np.testing.assert_allclose(np.mod(back - x + 0.5, 1.0) - 0.5, 0.0, atol=1e-12)
        unwrapped = np.mod(piecewise_circular_map(x + 0.10, source, target) - 0.92, 1.0)
        self.assertTrue(np.all(np.diff(unwrapped) > 0))

    def test_coincident_landmarks_give_an_offset(self):
        x = np.array([0.0, 0.25, 0.8])
        np.testing.assert_allclose(piecewise_circular_map(x, (0.3, 0.3), (0.5, 0.5)),
                                   np.mod(x + 0.2, 1.0))

    def test_the_radial_field_passes_through_its_displacements(self):
        rng = np.random.default_rng(1)
        sources = rng.normal(scale=10.0, size=(20, 3))
        displacements = rng.normal(size=(20, 3))
        field = RadialField(sources, displacements, width=5.0)
        np.testing.assert_allclose(field(sources), displacements, atol=1e-4)

    def test_a_wide_field_on_dense_nodes_stays_bounded(self):
        # 96 nodes 1.7 mm apart: exact Gaussian interpolation at 10 mm swings far
        # from the nodes; the two-level field must close the nodes and stay bounded.
        angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        ring = np.stack([26 * np.cos(angles), 26 * np.sin(angles), 3 * np.cos(2 * angles)], axis=1)
        edge = np.stack([12 * np.cos(angles), 12 * np.sin(angles), 3 * np.cos(2 * angles) - 6], axis=1)
        displacements = np.stack([1.5 * np.cos(angles + 0.3) + 0.5, np.sin(2 * angles),
                                  0.8 * np.cos(3 * angles) - 0.4], axis=1)
        displacements += np.random.default_rng(0).normal(scale=0.05, size=displacements.shape)
        largest = np.linalg.norm(displacements, axis=1).max()
        for width in (1.25, 5.0, 10.0, 40.0):
            field = RadialField(ring, displacements, width)
            np.testing.assert_allclose(field(ring), displacements, atol=1e-4)
            self.assertLess(np.linalg.norm(field(edge), axis=1).max(), largest)

    def test_ring_area_of_a_circle(self):
        angles = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
        ring = np.stack([10 * np.cos(angles), 10 * np.sin(angles), np.zeros_like(angles)], axis=1)
        self.assertAlmostEqual(float(ring_area(ring)), np.pi * 100.0, delta=0.05)


class ArmsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SolverConfig(max_iterations=25)
        cls.model = SceneModel.procedural(n_modes=10, n_motion=3, n_session=2, n_phase_nodes=6)
        cls.problem = make_joint_problem(cls.model, cls.config, seed=5)
        cls.alignment = input_derived_alignment(cls.problem.observations)

    def test_spatiotemporal_registration_shares_late_fusions_fits(self):
        late = run_arm("late_fusion_trajectory", self.problem, self.config, alignment=self.alignment)
        ours = run_arm("spatiotemporal_registration", self.problem, self.config,
                       alignment=self.alignment, options={"widths_mm": [10.0, 40.0]})
        self.assertTrue(ours.succeeded)
        self.assertAlmostEqual(ours.heldout_cmr_mm, late.heldout_cmr_mm, places=9)
        self.assertAlmostEqual(ours.extras["undeformed_heldout_tee_mm"], late.heldout_tee_mm, places=9)
        for width in ("10", "40"):
            self.assertLess(ours.extras["by_width"][width]["attachment_residual_mm"], 1e-2)
        self.assertGreater(late.attachment_residual_mm, 1e-2)

    def test_a_narrow_field_moves_the_gap_into_the_hinge(self):
        ours = run_arm("spatiotemporal_registration", self.problem, self.config,
                       alignment=self.alignment, options={"widths_mm": [0.5, 40.0]})
        narrow, wide = (ours.extras["by_width"][w]["hinge_shear_mm"] for w in ("0.5", "40"))
        self.assertTrue(np.isfinite(narrow) and np.isfinite(wide))
        self.assertGreater(narrow, wide)
        self.assertGreater(ours.extras["hinge_template_distance_mm"], 0.0)

    def test_model_based_fusion_has_one_model_and_scores_every_endpoint(self):
        result = run_arm("model_based_fusion", self.problem, self.config, alignment=self.alignment,
                         options={"grid": [[1.0, 100.0], [4.0, 10.0]], "centers": 6})
        self.assertTrue(result.succeeded)
        self.assertEqual(result.attachment_residual_mm, 0.0)
        for key in ("heldout_cmr_mm", "heldout_tee_mm", "rotation_deg", "phase_map_cycles"):
            self.assertTrue(np.isfinite(getattr(result, key)), key)
        self.assertEqual(set(result.extras["by_config"]), {"1,100", "4,10"})

    def test_the_inverse_transform_undoes_the_forward_one(self):
        fit = fit_model_based_fusion(self.model, self.problem.observations, self.alignment,
                                     shape_weight=1.0, warp_ridge=1.0, centers=6, iterations=5)
        points = np.random.default_rng(2).normal(scale=8.0, size=(30, 3))
        placed = fit["rigid"].apply(points)
        forward = placed + _gaussian(placed, fit["anchors"], fit["width"]) @ fit["weights"]
        np.testing.assert_allclose(_to_tee_frame(fit, forward), points, atol=1e-6)

    def test_each_cmr_frame_is_paired_with_the_nearest_tee_frame(self):
        fit = fit_model_based_fusion(self.model, self.problem.observations, self.alignment,
                                     shape_weight=1.0, warp_ridge=100.0, centers=6, iterations=2)
        tee = np.asarray(fit["timing"].evaluate(np.asarray(self.problem.observations.tee_times)))
        cmr = np.mod(np.asarray(self.problem.observations.cmr_times), 1.0)
        for frame, partner in enumerate(fit["pairing"]):
            distances = np.abs(np.mod(tee - cmr[frame] + 0.5, 1.0) - 0.5)
            self.assertAlmostEqual(distances[partner], distances.min())

    def test_farthest_points_are_distinct(self):
        points = np.random.default_rng(3).normal(size=(50, 3))
        self.assertEqual(len(set(farthest_points(points, 12).tolist())), 12)


if __name__ == "__main__":
    unittest.main()
