"""Tests for image-domain refinement against the section 4.5 operators."""

import unittest
from dataclasses import replace

import numpy as np

from rooted4d.inference import SceneModel, SolverConfig
from rooted4d.inference.image_refine import (
    ImageRefinementConfig,
    heldout_plane_error,
    heldout_tee_error,
    make_image_problem,
    refine_against_images,
    render_predicted,
)
from rooted4d.inference.problem import make_joint_problem, pose_errors


def _setup(seed: int = 0):
    model = SceneModel.procedural(n_modes=10, n_motion=3, n_session=2, n_phase_nodes=6)
    config = SolverConfig(fourier_order=2, session_order=1, max_iterations=20)
    refinement = ImageRefinementConfig(
        n_shape_modes=2, max_iterations=4, cmr_sigma_mm=3.0, tee_sigma_mm=3.0
    )
    truth = make_joint_problem(model, config, seed=seed).truth
    observations = make_image_problem(
        truth, model, config, refinement, seed=seed, n_cmr_frames=3,
        n_tee_frames=6, n_slices=4, image_shape=(8, 8), tee_grid_shape=(6, 5, 5),
    )
    return model, config, refinement, truth, observations


class ImageObservationTest(unittest.TestCase):
    def test_held_out_planes_are_disjoint_from_fitted_planes(self):
        _, _, _, _, observations = _setup()
        self.assertEqual(
            len(np.intersect1d(observations.fit_slices, observations.heldout_slices)), 0
        )
        self.assertGreater(len(observations.heldout_slices), 0)

    def test_held_out_tee_phases_are_withheld(self):
        _, _, _, _, observations = _setup()
        self.assertGreater(len(observations.heldout_tee_times), 0)
        self.assertEqual(
            len(np.intersect1d(observations.tee_times, observations.heldout_tee_times)), 0
        )

    def test_a_clean_reference_volume_is_recorded(self):
        """Held-out error against the noisy volume saturates at the noise floor."""

        _, _, _, _, observations = _setup()
        self.assertIsNotNone(observations.clean_cmr_volume)
        self.assertEqual(
            observations.clean_cmr_volume.shape, observations.cmr_volume.shape
        )

    def test_renderers_produce_bounded_likelihoods(self):
        model, config, refinement, truth, observations = _setup()
        cmr, tee = render_predicted(truth, observations, model, config, refinement)
        for volume in (cmr, tee):
            self.assertTrue(np.all(volume >= 0.0))
            self.assertTrue(np.all(volume <= 1.0))


class ImageRefinementTest(unittest.TestCase):
    def test_cmr_planes_are_blind_to_pose(self):
        """Pose enters only the TEE branch, so a pose shift leaves planes unchanged."""

        model, config, refinement, truth, observations = _setup()
        moved = replace(truth, pose_translation=truth.pose_translation + 5.0)
        first, _ = render_predicted(truth, observations, model, config, refinement)
        second, _ = render_predicted(moved, observations, model, config, refinement)
        np.testing.assert_allclose(first, second)

    def test_tee_sector_is_not_blind_to_pose(self):
        model, config, refinement, truth, observations = _setup()
        moved = replace(truth, pose_translation=truth.pose_translation + 5.0)
        _, first = render_predicted(truth, observations, model, config, refinement)
        _, second = render_predicted(moved, observations, model, config, refinement)
        self.assertGreater(float(np.max(np.abs(first - second))), 1e-6)

    def test_refinement_reduces_the_fitted_cost(self):
        model, config, refinement, truth, observations = _setup()
        rng = np.random.default_rng(3)
        start = replace(
            truth,
            pose_translation=truth.pose_translation + rng.normal(0.0, 1.5, 3),
            phase_offset=float((truth.phase_offset + 0.03) % 1.0),
        )
        result = refine_against_images(start, observations, model, config, refinement)
        self.assertLessEqual(result.cost_after, result.cost_before)
        self.assertEqual(result.n_parameters, 7 + refinement.n_shape_modes)

    def test_refinement_improves_a_degraded_pose(self):
        model, config, refinement, truth, observations = _setup()
        rng = np.random.default_rng(4)
        start = replace(
            truth, pose_translation=truth.pose_translation + rng.normal(0.0, 2.0, 3)
        )
        result = refine_against_images(start, observations, model, config, refinement)
        self.assertLess(
            pose_errors(result.parameters, truth)["translation_mm"],
            pose_errors(start, truth)["translation_mm"],
        )

    def test_heldout_metrics_are_reported(self):
        model, config, refinement, truth, observations = _setup()
        planes = heldout_plane_error(truth, observations, model, config, refinement)
        tee = heldout_tee_error(truth, observations, model, config, refinement)
        self.assertIn("heldout_plane_rmse_vs_clean", planes)
        self.assertIn("heldout_tee_rmse", tee)


if __name__ == "__main__":
    unittest.main()
