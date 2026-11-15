"""Tests for the experiment campaign and its forward model."""

import unittest

import numpy as np

from rooted4d.experiments import EXPERIMENTS, CoupledForwardModel, run_all
from rooted4d.experiments.linearization import N_PARAMETERS, PARAMETER_BLOCKS
from rooted4d.theory import constrained_identifiability


class ForwardModelTest(unittest.TestCase):
    def test_blocks_tile_the_parameter_vector(self):
        covered = sorted(
            index
            for block in PARAMETER_BLOCKS.values()
            for index in range(*block.indices(N_PARAMETERS))
        )
        self.assertEqual(covered, list(range(N_PARAMETERS)))

    def test_prediction_is_deterministic(self):
        model = CoupledForwardModel(seed=1)
        first = model.predict(model.reference_parameters)
        second = model.predict(model.reference_parameters)
        np.testing.assert_array_equal(first, second)

    def test_jacobian_matches_a_coarser_finite_difference(self):
        model = CoupledForwardModel(seed=1)
        theta = model.reference_parameters
        analytic = model.jacobian(theta, step=1e-6)
        coarse = model.jacobian(theta, step=1e-4)
        scale = max(float(np.max(np.abs(analytic))), 1.0)
        self.assertLess(float(np.max(np.abs(analytic - coarse))) / scale, 1e-4)

    def test_dropping_tee_removes_all_pose_evidence(self):
        """Pose is observable only through TEE, so a CMR-only arm must be singular."""

        model = CoupledForwardModel(seed=0, tee_frames=np.array([], dtype=int))
        result = constrained_identifiability(
            model.jacobian(), parameter_blocks=PARAMETER_BLOCKS
        )
        self.assertFalse(result.identifiable)

    def test_both_modalities_are_identifiable(self):
        model = CoupledForwardModel(seed=0)
        result = constrained_identifiability(
            model.jacobian(), parameter_blocks=PARAMETER_BLOCKS
        )
        self.assertTrue(result.identifiable)
        self.assertGreater(result.smallest_tangent_eigenvalue, 1e-3)

    def test_removing_every_frame_is_rejected(self):
        with self.assertRaises(ValueError):
            CoupledForwardModel(
                cmr_frames=np.array([], dtype=int), tee_frames=np.array([], dtype=int)
            )


class CampaignTest(unittest.TestCase):
    def test_every_experiment_is_registered_and_named(self):
        self.assertIn("t1_annular_resolution", EXPERIMENTS)
        self.assertIn("e1_synthetic_recovery", EXPERIMENTS)
        for name, function in EXPERIMENTS.items():
            self.assertTrue(callable(function), name)

    def test_a_fast_experiment_reports_a_verdict_and_provenance(self):
        payload = run_all(("t1_annular_resolution",))
        self.assertIn("environment", payload)
        result = payload["results"]["t1_annular_resolution"]
        self.assertIn("verdict", result)
        self.assertTrue(result["verdict"]["all_resolutions_hold"])
        self.assertEqual(result["verdict"]["max_attachment_residual_mm"], 0.0)

    def test_unknown_experiment_is_rejected(self):
        with self.assertRaises(ValueError):
            run_all(("not_an_experiment",))

    def test_results_are_json_serializable(self):
        import json

        payload = run_all(("t3_gauge_ablation",))
        json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
