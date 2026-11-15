"""Late fusion with E116's residual field (E119).

What the comparison depends on: with both scales zero the arm is late fusion
(held-out errors, gap, pose and timing); the per-modality field of each
reconstruction is that reconstruction's own field, computed from its own
residuals; the shared field reaches the TEE reconstruction through the
registration and leaves late fusion's gap unchanged; and the TEE residuals reach
the shared field in the CMR fit's frame.
"""

import unittest
from dataclasses import replace

import numpy as np

from rooted4d.inference import SceneModel, SolverConfig
from rooted4d.inference.baselines import late_fusion_trajectory_fit, run_arm
from rooted4d.inference.initialization import input_derived_alignment
from rooted4d.inference.joint import latent_trajectory
from rooted4d.inference.late_fusion_field import late_fusion_evidence
from rooted4d.inference.problem import make_joint_problem
from rooted4d.inference.residual_field import (
    evaluate_field,
    field_evidence,
    field_scales,
    gaussian_factor,
    normal_equations,
    solve_field,
)

CONFIGS = [["session", 0.0, 0.0, 20.0], ["session", 2.0, 1.0, 20.0], ["shared", 2.0, 1.0, 20.0],
           ["shared", 4.0, 0.0, 10.0]]
OVERRIDES = {"cmr": {"identity_prior_weight": 0.5}, "tee": {"identity_prior_weight": 2.0}}


class LateFusionFieldTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SolverConfig(max_iterations=25)
        cls.model = SceneModel.procedural(n_modes=10, n_motion=3, n_session=2, n_phase_nodes=6)
        cls.problem = make_joint_problem(cls.model, cls.config, seed=5)
        cls.alignment = input_derived_alignment(cls.problem.observations)
        options = {"modality_overrides": OVERRIDES}
        cls.late = run_arm("late_fusion_trajectory", cls.problem, cls.config, alignment=cls.alignment,
                           options=options)
        cls.result = run_arm("late_fusion_residual_field", cls.problem, cls.config, alignment=cls.alignment,
                             options=dict(options, configs=CONFIGS))
        _, cls.parts = late_fusion_trajectory_fit(cls.problem, cls.config, alignment=cls.alignment,
                                                  modality_overrides=OVERRIDES)

    def test_zero_scales_are_late_fusion(self):
        ours, late = self.result, self.late
        self.assertTrue(ours.succeeded)
        for key in ("heldout_cmr_mm", "heldout_tee_mm", "attachment_residual_mm", "rotation_deg",
                    "translation_mm", "phase_map_cycles"):
            self.assertAlmostEqual(getattr(ours, key), getattr(late, key), places=10, msg=key)
        for key in ("heldout_cmr_mm", "heldout_tee_mm", "attachment_residual_mm"):
            self.assertAlmostEqual(ours.extras["late_fusion"][key], getattr(late, key), places=12, msg=key)
        self.assertGreater(late.attachment_residual_mm, 1e-3)

    def test_each_reconstruction_gets_its_own_field(self):
        cmr_fit, tee_fit, _, _, _ = self.parts
        observations, model, order = self.problem.observations, self.model, self.config.fourier_order
        factor = gaussian_factor(model.template_physical, 20.0)
        scales = field_scales(2.0, 1.0, order)
        cmr_field = solve_field(*normal_equations([field_evidence(cmr_fit)["cmr"]], factor, order), factor, scales)
        tee_field = solve_field(*normal_equations([field_evidence(tee_fit)["tee"]], factor, order), factor, scales)

        times = np.asarray(observations.heldout_cmr_times)
        rows = model.observation_rows(np.asarray(observations.heldout_cmr_vertices))
        geometry = model.physical_coordinates(latent_trajectory(cmr_fit.parameters, times, "cmr", model,
                                                                cmr_fit.config))
        cmr = np.linalg.norm((geometry + evaluate_field(cmr_field, times, order))[:, rows]
                             - observations.heldout_cmr_points, axis=-1).mean()
        phases = tee_fit.parameters.warp().evaluate(np.asarray(observations.heldout_tee_times))
        rows = model.observation_rows(np.asarray(observations.heldout_tee_vertices))
        geometry = model.physical_coordinates(latent_trajectory(tee_fit.parameters, phases, "tee", model,
                                                                tee_fit.config))
        placed = tee_fit.parameters.transform().apply((geometry + evaluate_field(tee_field, phases, order))[:, rows])
        tee = np.linalg.norm(placed - observations.heldout_tee_points, axis=-1).mean()

        scores = self.result.extras["by_config"]["session/2,1,20"]
        self.assertAlmostEqual(cmr, scores["heldout_cmr_mm"], places=10)
        self.assertAlmostEqual(tee, scores["heldout_tee_mm"], places=10)
        self.assertNotAlmostEqual(cmr, self.late.heldout_cmr_mm, places=6)
        self.assertNotEqual(scores["attachment_residual_mm"], self.late.attachment_residual_mm)

    def test_the_shared_field_leaves_the_gap_and_differs_from_the_per_modality_one(self):
        by_config = self.result.extras["by_config"]
        self.assertEqual(set(by_config), {"session/0,0,20", "session/2,1,20", "shared/2,1,20", "shared/4,0,10"})
        for name in ("shared/2,1,20", "shared/4,0,10"):
            self.assertAlmostEqual(by_config[name]["attachment_residual_mm"], self.late.attachment_residual_mm,
                                   places=9, msg=name)
            self.assertGreater(by_config[name]["field_rms_mm"], 0.0)
        self.assertNotEqual(by_config["shared/2,1,20"]["heldout_tee_mm"], by_config["session/2,1,20"]["heldout_tee_mm"])

    def test_tee_residuals_reach_the_shared_field_in_the_cmr_frame(self):
        cmr_fit, tee_fit, _, _, registration = self.parts
        model, observations = self.model, self.problem.observations
        # TEE evidence that is the TEE fit's own geometry moved by an offset given in the CMR fit's frame.
        offset = np.array([2.0, -1.0, 1.5])
        rotation = np.asarray(registration.transform.rotation)
        phases = np.asarray(tee_fit.parameters.warp().evaluate(np.asarray(observations.tee_times)))
        geometry = model.physical_coordinates(latent_trajectory(tee_fit.parameters, phases, "tee", model,
                                                                tee_fit.config))
        rows = model.observation_rows(np.asarray(observations.tee_vertices))
        moved = tee_fit.parameters.transform().apply(geometry[:, rows] + offset @ rotation)
        shifted = replace(tee_fit, observations=replace(tee_fit.observations, tee_points=moved))
        evidence = late_fusion_evidence(cmr_fit, shifted, registration)
        np.testing.assert_allclose(evidence["tee_in_cmr"].residuals, np.broadcast_to(offset, moved.shape),
                                   atol=1e-9)
        np.testing.assert_allclose(evidence["tee_in_cmr"].phases, registration.warp.evaluate(phases), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
