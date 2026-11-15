"""Rooted4D with a Gaussian-kernel residual field (E116).

What the comparison depends on: the closed-form solve is the GP posterior mean
of the stated prior (checked against the dense GP in observation space); with
both scales zero the arm is Rooted4D; any field keeps the gap at zero, because
it lives on the physical vertices; the TEE residuals reach the field in the
CMR frame; the public fit gives the benchmark run's numbers; and the frozen
main configuration is the validation selection, with reference rows that
match its medians.
"""

import json
import unittest
from dataclasses import replace

import numpy as np

from rooted4d.inference import SceneModel, SolverConfig
from rooted4d.inference.baselines import run_arm
from rooted4d.inference.initialization import input_derived_alignment, start_from
from rooted4d.inference.joint import fit_joint_scene, latent_trajectory
from rooted4d.inference.problem import make_joint_problem
from rooted4d.inference.residual_field import (
    FieldEvidence,
    FieldSettings,
    evaluate_field,
    field_evidence,
    field_scales,
    fit_rooted4d_with_field,
    gaussian_factor,
    normal_equations,
    solve_field,
    temporal_design,
)
from rooted4d.paths import workspace_root


class SolveTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.template = rng.uniform(-10.0, 10.0, size=(12, 3))
        self.order = 2
        self.pieces = [
            FieldEvidence(phases=rng.uniform(0, 1, 5), rows=np.array([0, 2, 3, 3, 7]),
                          residuals=rng.normal(size=(5, 5, 3))),
            FieldEvidence(phases=rng.uniform(0, 1, 4), rows=np.array([1, 3, 9, 11]),
                          residuals=rng.normal(size=(4, 4, 3))),
        ]

    def test_the_design_has_a_constant_and_the_harmonics(self):
        design = temporal_design(np.array([0.0, 0.25]), 2)
        np.testing.assert_allclose(design[:, 0], 1.0)
        np.testing.assert_allclose(design[1], [1.0, 0.0, 1.0, -1.0, 0.0], atol=1e-12)

    def test_the_factor_reproduces_the_kernel(self):
        factor = gaussian_factor(self.template, 6.0)
        squared = ((self.template[:, None] - self.template[None]) ** 2).sum(-1)
        np.testing.assert_allclose(factor @ factor.T, np.exp(-squared / 72.0), atol=1e-9)

    def test_the_solve_is_the_dense_gp_posterior_mean(self):
        width, scales = 6.0, field_scales(1.5, 0.7, self.order)
        factor = gaussian_factor(self.template, width)
        coefficients = solve_field(*normal_equations(self.pieces, factor, self.order), factor, scales)

        rows = np.concatenate([np.repeat(p.rows[None], len(p.phases), 0).ravel() for p in self.pieces])
        phases = np.concatenate([np.repeat(p.phases, len(p.rows)) for p in self.pieces])
        values = np.concatenate([p.residuals.reshape(-1, 3) for p in self.pieces])

        def covariance(rows_a, phases_a, rows_b, phases_b):
            points_a, points_b = self.template[rows_a], self.template[rows_b]
            spatial = np.exp(-((points_a[:, None] - points_b[None]) ** 2).sum(-1) / (2 * width ** 2))
            temporal = (temporal_design(phases_a, self.order) * scales ** 2) @ temporal_design(
                phases_b, self.order).T
            return spatial * temporal

        weights = np.linalg.solve(covariance(rows, phases, rows, phases) + np.eye(len(rows)), values)
        queries = np.array([0.1, 0.55, 0.9])
        every = np.arange(len(self.template))
        for tau, field in zip(queries, evaluate_field(coefficients, queries, self.order)):
            dense = covariance(every, np.full(len(every), tau), rows, phases) @ weights
            np.testing.assert_allclose(field, dense, atol=1e-8)

    def test_zero_scales_give_no_field(self):
        factor = gaussian_factor(self.template, 6.0)
        coefficients = solve_field(*normal_equations(self.pieces, factor, self.order), factor,
                                   field_scales(0.0, 0.0, self.order))
        self.assertEqual(np.abs(coefficients).max(), 0.0)


class ArmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SolverConfig(max_iterations=25)
        cls.model = SceneModel.procedural(n_modes=10, n_motion=3, n_session=2, n_phase_nodes=6)
        cls.problem = make_joint_problem(cls.model, cls.config, seed=5)
        cls.alignment = input_derived_alignment(cls.problem.observations)
        cls.rooted = run_arm("rooted4d_full", cls.problem, cls.config, alignment=cls.alignment)
        cls.result = run_arm(
            "rooted4d_residual_field", cls.problem, cls.config, alignment=cls.alignment,
            options={"configs": [["shared", 0.0, 0.0, 20.0], ["shared", 2.0, 1.0, 20.0],
                                 ["session", 2.0, 1.0, 20.0], ["shared", 4.0, 0.0, 10.0]]})

    def test_zero_scales_are_rooted4d(self):
        ours = self.result
        self.assertTrue(ours.succeeded)
        for key in ("heldout_cmr_mm", "heldout_tee_mm", "rotation_deg", "translation_mm", "phase_map_cycles"):
            self.assertAlmostEqual(getattr(ours, key), getattr(self.rooted, key), places=12, msg=key)
        self.assertAlmostEqual(ours.extras["rooted4d"]["heldout_tee_mm"], self.rooted.heldout_tee_mm, places=12)

    def test_every_field_keeps_the_valve_attached(self):
        by_config = self.result.extras["by_config"]
        self.assertEqual(set(by_config), {"shared/0,0,20", "shared/2,1,20", "session/2,1,20", "shared/4,0,10"})
        for name, scores in by_config.items():
            self.assertLess(scores["attachment_residual_mm"], 1e-9, name)
            self.assertTrue(np.isfinite(scores["heldout_cmr_mm"]) and np.isfinite(scores["heldout_tee_mm"]), name)
        self.assertGreater(by_config["shared/2,1,20"]["field_rms_mm"], 0.0)
        self.assertNotEqual(by_config["shared/2,1,20"]["heldout_tee_mm"],
                            by_config["session/2,1,20"]["heldout_tee_mm"])

    def test_tee_residuals_reach_the_field_in_the_cmr_frame(self):
        model, config, observations = self.model, self.config, self.problem.observations
        fit = fit_joint_scene(observations, model, config, start=start_from(self.alignment, model, config))
        # TEE evidence that is the fit's own geometry moved by an offset in the CMR frame.
        offset = np.array([2.0, -1.0, 1.5])
        phases = np.asarray(fit.parameters.warp().evaluate(np.asarray(observations.tee_times)))
        geometry = model.physical_coordinates(latent_trajectory(fit.parameters, phases, "tee", model, config))
        rows = model.observation_rows(np.asarray(observations.tee_vertices))
        moved = fit.parameters.transform().apply(geometry[:, rows] + offset)
        shifted = replace(fit, observations=replace(observations, tee_points=moved))
        evidence = field_evidence(shifted)
        np.testing.assert_allclose(evidence["tee"].residuals, np.broadcast_to(offset, moved.shape), atol=1e-9)
        factor = gaussian_factor(model.template_physical, 80.0)
        coefficients = solve_field(*normal_equations([evidence["tee"]], factor, config.fourier_order),
                                   factor, field_scales(100.0, 0.0, config.fourier_order))
        recovered = evaluate_field(coefficients, phases, config.fourier_order)[:, rows]
        self.assertLess(np.abs(recovered - offset).max(), 0.05)
    def test_the_public_fit_gives_the_benchmark_run_numbers(self):
        observations = self.problem.observations
        for sharing in ("shared", "session"):
            settings = FieldSettings(sharing, 2.0, 1.0, 20.0)
            fit, field = fit_rooted4d_with_field(observations, self.model, self.config, settings,
                                                 alignment=self.alignment)
            geometry = field.geometry(fit, observations.heldout_cmr_times, "cmr")
            rows = self.model.observation_rows(np.asarray(observations.heldout_cmr_vertices))
            cmr = np.linalg.norm(geometry[:, rows] - observations.heldout_cmr_points, axis=-1).mean()
            phases = fit.parameters.warp().evaluate(np.asarray(observations.heldout_tee_times))
            rows = self.model.observation_rows(np.asarray(observations.heldout_tee_vertices))
            placed = fit.parameters.transform().apply(field.geometry(fit, phases, "tee")[:, rows])
            tee = np.linalg.norm(placed - observations.heldout_tee_points, axis=-1).mean()
            scores = self.result.extras["by_config"][f"{sharing}/2,1,20"]
            self.assertAlmostEqual(cmr, scores["heldout_cmr_mm"], places=12)
            self.assertAlmostEqual(tee, scores["heldout_tee_mm"], places=12)


class MainConfigurationTest(unittest.TestCase):
    """The frozen file agrees with the validation selection and with its reference rows."""

    ROOT = workspace_root()
    CONFIG = ROOT / "Codes/configs/rooted4d_main.json"
    SELECTION = ROOT / "results/benchmark/E116/E116_val.json"

    def test_the_frozen_settings_are_the_validation_selection(self):
        config = json.loads(self.CONFIG.read_text())
        if not self.SELECTION.exists():
            self.skipTest("the validation report is not in this checkout")
        selection = json.loads(self.SELECTION.read_text())["selection"]
        for key, entry in config["benchmarks"].items():
            SolverConfig(**entry["solver"])
            for name, variant in (("field", "R4D+GP"), ("reported_variant", "R4D+GP-session")):
                FieldSettings(**entry[name])
                self.assertEqual([entry[name][k] for k in ("sharing", "static", "dynamic", "width")],
                                 selection[key][variant])
        self.assertEqual(config["benchmarks"]["v4n96o"]["field"]["sharing"], "shared")

    def test_the_reference_medians_match_their_rows(self):
        config = json.loads(self.CONFIG.read_text())
        for key, entry in config["benchmarks"].items():
            for level, reference in entry["reference"].items():
                path = self.ROOT / reference["rows"]
                if not path.exists():
                    self.skipTest("the reference rows are not in this checkout")
                rows = [json.loads(line)["R4D+GP"] for line in path.read_text().splitlines() if line.strip()]
                self.assertEqual(len(rows), 60)
                for endpoint in ("heldout_cmr_mm", "heldout_tee_mm"):
                    self.assertAlmostEqual(float(np.median([r[endpoint] for r in rows])),
                                           reference["medians_mm"]["R4D+GP"][endpoint], places=5)
                self.assertTrue(all(r["attachment_residual_mm"] == 0.0 for r in rows))


if __name__ == "__main__":
    unittest.main()
