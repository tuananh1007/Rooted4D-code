"""Tests for trajectory registration and the late-fusion arm built on it.

What the comparison depends on: the registration recovers a known rigid map,
offset and warp between two trajectories; the arm shares late fusion's two
fits exactly, so the two late-fusion arms differ in registration alone; and a
stored parameter set reads back unchanged.
"""

import json
import unittest
from dataclasses import replace

import numpy as np

from rooted4d.geometry.rigid import RigidTransform
from rooted4d.inference import SceneModel, SolverConfig
from rooted4d.inference.baselines import parameters_from_record, run_arm
from rooted4d.inference.initialization import input_derived_alignment
from rooted4d.inference.joint import initial_parameters
from rooted4d.inference.problem import make_joint_problem
from rooted4d.inference.trajectory_registration import (
    REGISTRATION_PHASES,
    TrajectoryRegistration,
    inverse_phase,
    register_trajectories,
    register_trajectory_arrays,
)
from rooted4d.motion import MonotoneCircularPhaseWarp, circular_distance, periodic_basis


def _analytic_reference(seed: int = 3, vertices: int = 14, order: int = 2):
    """A ring-like periodic trajectory with harmonics strong enough to time it."""

    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, vertices, endpoint=False)
    base = np.stack([20.0 * np.cos(angles), 20.0 * np.sin(angles),
                     3.0 * np.sin(2.0 * angles)], axis=1)
    coefficients = rng.normal(scale=2.0, size=(2 * order, vertices, 3))

    def reference(tau):
        design = periodic_basis(np.mod(np.asarray(tau, dtype=np.float64), 1.0), order)
        return base + np.einsum("pk,kvc->pvc", design, coefficients)

    return reference


def _moving(reference, transform: RigidTransform, warp: MonotoneCircularPhaseWarp):
    """Points that ``transform`` and ``warp`` map exactly onto ``reference``."""

    target = reference(warp.evaluate(REGISTRATION_PHASES))
    return transform.inverse().apply(target.reshape(-1, 3)).reshape(target.shape)


class TrajectoryRegistrationTest(unittest.TestCase):
    def setUp(self):
        self.config = SolverConfig(max_iterations=40)
        self.reference = _analytic_reference()
        self.transform = RigidTransform.from_axis_angle(
            np.array([0.03, -0.05, 0.08]), np.array([2.0, -1.5, 3.0]))

    def test_recovers_a_known_rigid_map_and_offset(self):
        warp = MonotoneCircularPhaseWarp(np.zeros(8), offset=0.3)
        found = register_trajectory_arrays(
            self.reference, _moving(self.reference, self.transform, warp), self.config, knots=8)
        np.testing.assert_allclose(found.transform.rotation, self.transform.rotation, atol=1e-5)
        np.testing.assert_allclose(found.transform.translation, self.transform.translation,
                                   atol=1e-4)
        self.assertLess(circular_distance(np.array([found.warp.offset]), np.array([0.3]))[0], 1e-5)
        self.assertLess(found.rms_mm, 1e-4)

    def test_a_nonuniform_warp_is_followed(self):
        logits = np.random.default_rng(7).normal(scale=0.6, size=8)
        warp = MonotoneCircularPhaseWarp(logits, offset=0.62)
        moving = _moving(self.reference, self.transform, warp)
        found = register_trajectory_arrays(self.reference, moving, self.config, knots=8)
        # The warp is penalized for acceleration, as in the joint fit, so it is
        # followed closely rather than exactly.
        error = circular_distance(found.warp.evaluate(REGISTRATION_PHASES),
                                  warp.evaluate(REGISTRATION_PHASES))
        self.assertLess(float(error.max()), 0.02)
        # Rotation within a fraction of a degree despite the warp.
        relative = found.transform.rotation @ self.transform.rotation.T
        angle = np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
        self.assertLess(angle, 0.5)

    def test_inverse_phase_inverts_the_warp(self):
        warp = MonotoneCircularPhaseWarp(np.random.default_rng(1).normal(size=8), offset=0.85)
        phases = np.linspace(0.0, 1.0, 50, endpoint=False)
        back = warp.evaluate(inverse_phase(warp, phases))
        self.assertLess(float(circular_distance(back, phases).max()), 1e-12)

    def test_place_composes_the_pose_with_the_inverse_registration(self):
        registration = TrajectoryRegistration(
            transform=self.transform, warp=MonotoneCircularPhaseWarp(np.zeros(8)),
            rms_mm=0.0, cost=0.0, iterations=0)
        pose = RigidTransform.from_axis_angle(np.array([-0.1, 0.2, 0.05]), np.array([5.0, 0.0, -2.0]))
        points = np.random.default_rng(2).normal(size=(10, 3))
        # A point in the moving fit's frame reaches the acquisition by the fit's
        # own pose; the same point in the reference frame must reach it by the
        # placed pose.
        np.testing.assert_allclose(registration.place(pose).apply(self.transform.apply(points)),
                                   pose.apply(points), atol=1e-12)


class LateFusionTrajectoryArmTest(unittest.TestCase):
    def setUp(self):
        self.model = SceneModel.procedural(n_modes=10, n_motion=3, n_session=2, n_phase_nodes=6)
        self.config = SolverConfig(fourier_order=2, session_order=1, max_iterations=30)
        self.problem = make_joint_problem(self.model, self.config, seed=5)
        self.alignment = input_derived_alignment(self.problem.observations)

    def test_a_trajectory_registers_to_itself_with_the_identity(self):
        rng = np.random.default_rng(11)
        start = initial_parameters(self.model, self.config)
        parameters = replace(start, identity=rng.normal(size=start.identity.shape),
                             motion=rng.normal(size=start.motion.shape))
        vertices = np.sort(self.model.groups["annulus"])
        found = register_trajectories(parameters, parameters, self.model, self.config, vertices)
        np.testing.assert_allclose(found.transform.rotation, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(found.transform.translation, 0.0, atol=1e-5)
        self.assertLess(circular_distance(np.array([found.warp.offset]), np.array([0.0]))[0], 1e-6)
        self.assertLess(found.rms_mm, 1e-5)

    def test_shares_late_fusions_fits_and_differs_in_registration_only(self):
        feature = run_arm("late_fusion", self.problem, self.config, alignment=self.alignment)
        trajectory = run_arm("late_fusion_trajectory", self.problem, self.config,
                             alignment=self.alignment)
        self.assertTrue(trajectory.succeeded, trajectory.failure_reason)
        self.assertEqual(trajectory.baseline_id, "late_fusion_trajectory")
        for key in ("heldout_cmr_mm", "heldout_tee_mm", "heldout_cmr_p95_mm",
                    "heldout_tee_p95_mm", "cost"):
            self.assertEqual(getattr(trajectory, key), getattr(feature, key), key)
        for key in ("rotation_deg", "translation_mm", "phase_error_cycles", "phase_map_cycles",
                    "attachment_residual_mm", "repair_registration_rms_mm"):
            value = getattr(trajectory, key)
            self.assertIsNotNone(value, key)
            self.assertTrue(np.isfinite(value), key)
        # Two separately fitted rings never coincide exactly.
        self.assertGreater(trajectory.attachment_residual_mm, 1e-6)
        self.assertNotIn("extras", trajectory.as_dict())

    def test_the_composed_estimates_point_the_right_way(self):
        """Pose P_T o G^-1 and phase map chi o phi_T, checked against the truth.

        Finite numbers alone would not catch a swapped registration: with the
        reference and moving fits exchanged, or chi inverted, this problem gives
        about 0.29 deg, 0.39 mm and a phase-map error of 0.33 cycles, against
        0.16 deg, 0.11 mm and 0.009 cycles with the conventions right.
        """

        trajectory = run_arm("late_fusion_trajectory", self.problem, self.config,
                             alignment=self.alignment)
        self.assertLess(trajectory.phase_map_cycles, 0.05)
        self.assertLess(trajectory.translation_mm, 0.25)
        self.assertLess(trajectory.rotation_deg, 0.25)

    def test_stored_parameters_read_back_unchanged(self):
        """The stored parameters are the fits that were scored, bit for bit, after the driver's serializer."""

        import importlib.util
        from pathlib import Path

        from rooted4d.inference.baselines import _late_fusion_fits, _start
        from rooted4d.inference.joint import fit_joint_scene

        driver_path = (Path(__file__).resolve().parents[1] / "scripts" / "experiments"
                       / "e59_unknown_alignment.py")
        specification = importlib.util.spec_from_file_location("e59_driver", driver_path)
        driver = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(driver)

        def through_json(record):
            return json.loads(json.dumps(driver._clean(record)))

        def assert_same(read_back, fitted):
            for field in ("pose_axis_angle", "pose_translation", "phase_logits", "identity", "motion"):
                np.testing.assert_array_equal(getattr(read_back, field), getattr(fitted, field), field)
            self.assertEqual(read_back.phase_offset, fitted.phase_offset)
            self.assertEqual(set(read_back.session), set(fitted.session))
            for name in fitted.session:
                np.testing.assert_array_equal(read_back.session[name], fitted.session[name])

        stored = run_arm("late_fusion_trajectory", self.problem, self.config,
                         alignment=self.alignment, options={"store_parameters": True})
        extras = through_json(stored.as_dict()["extras"])
        self.assertEqual(set(extras), {"cmr", "tee", "registration"})
        cmr_fit, tee_fit, _, _ = _late_fusion_fits(self.problem, self.config, self.alignment, None)
        assert_same(parameters_from_record(extras["cmr"]), cmr_fit.parameters)
        assert_same(parameters_from_record(extras["tee"]), tee_fit.parameters)
        rotation = np.asarray(extras["registration"]["rotation"])
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)

        joint = run_arm("rooted4d_full", self.problem, self.config, alignment=self.alignment,
                        options={"store_parameters": True})
        fitted = fit_joint_scene(self.problem.observations, self.model, self.config,
                                 start=_start(self.alignment, self.model, self.config))
        assert_same(parameters_from_record(through_json(joint.as_dict()["extras"])["parameters"]),
                    fitted.parameters)


if __name__ == "__main__":
    unittest.main()
