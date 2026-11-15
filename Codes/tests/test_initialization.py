"""Tests for the observation-only starting alignment.

The start exists so that two shape models fitted to the same observations begin
from the same place, and so that no arm is handed the truth alignment. These
tests pin both properties down on the procedural fixture, independent of any
benchmark data.
"""

import unittest
from dataclasses import replace

import numpy as np

from rooted4d.geometry import RigidTransform
from rooted4d.inference import SceneModel, SolverConfig
from rooted4d.inference.initialization import (
    MINIMUM_SHARED_VERTICES,
    input_derived_alignment,
    input_derived_start,
)
from rooted4d.inference.problem import make_joint_problem, pose_errors


def _config() -> SolverConfig:
    return SolverConfig(fourier_order=2, session_order=1, max_iterations=30)


class InputDerivedStartTest(unittest.TestCase):
    def setUp(self):
        self.config = _config()
        self.model = SceneModel.procedural(n_modes=10, n_motion=3, n_session=2, n_phase_nodes=6)
        self.problem = make_joint_problem(self.model, self.config, seed=3)

    def test_start_is_identical_for_two_different_models(self):
        """The point of the start: it cannot depend on the shape model."""

        other = SceneModel.procedural(n_modes=14, n_motion=3, n_session=2, n_phase_nodes=6, seed=5)
        first = input_derived_start(self.problem.observations, self.model, self.config)
        second = input_derived_start(self.problem.observations, other, self.config)
        np.testing.assert_array_equal(first.pose_axis_angle, second.pose_axis_angle)
        np.testing.assert_array_equal(first.pose_translation, second.pose_translation)
        self.assertEqual(first.phase_offset, second.phase_offset)
        self.assertEqual(first.identity.size, self.model.n_modes)
        self.assertEqual(second.identity.size, other.n_modes)
        self.assertFalse(np.any(first.identity))

    def test_start_is_much_closer_to_the_truth_than_the_identity(self):
        start = input_derived_start(self.problem.observations, self.model, self.config)
        errors = pose_errors(start, self.problem.truth)
        identity = pose_errors(replace(start, pose_axis_angle=np.zeros(3),
                                       pose_translation=np.zeros(3)), self.problem.truth)
        self.assertLess(errors["rotation_deg"], 0.25 * identity["rotation_deg"])
        self.assertLess(errors["translation_mm"], 0.25 * identity["translation_mm"])

    def test_pose_follows_a_rigid_change_of_the_tee_frame(self):
        """Move the TEE acquisition frame; the estimated pose must move with it."""

        extra = RigidTransform.from_axis_angle(np.array([0.1, -0.2, 0.15]),
                                               np.array([4.0, -7.0, 2.5]))
        observations = self.problem.observations
        moved = replace(observations, tee_points=extra.apply(
            observations.tee_points.reshape(-1, 3)).reshape(observations.tee_points.shape))
        before = input_derived_alignment(observations)
        after = input_derived_alignment(moved)
        pose = RigidTransform.from_axis_angle(before.pose_axis_angle, before.pose_translation)
        composed = RigidTransform.from_axis_angle(after.pose_axis_angle, after.pose_translation)
        points = np.random.default_rng(0).normal(scale=30.0, size=(10, 3))
        np.testing.assert_allclose(composed.apply(points), extra.apply(pose.apply(points)),
                                   atol=1e-8)

    def test_refuses_too_few_shared_vertices(self):
        observations = self.problem.observations
        shared = np.intersect1d(observations.cmr_vertices, observations.tee_vertices)
        keep = np.isin(observations.tee_vertices, shared[: MINIMUM_SHARED_VERTICES - 1]) | ~np.isin(
            observations.tee_vertices, shared)
        thinned = replace(observations, tee_vertices=observations.tee_vertices[keep],
                          tee_points=observations.tee_points[:, keep, :])
        with self.assertRaises(ValueError):
            input_derived_alignment(thinned)


if __name__ == "__main__":
    unittest.main()


class SharedStartThroughArmsTest(unittest.TestCase):
    """Every ladder arm accepts the shared alignment; None changes nothing."""

    def setUp(self):
        from rooted4d.inference.baselines import BASELINE_IDS, run_arm

        self.run_arm, self.arms = run_arm, BASELINE_IDS
        self.config = replace(_config(), max_iterations=6)
        self.model = SceneModel.procedural(n_modes=8, n_motion=3, n_session=2, n_phase_nodes=6)
        self.problem = make_joint_problem(self.model, self.config, seed=11)

    def test_none_reproduces_the_historical_zero_start(self):
        for arm in ("rooted4d_full", "tee_only"):
            a = self.run_arm(arm, self.problem, self.config)
            b = self.run_arm(arm, self.problem, self.config, alignment=None)
            self.assertEqual(a.as_dict(), b.as_dict())

    def test_every_arm_runs_from_the_shared_alignment(self):
        alignment = input_derived_alignment(self.problem.observations)
        for arm in self.arms:
            result = self.run_arm(arm, self.problem, self.config, alignment=alignment)
            self.assertTrue(result.succeeded, arm)
