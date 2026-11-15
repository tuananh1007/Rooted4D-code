"""Curve evidence and commissure anchors (E101)."""

import unittest
from dataclasses import replace

import numpy as np

from rooted4d.inference import SceneModel, SolverConfig, fit_joint_scene, predict
from rooted4d.inference.joint import (
    _closest_on_loop,
    _residuals,
    _pack,
    turned_about_annulus,
)
from rooted4d.inference.problem import make_joint_problem, pose_errors


def _model() -> SceneModel:
    return SceneModel.procedural(n_modes=10, n_motion=3, n_session=2, n_phase_nodes=6)


def _config(**overrides) -> SolverConfig:
    return SolverConfig(fourier_order=2, session_order=1, max_iterations=30, **overrides)


def _clean(problem, model, config):
    """The problem with noise-free observations at the truth."""

    cmr, tee = predict(problem.truth, problem.observations, model, config)
    return replace(problem.observations, cmr_points=cmr, tee_points=tee)


def _renumbered(observations, model, shift):
    """TEE ring observations named ``shift`` vertices along: right points, wrong names."""

    ring = np.asarray(model.groups["annulus"])
    n = len(ring)
    points = observations.tee_points.copy()
    vertices = np.asarray(observations.tee_vertices)
    for column, vertex in enumerate(vertices):
        if vertex in ring:
            source = ring[(np.flatnonzero(ring == vertex)[0] + shift) % n]
            if source in vertices:
                points[:, column] = observations.tee_points[:, np.flatnonzero(vertices == source)[0]]
    return replace(observations, tee_points=points)


def _with_anchors(observations, truth_observations, model, vertices):
    columns = [int(np.flatnonzero(np.asarray(truth_observations.tee_vertices) == v)[0]) for v in vertices]
    return replace(observations, tee_anchor_vertices=np.asarray(vertices),
                   tee_anchor_points=truth_observations.tee_points[:, columns, :])


class ClosestOnLoopTest(unittest.TestCase):
    def test_square(self):
        loop = np.array([[[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]])
        points = np.array([[[0.5, -2.0, 0.0], [2.0, 0.5, 1.0], [-1.0, -1.0, 0.0]]])
        expected = np.array([[[0.5, 0.0, 0.0], [1.0, 0.5, 0.0], [0.0, 0.0, 0.0]]])
        np.testing.assert_allclose(_closest_on_loop(points, loop), expected, atol=1e-12)

    def test_closing_edge_counts(self):
        loop = np.array([[[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]])
        points = np.array([[[-0.5, 0.5, 0.0]]])
        np.testing.assert_allclose(_closest_on_loop(points, loop), [[[0.0, 0.5, 0.0]]], atol=1e-12)


class CurveEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.model = _model()
        self.config = _config()
        self.problem = make_joint_problem(self.model, self.config, seed=3)
        self.clean = _clean(self.problem, self.model, self.config)
        self.ring = np.asarray(self.model.groups["annulus"])

    def _cost(self, observations, config):
        vector = _pack(self.problem.truth, self.model, config)
        residual = _residuals(vector, self.problem.truth, observations, self.model, config)
        return residual

    def test_default_config_ignores_anchors(self):
        anchored = _with_anchors(self.clean, self.clean, self.model, [int(v) for v in self.ring[:2]
                                                                       if v in self.clean.tee_vertices])
        np.testing.assert_array_equal(self._cost(self.clean, self.config),
                                      self._cost(anchored, self.config))

    def test_curve_evidence_keeps_the_row_count(self):
        curve = replace(self.config, curve_evidence="annulus")
        self.assertEqual(self._cost(self.clean, self.config).size, self._cost(self.clean, curve).size)

    def test_renumbered_ring_is_free_under_curve_evidence(self):
        shifted = _renumbered(self.clean, self.model, 5)
        point = self._cost(shifted, self.config)
        curve = self._cost(shifted, replace(self.config, curve_evidence="annulus"))
        evidence = slice(0, self.clean.cmr_points.size + self.clean.tee_points.size)
        self.assertGreater(np.abs(point[evidence]).max(), 1.0)
        self.assertLess(np.abs(curve[evidence]).max(), 1e-9)

    def test_anchors_are_point_evidence(self):
        observed = [int(v) for v in self.ring if v in self.clean.tee_vertices]
        anchored = _with_anchors(self.clean, self.clean, self.model, [observed[0], observed[len(observed) // 2]])
        moved = replace(anchored, tee_anchor_points=anchored.tee_anchor_points + 1.0)
        curve = replace(self.config, curve_evidence="annulus")
        base = self._cost(anchored, curve)
        self.assertEqual(base.size, self._cost(self.clean, curve).size + anchored.tee_anchor_points.size)
        self.assertGreater(np.abs(self._cost(moved, curve) - base).max(), 0.5)

    def test_turning_is_a_rotation_about_the_axis(self):
        truth = self.problem.truth
        back = turned_about_annulus(turned_about_annulus(truth, self.model, 0.7), self.model, -0.7)
        np.testing.assert_allclose(back.transform().rotation, truth.transform().rotation, atol=1e-10)
        np.testing.assert_allclose(back.pose_translation, truth.pose_translation, atol=1e-9)
        half = turned_about_annulus(truth, self.model, np.pi)
        ring = self.model.template_physical[self.ring]
        np.testing.assert_allclose(half.transform().apply(ring.mean(axis=0)),
                                   truth.transform().apply(ring.mean(axis=0)), atol=1e-9)
        self.assertAlmostEqual(pose_errors(half, truth)["rotation_deg"], 180.0, places=6)


class CurveStartTest(unittest.TestCase):
    def test_anchors_turn_a_half_flipped_start_back(self):
        """A TEE ring named half a turn along: point evidence follows the names, anchors do not."""

        model = _model()
        config = _config()
        problem = make_joint_problem(model, config, seed=5)
        clean = _clean(problem, model, config)
        ring = np.asarray(model.groups["annulus"])
        flipped = _renumbered(clean, model, len(ring) // 2)
        observed = [int(v) for v in ring if v in clean.tee_vertices]
        anchored = _with_anchors(flipped, clean, model, [observed[0], observed[len(observed) // 2]])
        start = turned_about_annulus(replace(problem.truth, identity=np.zeros_like(problem.truth.identity),
                                             motion=np.zeros_like(problem.truth.motion)), model, np.pi)
        point = fit_joint_scene(flipped, model, config, start=start)
        self.assertGreater(pose_errors(point.parameters, problem.truth)["rotation_deg"], 90.0)
        curve = replace(config, curve_evidence="annulus")
        fit = fit_joint_scene(anchored, model, curve, start=start)
        self.assertLess(pose_errors(fit.parameters, problem.truth)["rotation_deg"], 10.0)


if __name__ == "__main__":
    unittest.main()
