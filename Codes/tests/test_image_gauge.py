"""E120: every geometry the fit compares with evidence is the gauged one.

The session prior charges the cross-session-centred session blocks, and
``predict`` evaluates the point evidence on them. Three other residuals used
the raw blocks: the NumPy image residuals, the soft-attachment penalty, and
every residual of the torch port. The cross-session mean of the blocks then
moved those geometries at no prior cost, and the fit's final gauge removed that
movement from the geometry it reported. These tests pin the three to the gauged
geometry: a common offset added to every session block changes none of them.
"""

import unittest
from dataclasses import replace

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is an optional dependency
    torch = None

from rooted4d.geometry import make_toy_complex
from rooted4d.inference import SceneModel, SolverConfig
from rooted4d.inference.image_evidence import ImageEvidence, image_residuals
from rooted4d.inference.joint import (
    _apply_gauges,
    _pack,
    _residuals,
    _unpack,
    initial_parameters,
    latent_trajectory,
)
from rooted4d.inference.problem import make_joint_problem


def _model() -> SceneModel:
    return SceneModel.procedural(n_modes=10, n_motion=3, n_session=2, n_phase_nodes=6)


def _config(**changes) -> SolverConfig:
    return SolverConfig(fourier_order=2, session_order=1, **changes)


def _parameters(model, config, seed=0):
    rng = np.random.default_rng(seed)
    start = initial_parameters(model, config)
    vector = _pack(start, model, config)
    return _unpack(vector + rng.normal(0.0, 0.3, vector.size), start, model, config)


def _shifted(parameters, seed=1):
    """The same parameters with one offset added to every session block."""

    first = next(iter(parameters.session.values()))
    offset = np.random.default_rng(seed).normal(0.0, 2.0, first.shape)
    return replace(parameters,
                   session={name: block + offset for name, block in parameters.session.items()})


def _evidence(model, seed=2) -> ImageEvidence:
    rng = np.random.default_rng(seed)
    complex_ = make_toy_complex()
    assert complex_.global_vertices.shape == model.template_physical.shape
    centre = model.template_physical.mean(axis=0)
    spread = model.template_physical.std(axis=0)
    cmr = centre + rng.normal(0.0, 1.0, (40, 3)) * spread
    tee = centre + rng.normal(0.0, 1.0, (40, 3)) * spread
    return ImageEvidence(
        cmr_times=np.array([0.1, 0.6]), cmr_points=cmr, cmr_values=np.zeros((2, 40)),
        cmr_faces=np.asarray(complex_.faces["cmr_surface"], dtype=np.int64),
        cmr_through_plane=np.array([-1.0, 0.0, 1.0]),
        tee_times=np.array([0.2, 0.7]), tee_points=tee, tee_values=np.zeros((2, 40)),
        tee_faces=np.asarray(complex_.faces["tee_surface"], dtype=np.int64),
        sigma_mm=2.0, cmr_normal=np.array([0.0, 0.0, 1.0]),
    )


class NumpyImageGaugeTest(unittest.TestCase):
    def setUp(self):
        self.model = _model()
        self.config = _config(cmr_image_weight=1.0, tee_image_weight=1.0)
        self.parameters = _parameters(self.model, self.config)
        self.evidence = _evidence(self.model)

    def test_session_mean_does_not_move_the_image_geometry(self):
        base = image_residuals(self.parameters, self.evidence, self.model, self.config)
        moved = image_residuals(_shifted(self.parameters), self.evidence, self.model,
                                self.config)
        self.assertGreater(np.abs(base).max(), 1e-3)
        np.testing.assert_allclose(moved, base, rtol=0.0, atol=1e-12)

    def test_image_geometry_is_the_reported_one(self):
        gauged = _apply_gauges(self.parameters, self.model, self.config)
        np.testing.assert_allclose(
            image_residuals(self.parameters, self.evidence, self.model, self.config),
            image_residuals(gauged, self.evidence, self.model, self.config),
            rtol=0.0, atol=1e-12)

    def test_without_gauges_the_raw_blocks_are_used(self):
        config = replace(self.config, enforce_gauges=False)
        base = image_residuals(self.parameters, self.evidence, self.model, config)
        moved = image_residuals(_shifted(self.parameters), self.evidence, self.model, config)
        self.assertGreater(np.abs(moved - base).max(), 1e-6)


class SoftAttachmentGaugeTest(unittest.TestCase):
    def test_session_mean_does_not_move_the_attachment_penalty(self):
        model = _model().detached()
        config = _config(soft_attachment_weight=1.0)
        problem = make_joint_problem(model, config, seed=3)
        parameters = _parameters(model, config)
        rows = {}
        for name, value in (("base", parameters), ("moved", _shifted(parameters))):
            vector = _pack(value, model, config)
            rows[name] = _residuals(vector, value, problem.observations, model, config)
        # Points and priors were already gauged; the penalty now is too, so
        # the whole residual vector is unchanged.
        np.testing.assert_allclose(rows["moved"], rows["base"], rtol=0.0, atol=1e-9)


@unittest.skipIf(torch is None, "torch is not installed")
class TorchGaugeTest(unittest.TestCase):
    def setUp(self):
        from rooted4d.differentiable.scene import TorchScene

        self.model = _model()
        self.config = _config()
        self.parameters = _parameters(self.model, self.config)
        self.scene = TorchScene.from_model(self.model, self.config)

    def _coordinates(self, parameters, phases, session):
        from rooted4d.differentiable.scene import _coordinates

        vector = torch.as_tensor(_pack(parameters, self.model, self.config))
        return _coordinates(self.scene, self.scene.unpack(vector),
                            torch.as_tensor(phases), session).numpy()

    def test_geometry_matches_the_gauged_numpy_geometry(self):
        phases = np.linspace(0.0, 1.0, 5, endpoint=False)
        gauged = _apply_gauges(self.parameters, self.model, self.config)
        for session in self.model.sessions:
            expected = self.model.physical_coordinates(
                latent_trajectory(gauged, phases, session, self.model, self.config))
            np.testing.assert_allclose(self._coordinates(self.parameters, phases, session),
                                       expected, rtol=0.0, atol=1e-10)

    def test_session_mean_does_not_move_the_geometry(self):
        phases = np.linspace(0.0, 1.0, 5, endpoint=False)
        for session in self.model.sessions:
            np.testing.assert_allclose(
                self._coordinates(_shifted(self.parameters), phases, session),
                self._coordinates(self.parameters, phases, session), rtol=0.0, atol=1e-10)

    def test_point_residuals_match_the_numpy_solver(self):
        from rooted4d.differentiable.scene import point_residuals
        from rooted4d.inference.joint import predict

        problem = make_joint_problem(self.model, self.config, seed=4)
        observations = problem.observations
        cmr, tee = predict(self.parameters, observations, self.model, self.config)
        expected = np.concatenate([(cmr - observations.cmr_points).reshape(-1),
                                   (tee - observations.tee_points).reshape(-1)])
        vector = torch.as_tensor(_pack(self.parameters, self.model, self.config))
        rows = self.model.observation_rows
        evidence = {
            "cmr_points": torch.as_tensor(observations.cmr_points),
            "cmr_times": torch.as_tensor(observations.cmr_times),
            "cmr_vertices": torch.as_tensor(rows(observations.cmr_vertices)),
            "tee_points": torch.as_tensor(observations.tee_points),
            "tee_times": torch.as_tensor(observations.tee_times),
            "tee_vertices": torch.as_tensor(rows(observations.tee_vertices)),
        }
        ported = point_residuals(vector, self.scene, evidence).numpy()
        np.testing.assert_allclose(ported, expected, rtol=0.0, atol=1e-9)

    def test_prior_follows_the_gauge_switch(self):
        from rooted4d.differentiable.scene import TorchScene, prior_residuals, prior_weights

        for enforce in (True, False):
            config = replace(self.config, enforce_gauges=enforce)
            scene = TorchScene.from_model(self.model, config)
            self.assertEqual(scene.centre_sessions, enforce)
            vector = _pack(self.parameters, self.model, config)
            ported = prior_residuals(torch.as_tensor(vector), scene,
                                     prior_weights(config, self.model)).numpy()
            gauged = _apply_gauges(self.parameters, self.model, config)
            blocks = np.concatenate([gauged.session[n].reshape(-1) for n in self.model.sessions])
            start = self.model.n_modes + self.parameters.motion.size
            np.testing.assert_allclose(
                ported[start:start + blocks.size],
                np.sqrt(config.session_prior_weight) * blocks, rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
