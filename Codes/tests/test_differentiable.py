import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is an optional dependency
    torch = None

from rooted4d.geometry import make_toy_complex
from rooted4d.motion import MonotoneCircularPhaseWarp
from rooted4d.observation.distance import point_to_mesh_distance as numpy_distance
from rooted4d.observation.distance import surface_likelihood as numpy_likelihood


@unittest.skipIf(torch is None, "torch is not installed")
class DifferentiableGeometryTests(unittest.TestCase):
    """The port must reproduce the NumPy definitions, which remain definitive."""

    def setUp(self) -> None:
        complex_ = make_toy_complex(n_annulus=24)
        self.vertices = complex_.global_vertices
        self.faces = np.asarray(complex_.faces["cmr_surface"], dtype=np.int64)
        rng = np.random.default_rng(0)
        self.points = self.vertices.mean(axis=0) + rng.normal(0.0, 25.0, (120, 3))

    def test_never_sees_a_cuda_device(self) -> None:
        # The machine owner's card also drives the display, and a solver that
        # fills its memory freezes the desktop. This package pins itself to CPU
        # at import; silently running on the card is the failure being excluded.
        from rooted4d.differentiable import assert_cpu_only

        assert_cpu_only()

    def test_distance_matches_numpy(self) -> None:
        from rooted4d.differentiable import point_to_mesh_distance

        expected = numpy_distance(self.points, self.vertices, self.faces)
        actual = point_to_mesh_distance(
            torch.as_tensor(self.points),
            torch.as_tensor(self.vertices),
            torch.as_tensor(self.faces),
        ).numpy()
        self.assertLess(float(np.abs(expected - actual).max()), 1e-10)

    def test_likelihood_matches_numpy(self) -> None:
        from rooted4d.differentiable import surface_likelihood

        expected = numpy_likelihood(
            self.points, self.vertices, self.faces, sigma_mm=1.5
        )
        actual = surface_likelihood(
            torch.as_tensor(self.points),
            torch.as_tensor(self.vertices),
            torch.as_tensor(self.faces),
            sigma_mm=1.5,
        ).numpy()
        self.assertLess(float(np.abs(expected - actual).max()), 1e-10)

    def test_gradient_matches_finite_differences(self) -> None:
        from rooted4d.differentiable import surface_likelihood

        vertices = torch.tensor(self.vertices, dtype=torch.float64, requires_grad=True)
        surface_likelihood(
            torch.as_tensor(self.points), vertices,
            torch.as_tensor(self.faces), sigma_mm=1.5,
        ).sum().backward()
        gradient = vertices.grad.numpy()
        step = 1e-6
        checked = 0
        for index in (0, 11, 37, 60):
            for axis in range(3):
                plus = self.vertices.copy()
                plus[index, axis] += step
                minus = self.vertices.copy()
                minus[index, axis] -= step
                numeric = (
                    numpy_likelihood(self.points, plus, self.faces, sigma_mm=1.5).sum()
                    - numpy_likelihood(self.points, minus, self.faces, sigma_mm=1.5).sum()
                ) / (2.0 * step)
                if abs(numeric) < 1e-6 and abs(gradient[index, axis]) < 1e-6:
                    continue
                self.assertAlmostEqual(numeric / gradient[index, axis], 1.0, places=4)
                checked += 1
        self.assertGreater(checked, 0, "no live gradient components were compared")


@unittest.skipIf(torch is None, "torch is not installed")
class DifferentiableWarpTests(unittest.TestCase):
    def test_warp_matches_numpy(self) -> None:
        # The offset is applied after the piecewise-linear map, not before.
        # Both orderings are monotone circular warps with the same parameters,
        # so getting it wrong raises nothing and simply returns a different
        # function; it disagreed with NumPy by 1.4 in the residual before it was
        # caught, and this test is what would have caught it.
        from rooted4d.differentiable.scene import _warp

        rng = np.random.default_rng(0)
        logits = rng.normal(0.0, 0.6, 8)
        offset = 0.37
        times = np.linspace(0.0, 1.0, 40, endpoint=False)
        expected = MonotoneCircularPhaseWarp(logits, offset=offset).evaluate(times)
        actual = _warp(
            torch.as_tensor(times), torch.as_tensor(logits), torch.tensor(offset)
        ).numpy()
        self.assertLess(float(np.abs(expected - actual).max()), 1e-12)

    def test_rotation_jacobian_matches_finite_differences_including_at_identity(self):
        """The Jacobian must be right at zero, not merely finite there.

        The previous version of this test differentiated the sum of the rotation
        matrix and asserted the gradient was finite. Both held while the true
        Jacobian at the identity was identically zero: normalizing the axis and
        clamping the norm leaves every term of the product rule vanishing there.
        Any optimizer starting from zero rotation therefore left rotation at the
        identity for an entire run while reporting finite gradients, which is
        what the cold-start image arm did.

        A later attempt substituted a dummy angle after taking the square root,
        which made the whole Jacobian NaN instead, because torch.where
        backpropagates through the inputs of both branches and d(sqrt)/dx is
        infinite at zero. Hence the comparison against finite differences at
        several magnitudes rather than a single smoke check.
        """

        from rooted4d.differentiable.scene import _rotation

        def finite_difference(vector, step=1e-6):
            columns = []
            for index in range(3):
                plus = torch.tensor(vector, dtype=torch.float64)
                minus = torch.tensor(vector, dtype=torch.float64)
                plus[index] += step
                minus[index] -= step
                columns.append((_rotation(plus) - _rotation(minus)) / (2.0 * step))
            return torch.stack(columns, dim=-1)

        for vector in ([0.0, 0.0, 0.0], [1e-5, 0.0, 0.0],
                       [0.3, -0.2, 0.7], [2.0, 1.0, -0.5]):
            axis_angle = torch.tensor(vector, dtype=torch.float64)
            analytic = torch.autograd.functional.jacobian(_rotation, axis_angle)
            self.assertTrue(bool(torch.isfinite(analytic).all()),
                            "non-finite Jacobian at %s" % vector)
            self.assertLess(
                float((analytic - finite_difference(vector)).abs().max()), 1e-8,
                "Jacobian disagrees with finite differences at %s" % vector)

        # At the identity specifically, the derivative is not merely finite: the
        # generators have unit magnitude, so a zero maximum is the defect.
        at_zero = torch.autograd.functional.jacobian(
            _rotation, torch.zeros(3, dtype=torch.float64))
        self.assertAlmostEqual(float(at_zero.abs().max()), 1.0, places=9)

    def test_rotation_is_recoverable_from_the_identity(self) -> None:
        """A gradient method starting at zero rotation must be able to move."""

        from rooted4d.differentiable.scene import _rotation

        target = _rotation(torch.tensor([0.4, -0.3, 0.9], dtype=torch.float64)).detach()
        axis_angle = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([axis_angle], lr=0.1)
        for _ in range(600):
            optimizer.zero_grad()
            loss = ((_rotation(axis_angle) - target) ** 2).sum()
            loss.backward()
            optimizer.step()
        self.assertLess(float(loss), 1e-12)
        self.assertLess(
            float((axis_angle.detach()
                   - torch.tensor([0.4, -0.3, 0.9], dtype=torch.float64)).abs().max()),
            1e-4)

    def test_rotation_stays_orthogonal(self) -> None:
        from rooted4d.differentiable.scene import _rotation

        for vector in ([0.0, 0.0, 0.0], [1e-9, 0.0, 0.0], [np.pi, 0.0, 0.0],
                       [0.3, -0.2, 0.7]):
            matrix = _rotation(torch.tensor(vector, dtype=torch.float64))
            product = matrix @ matrix.T
            identity = torch.eye(3, dtype=torch.float64)
            self.assertLess(float((product - identity).abs().max()), 1e-12)


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(torch is None, "torch is not installed")
class CovarianceAwarePriorPortTests(unittest.TestCase):
    """The torch objective must carry the covariance-aware identity prior exactly."""

    def test_identity_prior_rows_match_the_numpy_solver(self) -> None:
        from dataclasses import replace

        from rooted4d.differentiable.scene import TorchScene, prior_residuals, prior_weights
        from rooted4d.inference import SceneModel, SolverConfig
        from rooted4d.inference.joint import (
            _OBSERVATION_ROWS, _pack, _residuals, initial_parameters,
        )
        from rooted4d.inference.problem import make_joint_problem

        model = SceneModel.procedural(n_modes=10, n_motion=3, n_session=2, n_phase_nodes=6)
        model = replace(model, shape_scales=np.geomspace(5.0, 0.2, model.n_modes))
        config = SolverConfig(fourier_order=2, session_order=1, identity_prior_weight=2.0,
                              covariance_aware_identity_prior=True)
        problem = make_joint_problem(model, config, seed=3)
        parameters = replace(initial_parameters(model, config),
                             identity=np.linspace(-1.0, 1.0, model.n_modes))
        vector = _pack(parameters, model, config)
        rows = _residuals(vector, parameters, problem.observations, model, config)
        start = _OBSERVATION_ROWS["n"]
        expected = rows[start:start + model.n_modes]

        scene = TorchScene.from_model(model, config)
        ported = prior_residuals(torch.as_tensor(vector), scene, prior_weights(config, model))
        np.testing.assert_allclose(ported[: model.n_modes].numpy(), expected, rtol=1e-12)
        # The multiplier is not a relabelled isotropic prior on this spectrum.
        self.assertGreater(np.abs(expected - np.sqrt(2.0) * parameters.identity).max(), 1e-3)
        # Without the model the prior stays isotropic, as every earlier caller had it.
        isotropic = prior_residuals(torch.as_tensor(vector), scene, prior_weights(config))
        np.testing.assert_allclose(isotropic[: model.n_modes].numpy(),
                                   np.sqrt(2.0) * parameters.identity, rtol=1e-12)
