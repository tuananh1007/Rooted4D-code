"""The Stage C shape spaces on a real component topology (the ACDC chamber surface)."""

import unittest
from dataclasses import replace

import numpy as np

from rooted4d.active.acdc import lv_topology_faces
from rooted4d.experiments.component_shape import (
    chamber_structure,
    covariance_multiplier,
    fit_global_space,
    fit_structured_space,
    map_estimate,
)

N_RINGS, N_ANGLES = 8, 32


def _chamber(rng: np.random.Generator) -> np.ndarray:
    """An LV-like surface in the fit topology: rings then base and apex caps."""
    theta = np.linspace(0.0, 2 * np.pi, N_ANGLES, endpoint=False)
    heights = np.linspace(0.0, 70.0, N_RINGS)
    radius = 25.0 * np.sqrt(1.0 - (heights / 85.0) ** 2)
    radius = radius * (1.0 + 0.05 * rng.normal())
    wobble = rng.normal(0.0, 0.8, size=(N_RINGS, 3))
    rings = np.stack([
        np.stack([(r * (1 + 0.04 * w[0] * np.cos(2 * theta + w[1]))) * np.cos(theta),
                  (r * (1 + 0.04 * w[0] * np.cos(2 * theta + w[1]))) * np.sin(theta),
                  np.full_like(theta, z + w[2])], axis=-1)
        for r, z, w in zip(radius, heights, wobble)])
    vertices = rings.reshape(-1, 3)
    base, apex = rings[0].mean(axis=0), rings[-1].mean(axis=0) + np.array([0.0, 0.0, 4.0])
    return np.vstack([vertices, base, apex]) + rng.normal(0.0, 5.0, size=3)


class ComponentShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.shapes = np.stack([_chamber(rng) for _ in range(40)])
        structure = chamber_structure(N_RINGS, N_ANGLES)
        self.interface = structure["interface"]  # ring 0, the basal ring
        np.testing.assert_array_equal(structure["faces_by_component"]["chamber"],
                                      lv_topology_faces(N_RINGS, N_ANGLES))
        self.structured = fit_structured_space(self.shapes, **structure, ring_dim=6, interior_dim=6)
        self.global_ = fit_global_space(self.shapes, rank=12)

    def test_bases_are_orthonormal_with_descending_scales(self) -> None:
        for space in (self.structured, self.global_):
            gram = space.basis.T @ space.basis
            np.testing.assert_allclose(gram, np.eye(space.rank), atol=1e-10)
            self.assertTrue(np.all(np.diff(space.scales) <= 1e-12))
        self.assertLessEqual(self.structured.rank, 12)

    def test_interior_modes_cannot_move_the_ring(self) -> None:
        # The ring map of the structured span has rank equal to the ring block.
        ring_rows = np.concatenate([3 * self.interface + k for k in range(3)])
        ring_map = self.structured.basis[np.sort(ring_rows)]
        singular = np.linalg.svd(ring_map, compute_uv=False)
        rank = int(np.count_nonzero(singular > 1e-8 * singular[0]))
        self.assertEqual(rank, 6)

    def test_map_estimate_reproduces_the_span_and_tends_to_the_mean(self) -> None:
        space = self.structured
        rng = np.random.default_rng(3)
        inside = space.mean + (space.basis @ rng.normal(0, 2.0, space.rank)).reshape(space.mean.shape)
        shifted = inside + np.array([10.0, -4.0, 2.0])
        np.testing.assert_allclose(map_estimate(shifted, space, 0.0), shifted, atol=1e-8)
        # Large enough for the weakest-shrunk mode too: the leading mode's
        # multiplier is about 0.01 on this spectrum, so beta must beat 1e4 * 1e10.
        far = map_estimate(shifted, space, 1e15)
        np.testing.assert_allclose(far, space.mean + shifted.mean(axis=0), atol=1e-6)

    def test_complement_keeps_the_input_at_zero_and_the_mean_far_out(self) -> None:
        space = self.structured
        self.assertGreater(space.residual_scale, 0.0)
        shape = self.shapes[0] + np.random.default_rng(5).normal(0.0, 0.3, self.shapes[0].shape)
        np.testing.assert_allclose(map_estimate(shape, space, 0.0, complement=True), shape, atol=1e-8)
        far = map_estimate(shape, space, 1e15, complement=True)
        np.testing.assert_allclose(far, space.mean + shape.mean(axis=0), atol=1e-6)
        # In between, the complement is shrunk, not dropped: closer to the input than the projection.
        middle = map_estimate(shape, space, 1e-3, complement=True)
        projected = map_estimate(shape, space, 1e-3)
        self.assertLess(np.linalg.norm(middle - shape), np.linalg.norm(projected - shape))

    def test_multiplier_is_the_joint_solvers(self) -> None:
        from rooted4d.inference import SceneModel, SolverConfig
        from rooted4d.inference.joint import identity_prior_multiplier

        scales = np.geomspace(4.0, 0.001, 12)
        model = replace(SceneModel.procedural(n_modes=12), shape_scales=scales)
        config = SolverConfig(covariance_aware_identity_prior=True, identity_prior_scale_floor=0.005)
        np.testing.assert_allclose(covariance_multiplier(scales, 0.005),
                                   identity_prior_multiplier(model, config, 12), rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
