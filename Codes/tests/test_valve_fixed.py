"""The fixed-topology valve encoding: counts, commissures, and correspondence under rotation."""

import unittest

import numpy as np

from rooted4d.donors.valve import ANTERIOR_LABEL, POSTERIOR_LABEL
from rooted4d.donors.valve_fixed import fit_valve_fixed, fixed_valve_topology

SPACING = (0.6, 0.6, 0.6)
HALF_ANTERIOR_DEG = 55.0


def _synthetic_valve(anterior_axis_deg: float) -> np.ndarray:
    """A funnel from a 20 mm annulus to a 10 mm coaptation ring, 2 mm thick.

    The anterior leaflet spans +-55 degrees about ``anterior_axis_deg``; the
    posterior leaflet is the rest. The annulus is saddle-shaped, as a real one
    is, so a rotation about the axis is not a symmetry of the geometry.
    """

    shape = (80, 80, 36)
    grid = np.stack(np.meshgrid(*[np.arange(n) * s for n, s in zip(shape, SPACING)], indexing="ij"), axis=-1)
    centre = np.array([shape[0] * SPACING[0] / 2, shape[1] * SPACING[1] / 2, 3.0])
    offset = grid - centre
    theta = np.arctan2(offset[..., 1], offset[..., 0])
    relative = np.angle(np.exp(1j * (theta - np.radians(anterior_axis_deg))))
    saddle = 1.5 * np.cos(2.0 * relative)
    depth = offset[..., 2] - saddle
    radius = np.hypot(offset[..., 0], offset[..., 1])
    target = 20.0 - (10.0 / 12.0) * depth
    on_sheet = (depth >= 0.0) & (depth <= 12.0) & (np.abs(radius - target) <= 1.0)
    labels = np.zeros(shape, dtype=np.uint8)
    anterior = np.abs(relative) <= np.radians(HALF_ANTERIOR_DEG)
    labels[on_sheet & anterior] = ANTERIOR_LABEL
    labels[on_sheet & ~anterior] = POSTERIOR_LABEL
    return labels


class FixedValveTest(unittest.TestCase):
    def setUp(self):
        self.fit = fit_valve_fixed(_synthetic_valve(30.0), SPACING)

    def test_counts_and_topology_are_fixed(self):
        topology = fixed_valve_topology()
        self.assertIsNotNone(self.fit)
        self.assertEqual(self.fit.anterior.shape, (12, 6, 3))
        self.assertEqual(self.fit.posterior.shape, (20, 6, 3))
        self.assertEqual(len(self.fit.vertices()), topology["n_vertices"])
        for name, grid in (("anterior", topology["anterior_grid"]), ("posterior", topology["posterior_grid"])):
            faces = topology["faces"][name]
            self.assertEqual(len(faces), 2 * (grid.shape[0] - 1) * (grid.shape[1] - 1))
            self.assertLess(faces.max(), topology["n_vertices"])
        # Roots are annulus nodes: attachment is structural in the encoding too.
        vertices = self.fit.vertices()
        np.testing.assert_array_equal(vertices[topology["anterior_grid"][:, 0]], self.fit.anterior[:, 0])
        np.testing.assert_array_equal(vertices[topology["anterior_grid"][:, 3]], self.fit.anterior[:, 3])
        np.testing.assert_array_equal(vertices[topology["posterior_grid"][:, 5]], self.fit.posterior[:, 5])

    def test_commissures_are_found_where_the_leaflets_meet(self):
        low, high = np.degrees(self.fit.commissure_angles)
        self.assertAlmostEqual(low, -HALF_ANTERIOR_DEG, delta=8.0)
        self.assertAlmostEqual(high, HALF_ANTERIOR_DEG, delta=8.0)

    def test_vertices_mean_the_same_thing_after_a_rotation(self):
        rotated = fit_valve_fixed(_synthetic_valve(100.0), SPACING)
        centre_a = self.fit.vertices().mean(axis=0)
        centre_b = rotated.vertices().mean(axis=0)
        turn = np.radians(-70.0)
        rotation = np.array([[np.cos(turn), -np.sin(turn), 0.0], [np.sin(turn), np.cos(turn), 0.0], [0.0, 0.0, 1.0]])
        back = (rotated.vertices() - centre_b) @ rotation.T
        gap = np.linalg.norm(back - (self.fit.vertices() - centre_a), axis=1)
        # Voxelization differs between the two rotations, so agreement is to a
        # couple of voxels, not exact; an off-by-one node would be ~5 mm.
        self.assertLess(float(np.median(gap)), 1.0)
        self.assertLess(float(np.percentile(gap, 95)), 2.5)


if __name__ == "__main__":
    unittest.main()
