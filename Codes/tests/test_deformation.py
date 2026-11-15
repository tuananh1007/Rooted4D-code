import unittest

import numpy as np

from rooted4d.geometry.deformation import (
    barrier_value_and_gradient,
    boundary_faces,
    collision_value_and_gradient,
    elastic_value_and_gradient,
    fit_deformation,
    lattice_mesh,
    tetrahedron_edges,
)
from rooted4d.geometry.tetrahedra import (
    signed_tetrahedron_volumes,
    tetrahedron_jacobian_barrier,
)


class LatticeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vertices, self.cells = lattice_mesh((4, 4, 4), spacing=5.0)

    def test_all_cells_are_positively_oriented(self) -> None:
        volumes = signed_tetrahedron_volumes(self.vertices, self.cells)
        self.assertTrue(np.all(volumes > 0.0))

    def test_boundary_is_closed(self) -> None:
        # Every directed boundary edge must have exactly one opposite twin. A
        # lattice whose cube split does not alternate leaves cracks between
        # neighbouring cubes, and the resulting surface is not closed, which
        # would silently invalidate any self-intersection count taken on it.
        faces = boundary_faces(self.cells)
        directed: set[tuple[int, int]] = set()
        for face in faces:
            for start, end in ((0, 1), (1, 2), (2, 0)):
                edge = (int(face[start]), int(face[end]))
                self.assertNotIn(edge, directed)
                directed.add(edge)
        for start, end in directed:
            self.assertIn((end, start), directed)

    def test_lattice_volume_matches_the_box(self) -> None:
        volumes = signed_tetrahedron_volumes(self.vertices, self.cells)
        self.assertAlmostEqual(float(volumes.sum()), 15.0**3, places=6)

    def test_shape_must_be_at_least_two(self) -> None:
        with self.assertRaises(ValueError):
            lattice_mesh((1, 4, 4))


class BarrierGradientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vertices, self.cells = lattice_mesh((3, 3, 3), spacing=5.0)

    def _active(self) -> np.ndarray:
        # Squash until the determinant ratio sits inside the active band, or the
        # barrier is zero and any gradient agreement is vacuous.
        deformed = self.vertices.copy()
        deformed[:, 2] *= 0.18
        return deformed

    def test_value_matches_the_reference_implementation(self) -> None:
        deformed = self._active()
        value, _ = barrier_value_and_gradient(self.vertices, deformed, self.cells)
        self.assertGreater(value, 0.0)
        self.assertAlmostEqual(
            value,
            tetrahedron_jacobian_barrier(self.vertices, deformed, self.cells),
            places=10,
        )

    def test_gradient_matches_finite_differences(self) -> None:
        deformed = self._active()
        _, gradient = barrier_value_and_gradient(self.vertices, deformed, self.cells)
        step = 1e-7
        checked = 0
        for index in (0, 5, 11):
            for axis in range(3):
                plus = deformed.copy()
                plus[index, axis] += step
                minus = deformed.copy()
                minus[index, axis] -= step
                numeric = (
                    barrier_value_and_gradient(self.vertices, plus, self.cells)[0]
                    - barrier_value_and_gradient(self.vertices, minus, self.cells)[0]
                ) / (2.0 * step)
                if abs(numeric) < 1e-6:
                    continue
                self.assertAlmostEqual(
                    numeric / gradient[index, axis], 1.0, places=4
                )
                checked += 1
        self.assertGreater(checked, 0, "no live gradient components were compared")

    def test_infeasible_configuration_is_infinite(self) -> None:
        deformed = self.vertices.copy()
        deformed[:, 2] *= -1.0
        value, gradient = barrier_value_and_gradient(
            self.vertices, deformed, self.cells
        )
        self.assertFalse(np.isfinite(value))
        self.assertTrue(np.all(gradient == 0.0))


class ElasticGradientTests(unittest.TestCase):
    def test_gradient_matches_finite_differences(self) -> None:
        vertices, cells = lattice_mesh((3, 3, 3), spacing=5.0)
        edges = tetrahedron_edges(cells)
        deformed = vertices + np.random.default_rng(0).normal(0.0, 0.6, vertices.shape)
        value, gradient = elastic_value_and_gradient(vertices, deformed, edges)
        self.assertGreater(value, 0.0)
        step = 1e-7
        for index in (1, 7, 13):
            for axis in range(3):
                plus = deformed.copy()
                plus[index, axis] += step
                minus = deformed.copy()
                minus[index, axis] -= step
                numeric = (
                    elastic_value_and_gradient(vertices, plus, edges)[0]
                    - elastic_value_and_gradient(vertices, minus, edges)[0]
                ) / (2.0 * step)
                self.assertAlmostEqual(numeric / gradient[index, axis], 1.0, places=4)

    def test_rest_configuration_has_zero_energy(self) -> None:
        vertices, cells = lattice_mesh((3, 3, 3), spacing=5.0)
        edges = tetrahedron_edges(cells)
        value, gradient = elastic_value_and_gradient(vertices, vertices, edges)
        self.assertAlmostEqual(value, 0.0)
        self.assertLess(float(np.abs(gradient).max()), 1e-12)


class CollisionBarrierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vertices, self.cells = lattice_mesh((3, 3, 3), spacing=5.0)
        self.faces = boundary_faces(self.cells)

    def test_rest_configuration_is_free_of_contact(self) -> None:
        value, gradient = collision_value_and_gradient(
            self.vertices, self.faces, thickness=1.0
        )
        self.assertEqual(value, 0.0)
        self.assertLess(float(np.abs(gradient).max()), 1e-12)

    def test_gradient_matches_finite_differences(self) -> None:
        # Checked in a configuration where the term is actually active. The
        # reaction on each triangle is distributed by barycentric weight rather
        # than evenly, without which the gradient does not match the energy and
        # a line search is handed a direction that need not descend.
        squashed = self.vertices.copy()
        squashed[:, 2] *= 0.08
        value, gradient = collision_value_and_gradient(
            squashed, self.faces, thickness=1.0
        )
        self.assertGreater(value, 0.0)
        step = 1e-7
        checked = 0
        for index in (0, 4, 9, 17):
            for axis in range(3):
                plus = squashed.copy()
                plus[index, axis] += step
                minus = squashed.copy()
                minus[index, axis] -= step
                numeric = (
                    collision_value_and_gradient(plus, self.faces, thickness=1.0)[0]
                    - collision_value_and_gradient(minus, self.faces, thickness=1.0)[0]
                ) / (2.0 * step)
                if abs(numeric) < 1e-4 and abs(gradient[index, axis]) < 1e-4:
                    continue
                self.assertAlmostEqual(numeric / gradient[index, axis], 1.0, places=4)
                checked += 1
        self.assertGreater(checked, 0, "no live gradient components were compared")

    def test_thickness_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            collision_value_and_gradient(self.vertices, self.faces, thickness=0.0)

    def test_collision_weight_requires_faces(self) -> None:
        with self.assertRaises(ValueError):
            fit_deformation(
                self.vertices, self.cells,
                np.array([0]), self.vertices[:1],
                collision_weight=1.0,
            )


class FitDeformationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vertices, self.cells = lattice_mesh((4, 4, 4), spacing=5.0)
        top = np.flatnonzero(np.isclose(self.vertices[:, 2], self.vertices[:, 2].max()))
        bottom = np.flatnonzero(
            np.isclose(self.vertices[:, 2], self.vertices[:, 2].min())
        )
        extent = self.vertices[:, 2].max() - self.vertices[:, 2].min()
        targets = self.vertices[top].copy()
        targets[:, 2] -= 1.2 * extent
        self.indices = np.concatenate([top, bottom])
        self.positions = np.concatenate([targets, self.vertices[bottom]])

    def _fit(self, barrier_weight: float):
        return fit_deformation(
            self.vertices,
            self.cells,
            self.indices,
            self.positions,
            barrier_weight=barrier_weight,
            elastic_weight=0.05,
            max_iterations=600,
            step=0.5,
        )

    def test_barrier_prevents_inversion_where_the_ablation_does_not(self) -> None:
        # The elastic weight is identical in both arms, so the only difference
        # is the barrier. Without that matching the disabled arm would invert
        # for want of any interior coupling rather than for want of a barrier.
        guarded = self._fit(1.0)
        unguarded = self._fit(0.0)
        self.assertEqual(guarded.inverted_cells, 0)
        self.assertGreater(unguarded.inverted_cells, 0)
        self.assertGreater(guarded.minimum_determinant, 0.0)
        self.assertLess(unguarded.minimum_determinant, 0.0)

    def test_the_barrier_is_paid_for_in_data_error(self) -> None:
        # An infeasible target set cannot be met without inversion, so refusing
        # to invert must cost fit. A barrier that were free here would mean the
        # task was not actually demanding inversion.
        guarded = self._fit(1.0)
        unguarded = self._fit(0.0)
        self.assertGreater(guarded.data_rmse_mm, unguarded.data_rmse_mm)

    def test_rejects_negative_weights(self) -> None:
        with self.assertRaises(ValueError):
            fit_deformation(
                self.vertices, self.cells, self.indices, self.positions,
                barrier_weight=-1.0,
            )

    def test_rejects_mismatched_targets(self) -> None:
        with self.assertRaises(ValueError):
            fit_deformation(
                self.vertices, self.cells, self.indices, self.positions[:-1],
            )


if __name__ == "__main__":
    unittest.main()
