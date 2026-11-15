import unittest

import numpy as np

from rooted4d.constraints import NullspaceParameterization, constraint_matrix_from_equalities
from rooted4d.geometry import make_toy_complex


class AttachmentTests(unittest.TestCase):
    def test_shared_dofs_are_exact(self) -> None:
        toy = make_toy_complex()
        parts = toy.attachment_map.assemble(toy.global_vertices)
        self.assertEqual(toy.attachment_map.max_residual(parts), 0.0)

    def test_detached_local_copy_is_detected(self) -> None:
        toy = make_toy_complex()
        parts = {name: values.copy() for name, values in toy.attachment_map.assemble(toy.global_vertices).items()}
        parts["anterior_leaflet"][0, 2] += 1.25
        self.assertGreater(toy.attachment_map.max_residual(parts), 1.0)

    def test_merge_round_trip(self) -> None:
        toy = make_toy_complex()
        parts = toy.attachment_map.assemble(toy.global_vertices)
        merged = toy.attachment_map.merge(parts)
        np.testing.assert_allclose(merged, toy.global_vertices)

    def test_binary_assembly_matrix_matches_local_parts(self) -> None:
        toy = make_toy_complex()
        scalar = toy.attachment_map.scalar_assembly_matrix()
        coordinate = toy.attachment_map.coordinate_assembly_matrix()
        np.testing.assert_allclose(scalar.sum(axis=1), 1.0)
        self.assertTrue(np.all((scalar == 0.0) | (scalar == 1.0)))
        assembled = coordinate @ toy.global_vertices.reshape(-1)
        parts = toy.attachment_map.assemble(toy.global_vertices)
        expected = np.concatenate([parts[name] for name in toy.attachment_map.parts]).reshape(-1)
        np.testing.assert_allclose(assembled, expected)


class NullspaceTests(unittest.TestCase):
    def test_decoded_vectors_satisfy_equalities(self) -> None:
        constraint = constraint_matrix_from_equalities(4, [(0, 1), (1, 2)])
        parameterization = NullspaceParameterization.from_constraints(constraint)
        rng = np.random.default_rng(4)
        latent = rng.normal(size=(5, parameterization.latent_dimension))
        decoded = parameterization.decode(latent)
        residual = parameterization.residual(decoded)
        self.assertLess(float(np.max(np.abs(residual))), 1e-10)

    def test_projection_is_idempotent(self) -> None:
        constraint = constraint_matrix_from_equalities(3, [(0, 1)])
        parameterization = NullspaceParameterization.from_constraints(constraint)
        rng = np.random.default_rng(2)
        values = rng.normal(size=(3 * 3,))
        projected = parameterization.project(values)
        np.testing.assert_allclose(parameterization.project(projected), projected, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
