import unittest

import numpy as np

from rooted4d.geometry import (
    deformation_jacobian_determinants,
    tetrahedron_jacobian_barrier,
    tetrahedron_quality_report,
    tetrahedron_scaled_jacobians,
)


class TetrahedronValidityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.regular = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, np.sqrt(3.0) / 2.0, 0.0],
                [0.5, np.sqrt(3.0) / 6.0, np.sqrt(2.0 / 3.0)],
            ]
        )
        self.tetrahedra = np.array([[0, 1, 2, 3]], dtype=np.int64)

    def test_regular_tetrahedron_has_unit_scaled_jacobian(self) -> None:
        scaled = tetrahedron_scaled_jacobians(self.regular, self.tetrahedra)
        np.testing.assert_allclose(scaled, [1.0], atol=1e-12)

    def test_rigid_and_uniform_scale_jacobian(self) -> None:
        translated = 1.5 * self.regular + np.array([3.0, -4.0, 2.0])
        determinant = deformation_jacobian_determinants(
            self.regular,
            translated,
            self.tetrahedra,
        )
        np.testing.assert_allclose(determinant, [1.5**3], atol=1e-12)

    def test_barrier_is_singular_at_invalid_configuration(self) -> None:
        near = self.regular.copy()
        near[3] = self.regular[0] + 0.10 * (self.regular[3] - self.regular[0])
        valid_value = tetrahedron_jacobian_barrier(
            self.regular,
            near,
            self.tetrahedra,
            minimum_determinant=0.05,
            activation_determinant=0.20,
        )
        self.assertTrue(np.isfinite(valid_value))
        self.assertGreater(valid_value, 0.0)

        inverted = self.regular.copy()
        inverted[[1, 2]] = inverted[[2, 1]]
        invalid_value = tetrahedron_jacobian_barrier(
            self.regular,
            inverted,
            self.tetrahedra,
        )
        self.assertTrue(np.isinf(invalid_value))

    def test_bounded_coordinate_offsets_do_not_imply_validity(self) -> None:
        right_tet = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        offsets = np.array(
            [
                [0.45, 0.45, 0.45],
                [-0.45, -0.45, -0.45],
                [-0.45, -0.45, -0.45],
                [-0.45, -0.45, -0.45],
            ]
        )
        report = tetrahedron_quality_report(right_tet + offsets, self.tetrahedra)
        self.assertEqual(report.inverted_or_degenerate, 1)
        self.assertLess(report.minimum_scaled_jacobian, 0.0)


if __name__ == "__main__":
    unittest.main()
