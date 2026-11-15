import unittest

import numpy as np

from rooted4d.geometry import RigidTransform


class GeometryTests(unittest.TestCase):
    def test_rigid_inverse(self) -> None:
        transform = RigidTransform.from_axis_angle(
            np.array([0.2, -0.1, 0.05]),
            np.array([3.0, -4.0, 9.0]),
        )
        points = np.array([[1.0, 2.0, 3.0], [-5.0, 0.5, 2.0]])
        recovered = transform.inverse().apply(transform.apply(points))
        np.testing.assert_allclose(recovered, points, atol=1e-10)


if __name__ == "__main__":
    unittest.main()

