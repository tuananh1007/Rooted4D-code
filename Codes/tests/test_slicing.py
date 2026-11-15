import unittest

import numpy as np

from rooted4d.observation import (
    CMRPlane,
    intersect_surface_with_plane,
    symmetric_contour_chamfer,
)


class AnalyticSlicingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plane = CMRPlane(
            origin=np.zeros(3),
            row_direction=np.array([1.0, 0.0, 0.0]),
            column_direction=np.array([0.0, 1.0, 0.0]),
        )

    def test_triangle_plane_intersection_is_exact(self) -> None:
        vertices = np.array(
            [[-1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [0.0, 1.0, -1.0]]
        )
        result = intersect_surface_with_plane(
            vertices,
            np.array([[0, 1, 2]], dtype=np.int64),
            self.plane,
        )
        self.assertEqual(result.segments_3d.shape, (1, 2, 3))
        recovered = result.segments_3d[0][np.argsort(result.segments_3d[0, :, 1])]
        np.testing.assert_allclose(
            recovered,
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
            atol=1e-12,
        )
        self.assertEqual(result.source_face_indices.tolist(), [0])

    def test_coplanar_face_is_reported_not_arbitrarily_sliced(self) -> None:
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        result = intersect_surface_with_plane(
            vertices,
            np.array([[0, 1, 2]], dtype=np.int64),
            self.plane,
        )
        self.assertEqual(len(result.segments_3d), 0)
        self.assertEqual(result.coplanar_face_indices.tolist(), [0])

    def test_contour_distance_is_zero_for_matching_samples(self) -> None:
        segments = np.array([[[0.0, 0.0], [1.0, 0.0]]])
        targets = np.array([[0.125, 0.0], [0.375, 0.0], [0.625, 0.0], [0.875, 0.0]])
        self.assertAlmostEqual(
            symmetric_contour_chamfer(
                segments,
                targets,
                samples_per_segment=4,
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
