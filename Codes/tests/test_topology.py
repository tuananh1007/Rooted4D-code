import unittest

import numpy as np

from rooted4d.metrics import (
    component_overlap_volumes,
    count_self_intersections,
    surface_topology_report,
    triangles_intersect,
)


class TopologyMetricTests(unittest.TestCase):
    def test_empty_surface_is_not_closed(self) -> None:
        report = surface_topology_report(
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.int64),
        )
        self.assertFalse(report.closed)
        self.assertIsNone(report.signed_enclosed_volume)

    def test_closed_tetrahedron_report(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        faces = np.array(
            [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]],
            dtype=np.int64,
        )
        report = surface_topology_report(vertices, faces)
        self.assertTrue(report.closed)
        self.assertTrue(report.edge_manifold)
        self.assertEqual(report.euler_characteristic, 2)
        self.assertEqual(report.self_intersections, 0)
        self.assertAlmostEqual(report.signed_enclosed_volume or 0.0, 1.0 / 6.0)

    def test_coplanar_and_transverse_triangle_predicates(self) -> None:
        first = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertFalse(triangles_intersect(first, first + np.array([3.0, 0.0, 0.0])))
        self.assertTrue(triangles_intersect(first, first + np.array([0.2, 0.2, 0.0])))
        crossing = np.array(
            [[0.25, 0.25, -1.0], [0.25, 0.25, 1.0], [0.75, 0.10, 0.0]]
        )
        self.assertTrue(triangles_intersect(first, crossing))

    def test_self_intersection_count_uses_nonadjacent_faces(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.25, 0.25, -1.0],
                [0.25, 0.25, 1.0],
                [0.75, 0.10, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        self.assertEqual(count_self_intersections(vertices, faces), 1)

    def test_component_overlap_is_reported_in_physical_units(self) -> None:
        masks = {
            "lv": np.array([[[True, True], [False, False]]]),
            "la": np.array([[[False, True], [True, False]]]),
        }
        report = component_overlap_volumes(masks, voxel_volume=2.5)
        self.assertEqual(report["la|lv"], 2.5)
        self.assertEqual(report["total_overlap_volume"], 2.5)


if __name__ == "__main__":
    unittest.main()
