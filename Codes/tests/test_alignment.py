import unittest

import numpy as np

from rooted4d.alignment import (
    area_weighted_vertex_normals,
    barycentric_projection_matrix,
    directed_arc_alignment_loss,
    directed_arc_coordinates,
    normal_compatibility_loss,
    patch_admissibility_mask,
    soft_correspondence_weights,
)


class NormalFeatureTests(unittest.TestCase):
    def test_area_weighted_normals_on_plane(self) -> None:
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        normals = area_weighted_vertex_normals(vertices, faces)
        np.testing.assert_allclose(normals, np.tile([0.0, 0.0, 1.0], (4, 1)))

    def test_opposite_normals_are_penalized_at_zero_distance(self) -> None:
        correspondence = np.array([[1.0]])
        aligned = normal_compatibility_loss(
            np.array([[0.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0, 1.0]]),
            correspondence,
        )
        reversed_loss = normal_compatibility_loss(
            np.array([[0.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0, -1.0]]),
            correspondence,
        )
        self.assertEqual(aligned, 0.0)
        self.assertEqual(reversed_loss, 1.0)


class CorrespondenceTests(unittest.TestCase):
    def test_patch_mask_blocks_cross_anatomy_matches(self) -> None:
        source_labels = np.array(["lv", "la"])
        target_labels = np.array(["la", "lv"])
        mask = patch_admissibility_mask(source_labels, target_labels)
        weights = soft_correspondence_weights(
            np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            admissible_mask=mask,
        )
        self.assertEqual(weights[0, 0], 0.0)
        self.assertEqual(weights[1, 1], 0.0)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)

    def test_normal_cost_changes_soft_assignment(self) -> None:
        weights = soft_correspondence_weights(
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            source_normals=np.array([[0.0, 0.0, 1.0]]),
            target_normals=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
            normal_weight=4.0,
            temperature=0.25,
        )
        self.assertGreater(weights[0, 0], 0.999)

    def test_barycentric_projection_is_row_stochastic(self) -> None:
        projection = barycentric_projection_matrix(
            np.array([[0, 1, 2], [1, 2, 3]]),
            np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]]),
            n_target_vertices=4,
        )
        np.testing.assert_allclose(projection.sum(axis=1), 1.0)
        target = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        )
        np.testing.assert_allclose(projection[0] @ target, [0.3, 0.5, 0.0])


class DirectedArcTests(unittest.TestCase):
    def setUp(self) -> None:
        theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        self.curve = np.stack((np.cos(theta), 0.8 * np.sin(theta), 0.1 * np.cos(2 * theta)), axis=1)

    def test_anchor_and_orientation_are_explicit(self) -> None:
        forward = directed_arc_coordinates(self.curve, anchor_index=2, orientation=1)
        reverse = directed_arc_coordinates(self.curve, anchor_index=2, orientation=-1)
        self.assertEqual(forward[2], 0.0)
        self.assertEqual(reverse[2], 0.0)
        self.assertGreater(directed_arc_alignment_loss(forward, reverse), 1e-3)

    def test_circular_wrap_has_small_loss(self) -> None:
        left = np.array([0.99, 0.01])
        right = np.array([0.01, 0.99])
        self.assertLess(directed_arc_alignment_loss(left, right), 5e-4)


if __name__ == "__main__":
    unittest.main()
