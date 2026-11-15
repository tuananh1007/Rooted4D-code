import unittest

import numpy as np

from rooted4d.geometry import (
    BarycentricEmbedding,
    bounded_residual,
    fit_tetrahedral_embedding,
)


class EmbeddingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parent = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self.embedding = BarycentricEmbedding(
            simplex_vertex_indices=np.array([[0, 1, 2, 3]]),
            barycentric_weights=np.array([[0.1, 0.2, 0.3, 0.4]]),
            n_parent_vertices=4,
        )

    def test_landmark_is_tetrahedrally_embedded(self) -> None:
        point = self.embedding.evaluate(self.parent)
        np.testing.assert_allclose(point[0], [0.2, 0.3, 0.4])

    def test_embedding_follows_parent_translation(self) -> None:
        translation = np.array([4.0, -2.0, 7.0])
        before = self.embedding.evaluate(self.parent)
        after = self.embedding.evaluate(self.parent + translation)
        np.testing.assert_allclose(after - before, translation[None, :])

    def test_residual_is_smoothly_bounded(self) -> None:
        residual = bounded_residual(np.array([[10.0, 0.0, 0.0]]), maximum_norm=2.0)
        self.assertLessEqual(float(np.linalg.norm(residual[0])), 2.0)
        self.assertGreater(float(np.linalg.norm(residual[0])), 1.9)

    def test_embedding_can_be_fitted_from_declared_host_tetrahedron(self) -> None:
        expected = np.array([[0.2, 0.3, 0.1]])
        embedding = fit_tetrahedral_embedding(
            expected,
            self.parent,
            np.array([[0, 1, 2, 3]], dtype=np.int64),
            np.array([0], dtype=np.int64),
        )
        np.testing.assert_allclose(embedding.evaluate(self.parent), expected, atol=1e-12)

    def test_outside_host_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            fit_tetrahedral_embedding(
                np.array([[2.0, 2.0, 2.0]]),
                self.parent,
                np.array([[0, 1, 2, 3]], dtype=np.int64),
                np.array([0], dtype=np.int64),
            )


if __name__ == "__main__":
    unittest.main()
