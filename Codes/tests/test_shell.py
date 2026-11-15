import unittest

import numpy as np

from rooted4d.geometry import (
    build_shell_reference,
    shell_bending_energy,
    shell_membrane_energy,
)


class ShellRegularizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        self.faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        self.reference = build_shell_reference(self.vertices, self.faces)

    def test_energies_vanish_at_rest_and_under_rigid_motion(self) -> None:
        self.assertAlmostEqual(shell_membrane_energy(self.vertices, self.reference), 0.0)
        self.assertAlmostEqual(shell_bending_energy(self.vertices, self.reference), 0.0)

        angle = 0.61
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        moved = self.vertices @ rotation.T + np.array([4.0, -3.0, 2.0])
        self.assertLess(shell_membrane_energy(moved, self.reference), 1e-25)
        self.assertLess(shell_bending_energy(moved, self.reference), 1e-25)

    def test_membrane_detects_in_plane_stretch(self) -> None:
        stretched = self.vertices.copy()
        stretched[:, 0] *= 1.25
        self.assertGreater(shell_membrane_energy(stretched, self.reference), 1e-4)

    def test_bending_detects_fold(self) -> None:
        folded = self.vertices.copy()
        folded[3, 2] = 0.6
        self.assertGreater(shell_bending_energy(folded, self.reference), 1e-4)

    def test_nonmanifold_reference_is_rejected(self) -> None:
        faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 3]], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "non-manifold"):
            build_shell_reference(self.vertices, faces)


if __name__ == "__main__":
    unittest.main()
