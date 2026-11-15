"""Tests for AHA-style regional quantities."""

import unittest

import numpy as np

from rooted4d.active.regional import (
    N_SEGMENTS,
    regional_excursion,
    segment_assignment,
)


class SegmentAssignmentTest(unittest.TestCase):
    def test_every_wall_vertex_gets_a_segment(self):
        rings, angles = 9, 36
        assignment = segment_assignment(rings, angles)
        wall = assignment[: rings * angles]
        self.assertTrue(np.all(wall >= 0))
        self.assertTrue(np.all(wall < N_SEGMENTS))

    def test_cap_vertices_belong_to_no_segment(self):
        rings, angles = 9, 36
        assignment = segment_assignment(rings, angles)
        self.assertEqual(int(assignment[rings * angles]), -1)
        self.assertEqual(int(assignment[rings * angles + 1]), -1)

    def test_all_sixteen_segments_are_populated(self):
        assignment = segment_assignment(9, 36)
        counts = np.bincount(assignment[assignment >= 0], minlength=N_SEGMENTS)
        self.assertEqual(len(counts), N_SEGMENTS)
        self.assertTrue(np.all(counts > 0))

    def test_basal_and_mid_have_six_sectors_and_apical_four(self):
        assignment = segment_assignment(9, 36)
        self.assertEqual(len(set(assignment[assignment >= 0].tolist()) & set(range(0, 6))), 6)
        self.assertEqual(len(set(assignment[assignment >= 0].tolist()) & set(range(6, 12))), 6)
        self.assertEqual(len(set(assignment[assignment >= 0].tolist()) & set(range(12, 16))), 4)

    def test_too_few_rings_is_rejected(self):
        with self.assertRaises(ValueError):
            segment_assignment(2, 32)


class RegionalExcursionTest(unittest.TestCase):
    def _cylinder(self, radius: float, rings: int = 9, angles: int = 36) -> np.ndarray:
        angle = np.linspace(0.0, 2 * np.pi, angles, endpoint=False)
        levels = np.linspace(0.0, 80.0, rings)
        points = []
        for z in levels:
            points.extend(
                [(radius * np.cos(a), radius * np.sin(a), z) for a in angle]
            )
        points.append((0.0, 0.0, -5.0))
        points.append((0.0, 0.0, 85.0))
        return np.asarray(points, dtype=np.float64)

    def test_uniform_contraction_gives_equal_excursion_everywhere(self):
        assignment = segment_assignment(9, 36)
        values = regional_excursion(
            self._cylinder(30.0), self._cylinder(20.0), assignment
        )
        self.assertEqual(values.shape, (N_SEGMENTS,))
        self.assertTrue(np.allclose(values, 10.0, atol=0.5))

    def test_no_motion_gives_zero_excursion(self):
        assignment = segment_assignment(9, 36)
        shape = self._cylinder(25.0)
        self.assertTrue(
            np.allclose(regional_excursion(shape, shape, assignment), 0.0, atol=1e-9)
        )

    def test_whole_heart_translation_is_not_read_as_motion(self):
        """Radii are taken about each phase's own centre, so a shift is not wall motion."""

        assignment = segment_assignment(9, 36)
        shape = self._cylinder(25.0)
        shifted = shape + np.array([12.0, -7.0, 0.0])
        values = regional_excursion(shape, shifted, assignment)
        self.assertTrue(np.allclose(values, 0.0, atol=1e-6))

    def test_a_localized_akinetic_segment_is_isolated(self):
        """A wall that does not move must show up in its own segment, not smeared."""

        rings, angles = 9, 36
        assignment = segment_assignment(rings, angles)
        end_diastolic = self._cylinder(30.0)
        end_systolic = self._cylinder(20.0)
        # Restore the original radius over one angular wedge: that wall is akinetic.
        wedge = [
            ring * angles + angle
            for ring in range(rings)
            for angle in range(angles)
            if 0 <= angle < angles // 6
        ]
        end_systolic[wedge] = end_diastolic[wedge]
        values = regional_excursion(end_diastolic, end_systolic, assignment)
        affected = [values[s] for s in (0, 6, 12)]
        others = [values[s] for s in range(N_SEGMENTS) if s not in (0, 6, 12)]
        self.assertLess(max(affected), min(others))


if __name__ == "__main__":
    unittest.main()
