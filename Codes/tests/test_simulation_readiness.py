"""Tests for downstream mesh usability of the reconstructed complex."""

import unittest

import numpy as np

from rooted4d.geometry import make_toy_complex
from rooted4d.metrics import (
    assess_simulation_readiness,
    interface_welds_at,
    weld_vertices,
)
from rooted4d.metrics.simulation_readiness import connected_components


def _complex_and_faces():
    complex_ = make_toy_complex(24)
    faces = np.concatenate(
        [
            complex_.faces["chamber"],
            complex_.faces["anterior_leaflet"],
            complex_.faces["posterior_leaflet"],
        ]
    )
    return complex_, faces


class WeldTest(unittest.TestCase):
    def test_coincident_vertices_merge(self):
        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        welded, relabel = weld_vertices(points, 1e-6)
        self.assertEqual(len(welded), 2)
        self.assertEqual(relabel[0], relabel[1])

    def test_vertices_beyond_tolerance_stay_separate(self):
        points = np.array([[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0]])
        self.assertEqual(len(weld_vertices(points, 1e-6)[0]), 2)
        self.assertEqual(len(weld_vertices(points, 1e-2)[0]), 1)

    def test_non_positive_tolerance_is_rejected(self):
        with self.assertRaises(ValueError):
            weld_vertices(np.zeros((2, 3)), 0.0)


class SimulationReadinessTest(unittest.TestCase):
    def test_exact_template_is_ready_at_every_tolerance(self):
        complex_, faces = _complex_and_faces()
        for tolerance in (1e-8, 1e-6, 1e-4, 1e-2):
            report = assess_simulation_readiness(
                complex_.global_vertices,
                faces,
                weld_tolerance_mm=tolerance,
                n_expected_vertices=len(complex_.global_vertices),
            )
            self.assertTrue(report.simulation_ready, tolerance)
            self.assertEqual(report.n_components, 1)

    def test_a_separated_interface_detaches_the_leaflets(self):
        """The failure a mesher actually produces: the leaflets fall off."""

        complex_, faces = _complex_and_faces()
        annulus = complex_.groups["annulus"]
        moved = complex_.global_vertices.copy()
        # Duplicate the annular ring and point the leaflets at the copy, offset
        # by more than the weld tolerance.
        offset = moved[annulus] + np.array([0.0, 0.0, 1e-3])
        extended = np.concatenate([moved, offset], axis=0)
        remap = {int(v): len(moved) + i for i, v in enumerate(annulus.tolist())}
        leaflet_faces = np.concatenate(
            [complex_.faces["anterior_leaflet"], complex_.faces["posterior_leaflet"]]
        )
        leaflet_faces = np.vectorize(lambda v: remap.get(int(v), int(v)))(leaflet_faces)
        detached_faces = np.concatenate([complex_.faces["chamber"], leaflet_faces])

        report = assess_simulation_readiness(
            extended,
            detached_faces,
            weld_tolerance_mm=1e-6,
            n_expected_vertices=len(complex_.global_vertices),
        )
        self.assertFalse(report.simulation_ready)
        self.assertGreater(report.n_components, 1)

    def test_a_loose_tolerance_reattaches_the_same_geometry(self):
        """Whether a gap matters is a question about tolerance, not about size."""

        complex_, faces = _complex_and_faces()
        annulus = complex_.groups["annulus"]
        offset = complex_.global_vertices[annulus] + np.array([0.0, 0.0, 1e-3])
        extended = np.concatenate([complex_.global_vertices, offset], axis=0)
        remap = {int(v): len(complex_.global_vertices) + i for i, v in enumerate(annulus.tolist())}
        leaflet_faces = np.concatenate(
            [complex_.faces["anterior_leaflet"], complex_.faces["posterior_leaflet"]]
        )
        leaflet_faces = np.vectorize(lambda v: remap.get(int(v), int(v)))(leaflet_faces)
        detached_faces = np.concatenate([complex_.faces["chamber"], leaflet_faces])
        report = assess_simulation_readiness(
            extended,
            detached_faces,
            weld_tolerance_mm=1e-2,
            n_expected_vertices=len(complex_.global_vertices),
        )
        self.assertEqual(report.n_components, 1)

    def test_components_counts_disconnected_pieces(self):
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        self.assertEqual(connected_components(6, faces), 2)
        joined = np.array([[0, 1, 2], [2, 3, 4]])
        self.assertEqual(connected_components(5, joined), 1)

    def test_weld_predicate_matches_the_tolerance(self):
        self.assertTrue(interface_welds_at(0.0, 1e-8))
        self.assertFalse(interface_welds_at(1e-6, 1e-8))
        with self.assertRaises(ValueError):
            interface_welds_at(1.0, 0.0)


if __name__ == "__main__":
    unittest.main()
