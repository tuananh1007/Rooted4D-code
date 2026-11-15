"""Tests for fitting MVSeg2023 valves to the shared topology."""

import unittest

import numpy as np

from rooted4d.donors.valve import ANTERIOR_LABEL, POSTERIOR_LABEL, fit_valve


def _synthetic_valve(size: int = 48, spacing: float = 1.0) -> np.ndarray:
    """A funnel labelled like MVSeg2023: wide at the annulus, narrow at the edge.

    The posterior leaflet is given the larger annular arc, as anatomy has it, so
    a fit that swaps them is detectable.
    """

    volume = np.zeros((size, size, size), dtype=np.uint8)
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size]
    centre = size / 2
    height = zz - centre
    radius = np.hypot(yy - centre, xx - centre)
    angle = np.arctan2(yy - centre, xx - centre)
    # Funnel: outer radius shrinks as height rises.
    outer = 18.0 - 0.55 * (height + centre) * 0.5
    shell = (radius < outer) & (radius > outer - 3.0) & (np.abs(height) < 12)
    volume[shell & (np.abs(angle) < np.pi / 3)] = ANTERIOR_LABEL
    volume[shell & (np.abs(angle) >= np.pi / 3)] = POSTERIOR_LABEL
    return volume


class ValveFitTest(unittest.TestCase):
    def setUp(self):
        self.volume = _synthetic_valve()
        self.fit = fit_valve(self.volume, (1.0, 1.0, 1.0), n_annulus=24, n_depth=5)

    def test_a_valve_is_fitted(self):
        self.assertIsNotNone(self.fit)

    def test_topology_is_fixed_across_cases(self):
        other = fit_valve(
            _synthetic_valve(size=56), (1.0, 1.0, 1.0), n_annulus=24, n_depth=5
        )
        self.assertIsNotNone(other)
        self.assertEqual(self.fit.annulus.shape, other.annulus.shape)
        self.assertEqual(self.fit.n_depth, other.n_depth)

    def test_annulus_has_one_vertex_per_sector(self):
        self.assertEqual(self.fit.annulus.shape, (24, 3))
        self.assertTrue(np.all(np.isfinite(self.fit.annulus)))

    def test_leaflet_roots_are_annular_vertices_not_copies(self):
        """Attachment is structural in the donor, as it is in the model."""

        for sheet, indices in (
            (self.fit.anterior, self.fit.anterior_root_indices),
            (self.fit.posterior, self.fit.posterior_root_indices),
        ):
            for row, index in zip(sheet, indices):
                # The root row coincides with its annular vertex, up to the
                # averaging done where both leaflets claim a sector.
                self.assertLess(
                    float(np.linalg.norm(row[0] - self.fit.annulus[index])), 3.0
                )

    def test_posterior_takes_the_larger_annular_arc(self):
        """Voting over whole sectors instead of roots reversed this."""

        self.assertGreater(len(self.fit.posterior), len(self.fit.anterior))

    def test_depth_rows_run_from_root_outward(self):
        for sheet in (self.fit.anterior, self.fit.posterior):
            spans = np.linalg.norm(sheet[:, -1] - sheet[:, 0], axis=1)
            self.assertTrue(np.all(spans > 0.5))

    def test_an_empty_volume_is_rejected(self):
        self.assertIsNone(
            fit_valve(np.zeros((32, 32, 32), dtype=np.uint8), (1.0, 1.0, 1.0))
        )

    def test_a_single_leaflet_is_rejected(self):
        volume = _synthetic_valve()
        volume[volume == POSTERIOR_LABEL] = 0
        self.assertIsNone(fit_valve(volume, (1.0, 1.0, 1.0), n_annulus=24))
