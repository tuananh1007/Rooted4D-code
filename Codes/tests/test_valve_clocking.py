"""Anatomical clocking of composite valves (E108): the anterior leaflet faces the septum."""

import unittest

import numpy as np

from rooted4d.active.acdc import LVFit
from rooted4d.donors.anatomy import (
    anterior_span,
    chamber_vertices_clocked,
    fit_valve_commissural,
    resample_ring_at_angles,
    ring_target_angles,
    septal_direction,
    valve_clocking,
    valve_vertices_commissural,
)
from rooted4d.donors.valve import ANTERIOR_LABEL, POSTERIOR_LABEL
from rooted4d.donors.valve_fixed import fit_valve_fixed, valve_commissures
from rooted4d.synthetic.donors import Donor, combine_donors, make_toy_complex

N = 24
SPACING = (0.6, 0.6, 0.6)


def _synthetic_valve(anterior_axis_deg: float, half_anterior_deg: float = 55.0) -> np.ndarray:
    """A saddle-shaped funnel, 20 mm annulus to 10 mm coaptation (as in test_valve_fixed)."""

    shape = (80, 80, 36)
    grid = np.stack(np.meshgrid(*[np.arange(n) * s for n, s in zip(shape, SPACING)], indexing="ij"), axis=-1)
    offset = grid - np.array([shape[0] * SPACING[0] / 2, shape[1] * SPACING[1] / 2, 3.0])
    theta = np.arctan2(offset[..., 1], offset[..., 0])
    relative = np.angle(np.exp(1j * (theta - np.radians(anterior_axis_deg))))
    depth = offset[..., 2] - 1.5 * np.cos(2.0 * relative)
    radius = np.hypot(offset[..., 0], offset[..., 1])
    on_sheet = (depth >= 0.0) & (depth <= 12.0) & (np.abs(radius - (20.0 - (10.0 / 12.0) * depth)) <= 1.0)
    labels = np.zeros(shape, dtype=np.uint8)
    anterior = np.abs(relative) <= np.radians(half_anterior_deg)
    labels[on_sheet & anterior] = ANTERIOR_LABEL
    labels[on_sheet & ~anterior] = POSTERIOR_LABEL
    return labels


def _short_axis_stack(septal_deg: float) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Cavity disc, myocardium ring and a right ventricle over +-60 degrees about ``septal_deg``.

    Angles are ``arctan2(row, column)`` about the cavity centre, the convention
    ``fit_lv_surface`` samples contours in.
    """

    zooms = (1.5, 1.5, 10.0)
    shape = (80, 80, 8)
    rows, columns = np.meshgrid(np.arange(shape[0]) * zooms[0], np.arange(shape[1]) * zooms[1], indexing="ij")
    relative_row, relative_column = rows - 60.0, columns - 57.0
    radius = np.hypot(relative_row, relative_column)
    angle = np.arctan2(relative_row, relative_column)
    near_septum = np.abs(np.angle(np.exp(1j * (angle - np.radians(septal_deg))))) <= np.radians(60.0)
    labels = np.zeros(shape, dtype=np.uint8)
    for k in range(shape[2]):
        shrink = 1.0 - 0.08 * k
        plane = np.zeros(shape[:2], dtype=np.uint8)
        plane[radius <= 15.0 * shrink] = 3
        plane[(radius > 15.0 * shrink) & (radius <= 22.0 * shrink)] = 2
        plane[(radius > 22.0 * shrink) & (radius <= 32.0 * shrink) & near_septum] = 1
        labels[:, :, k] = plane
    return labels, zooms


def _fit(n_rings: int = 8, n_angles: int = 96) -> LVFit:
    """An elliptical chamber in ACDC fit layout: rings from base (z = 0) to apex, then two caps."""

    t = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
    rings = []
    for k in range(n_rings):
        r = (24.0 - 2.5 * k) * (1.0 + 0.15 * np.cos(2.0 * (t - 0.3)))
        rings.append(np.stack([5.0 + r * np.sin(t), -3.0 + r * np.cos(t), np.full_like(t, 10.0 * k)], axis=1))
    rings = np.concatenate(rings)
    base, apex = rings[:n_angles].mean(axis=0), rings[-n_angles:].mean(axis=0) + np.array([0.0, 0.0, 5.0])
    return LVFit(vertices=np.vstack([rings, base, apex]), n_rings=n_rings, n_angles=n_angles,
                 n_slices_used=n_rings, patient="synthetic", frame=1)


def _height(vertices: np.ndarray, groups: dict) -> float:
    ring = vertices[groups["annulus"]].mean(axis=0)
    apex = vertices[np.setdiff1d(groups["cmr_visible"], groups["annulus"])[-1]]
    axis = (apex - ring) / np.linalg.norm(apex - ring)
    return float(((vertices[groups["leaflet_free"]] - ring) @ axis).mean())


class SeptalDirectionTest(unittest.TestCase):
    def test_bisects_the_contact_arc(self):
        for septal_deg in (-150.0, 20.0, 110.0):
            labels, zooms = _short_axis_stack(septal_deg)
            found = septal_direction(labels, zooms)
            self.assertIsNotNone(found)
            expected = np.radians(septal_deg)
            error = np.degrees(np.arccos(found.direction[:2] @ np.array([np.sin(expected), np.cos(expected)])))
            self.assertLess(error, 3.0, septal_deg)
            self.assertEqual(found.direction[2], 0.0)
            # The basal half of the eight cavity slices.
            self.assertEqual(found.slices, (0, 1, 2, 3))

    def test_none_without_a_right_ventricle(self):
        labels, zooms = _short_axis_stack(0.0)
        labels[labels == 1] = 0
        self.assertIsNone(septal_direction(labels, zooms))


class ClockedChamberTest(unittest.TestCase):
    def test_ring_vertices_sit_at_the_target_angles(self):
        alpha = np.radians(130.0)
        septal = np.array([np.sin(0.7), np.cos(0.7), 0.0])
        fit = _fit()
        vertices = chamber_vertices_clocked(fit, septal, alpha, n_annulus=N)
        groups = make_toy_complex(n_annulus=N).groups
        annulus = vertices[groups["annulus"]]
        # Angles are measured about the fitted ring's own centroid (uniform in angle);
        # the resampled ring's is pulled toward its denser anterior half.
        centre = fit.vertices[: fit.n_angles].mean(axis=0)
        apex = vertices[np.setdiff1d(groups["cmr_visible"], groups["annulus"])[-1]]
        u = (apex - centre) / np.linalg.norm(apex - centre)
        e1 = septal - (septal @ u) * u
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        centred = annulus - centre
        angles = np.arctan2(centred @ e2, centred @ e1)
        error = np.degrees(np.abs(np.angle(np.exp(1j * (angles - ring_target_angles(N, alpha))))))
        self.assertLess(float(error.max()), 1.0)
        # The anterior half is centred on the septum.
        self.assertAlmostEqual(float(np.degrees(ring_target_angles(N, alpha)[: N // 2].mean())), 0.0, places=9)

    def test_rejects_a_ring_that_is_not_star_shaped(self):
        t = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
        ring = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)
        ring[[5, 6]] = ring[[6, 5]]
        with self.assertRaises(ValueError):
            resample_ring_at_angles(ring, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.zeros(3))


class CommissuralValveTest(unittest.TestCase):
    def test_equals_the_fixed_encoding_without_gaps(self):
        labels = _synthetic_valve(30.0)
        fixed = fit_valve_fixed(labels, SPACING, n_anterior=N // 2, n_posterior=N // 2)
        encoded, missing = fit_valve_commissural(labels, SPACING, n_per_leaflet=N // 2)
        self.assertEqual(missing, 0)
        np.testing.assert_allclose(encoded.anterior, fixed.anterior, atol=1e-9)
        np.testing.assert_allclose(encoded.posterior, fixed.posterior, atol=1e-9)
        self.assertEqual(encoded.commissure_angles, fixed.commissure_angles)

    def test_bridges_a_gap_at_a_commissure(self):
        # Clear the posterior leaflet for 12 degrees past one commissure: the walk
        # stops in the gap, and the posterior nodes there hold no tissue.
        labels = _synthetic_valve(30.0)
        shape = labels.shape
        grid = np.stack(np.meshgrid(*[np.arange(n) * s for n, s in zip(shape, SPACING)], indexing="ij"), axis=-1)
        offset = grid - np.array([shape[0] * SPACING[0] / 2, shape[1] * SPACING[1] / 2, 3.0])
        theta = np.degrees(np.arctan2(offset[..., 1], offset[..., 0]))
        gap = (theta > 85.0) & (theta < 97.0) & (labels == POSTERIOR_LABEL)
        labels[gap] = 0
        self.assertIsNone(fit_valve_fixed(labels, SPACING, n_anterior=48, n_posterior=48))
        encoded, missing = fit_valve_commissural(labels, SPACING, n_per_leaflet=48)
        self.assertGreater(missing, 0)
        self.assertLess(missing, 12)
        vertices = np.vstack([encoded.annulus, encoded.anterior[:, -1], encoded.posterior[:, -1]])
        self.assertTrue(np.isfinite(vertices).all())
        # Bridged ring nodes continue the ring: no step longer than twice the median.
        ring = np.vstack([encoded.annulus, encoded.annulus[:1]])
        steps = np.linalg.norm(np.diff(ring, axis=0), axis=1)
        self.assertLess(float(steps.max()), 2.5 * float(np.median(steps)))

    def test_gives_up_when_a_leaflet_is_mostly_missing(self):
        labels = _synthetic_valve(30.0)
        shape = labels.shape
        grid = np.stack(np.meshgrid(*[np.arange(n) * s for n, s in zip(shape, SPACING)], indexing="ij"), axis=-1)
        offset = grid - np.array([shape[0] * SPACING[0] / 2, shape[1] * SPACING[1] / 2, 3.0])
        theta = np.degrees(np.arctan2(offset[..., 1], offset[..., 0]))
        labels[((theta > 85.0) | (theta < -100.0)) & (labels == POSTERIOR_LABEL)] = 0  # 85 to 260 degrees
        self.assertIsNone(fit_valve_commissural(labels, SPACING, n_per_leaflet=48))


class ClockedCompositeTest(unittest.TestCase):
    def setUp(self):
        self.groups = make_toy_complex(n_annulus=N).groups
        self.septal = np.array([np.sin(-2.0), np.cos(-2.0), 0.0])

    def _composite(self, anterior_axis_deg: float, orient: bool = True):
        labels = _synthetic_valve(anterior_axis_deg)
        fit = fit_valve_fixed(labels, SPACING, n_anterior=N // 2, n_posterior=N // 2)
        alpha = anterior_span(fit)
        heart = Donor("h", "ph", "whole_heart", chamber_vertices_clocked(_fit(), self.septal, alpha, n_annulus=N),
                      "test", None)
        valve = Donor("v", "pv", "valve", valve_vertices_commissural(fit, n_annulus=N), "test", None)
        clocking = valve_clocking(heart.vertices, valve.vertices, valve_commissures(labels, SPACING).reference,
                                  self.septal, N, orient=orient)
        return combine_donors(heart, valve, N, orient_valve=orient), alpha, clocking

    def test_the_anterior_leaflet_faces_the_septum_whatever_the_valve_frame(self):
        for axis_deg in (30.0, 160.0, -95.0):
            composite, alpha, clocking = self._composite(axis_deg)
            self.assertAlmostEqual(np.degrees(alpha), 110.0, delta=10.0)
            self.assertLess(abs(clocking), 5.0, axis_deg)
            self.assertGreater(_height(composite, self.groups), 0.0)
            # The anterior free edge sits on the septal side of the annulus too.
            centre = composite[self.groups["annulus"]].mean(axis=0)
            anterior = composite[self.groups["leaflet_free"][: N // 2]].mean(axis=0) - centre
            self.assertGreater(float(anterior[:2] @ self.septal[:2]), 0.0)

    def test_an_arbitrarily_numbered_valve_is_turned_arbitrarily(self):
        # The pool's previous valve numbering (a principal axis of arbitrary sign),
        # imitated by rotating the commissure-numbered ring by a quarter turn.
        labels = _synthetic_valve(30.0)
        fit = fit_valve_fixed(labels, SPACING, n_anterior=N // 2, n_posterior=N // 2)
        heart = chamber_vertices_clocked(_fit(), self.septal, anterior_span(fit), n_annulus=N)
        valve = valve_vertices_commissural(fit, n_annulus=N)
        shifted = valve.copy()
        roll = np.roll(np.arange(N), N // 4)
        shifted[self.groups["annulus"]] = valve[self.groups["annulus"][roll]]
        shifted[self.groups["leaflet_free"]] = valve[self.groups["leaflet_free"][roll]]
        clocking = valve_clocking(heart, shifted, valve_commissures(labels, SPACING).reference, self.septal, N)
        self.assertGreater(abs(clocking), 45.0)

    def test_orientation_is_already_consistent(self):
        oriented, _, _ = self._composite(30.0, orient=True)
        as_built, _, _ = self._composite(30.0, orient=False)
        np.testing.assert_array_equal(oriented, as_built)


if __name__ == "__main__":
    unittest.main()
