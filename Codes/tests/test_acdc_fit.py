"""Tests for fitting a fixed-topology LV surface to ACDC segmentations."""

import unittest

import numpy as np

from rooted4d.active.acdc import (
    build_acdc_shape_model,
    fit_lv_surface,
    lv_topology_faces,
    slice_cavity_contour,
)
from rooted4d.metrics.topology import signed_enclosed_volume, surface_topology_report


def _synthetic_stack(
    n_slices: int = 9, radius: float = 18.0, size: int = 64, taper: bool = True
) -> np.ndarray:
    """A tapering cylinder labelled like ACDC, cavity label 3."""

    volume = np.zeros((size, size, n_slices), dtype=np.uint8)
    rows, columns = np.mgrid[0:size, 0:size]
    for index in range(n_slices):
        scale = 1.0 - 0.6 * index / n_slices if taper else 1.0
        volume[:, :, index][
            np.hypot(rows - size / 2, columns - size / 2) < radius * scale
        ] = 3
    return volume


class TopologyTest(unittest.TestCase):
    """The cap winding bug was silent: closed, manifold, and wrong volume."""

    def test_faces_are_a_consistently_oriented_closed_manifold(self):
        from collections import Counter

        faces = lv_topology_faces(8, 32)
        directed = Counter()
        for a, b, c in faces:
            for edge in ((a, b), (b, c), (c, a)):
                directed[edge] += 1
        self.assertEqual([e for e, k in directed.items() if k > 1], [])
        unpaired = [e for e in directed if (e[1], e[0]) not in directed]
        self.assertEqual(unpaired, [])

    def test_face_count_matches_the_topology(self):
        rings, angles = 6, 16
        faces = lv_topology_faces(rings, angles)
        self.assertEqual(len(faces), 2 * (rings - 1) * angles + 2 * angles)

    def test_volume_agrees_with_direct_voxel_counting(self):
        """Wrong cap winding gave a volume 25 percent low and looked plausible.

        The comparison is against voxel counting rather than an idealized
        cylinder, because the two caps are deliberately not symmetric: the valve
        plane truncates the cavity so the basal cap is flat, while the apex
        tapers so its cap is a cone. Voxel counting over-counts partial-volume
        boundary voxels, so the surface should land slightly below it.
        """

        spacing = (1.0, 1.0, 5.0)
        volume = _synthetic_stack(n_slices=9, radius=18.0, taper=False)
        fit = fit_lv_surface(volume, spacing)
        self.assertIsNotNone(fit)
        faces = lv_topology_faces(fit.n_rings, fit.n_angles)
        enclosed = abs(signed_enclosed_volume(fit.vertices, faces))
        counted = (volume == 3).sum() * spacing[0] * spacing[1] * spacing[2]
        self.assertGreater(enclosed / counted, 0.80)
        self.assertLess(enclosed / counted, 1.02)

    def test_the_basal_cap_is_flat_and_the_apical_cap_is_extended(self):
        """Capping the base with a cone loses two thirds of a half slice there."""

        fit = fit_lv_surface(_synthetic_stack(n_slices=9, radius=18.0), (1.0, 1.0, 5.0))
        rings = fit.vertices[: fit.n_rings * fit.n_angles].reshape(
            fit.n_rings, fit.n_angles, 3
        )
        base_cap = fit.vertices[fit.n_rings * fit.n_angles]
        apex_cap = fit.vertices[fit.n_rings * fit.n_angles + 1]
        base_ring_z = rings[0, :, 2].mean()
        apex_ring_z = rings[-1, :, 2].mean()
        # The wider ring keeps a flat cap; the narrower one is extended outward.
        wider_first = np.mean(
            np.linalg.norm(rings[0, :, :2] - rings[0, :, :2].mean(axis=0), axis=1)
        ) >= np.mean(
            np.linalg.norm(rings[-1, :, :2] - rings[-1, :, :2].mean(axis=0), axis=1)
        )
        if wider_first:
            self.assertAlmostEqual(float(base_cap[2]), float(base_ring_z), places=6)
            self.assertGreater(float(apex_cap[2]), float(apex_ring_z))
        else:
            self.assertAlmostEqual(float(apex_cap[2]), float(apex_ring_z), places=6)
            self.assertLess(float(base_cap[2]), float(base_ring_z))


class BaseOrderTest(unittest.TestCase):
    """ACDC stacks are basal first; the radius test can say otherwise."""

    def test_explicit_order_matches_the_radius_test_where_it_is_right(self):
        stack = _synthetic_stack(n_slices=9, radius=18.0)
        tested = fit_lv_surface(stack, (1.0, 1.0, 5.0))
        explicit = fit_lv_surface(stack, (1.0, 1.0, 5.0), base_first=True)
        np.testing.assert_array_equal(tested.vertices, explicit.vertices)

    def test_explicit_order_overrides_a_misleading_radius_test(self):
        # Wider at the last slice, as in an end-systolic frame whose last
        # labelled slice is still large: the radius test would call it basal.
        stack = _synthetic_stack(n_slices=9, radius=18.0)[:, :, ::-1]
        fit = fit_lv_surface(stack, (1.0, 1.0, 5.0), base_first=True)
        ring_count = fit.n_rings * fit.n_angles
        rings = fit.vertices[:ring_count].reshape(fit.n_rings, fit.n_angles, 3)
        base_cap, apex_cap = fit.vertices[ring_count], fit.vertices[ring_count + 1]
        # Flat cap on ring 0, cone half a slice beyond the last ring.
        self.assertAlmostEqual(float(base_cap[2]), float(rings[0, :, 2].mean()), places=6)
        self.assertAlmostEqual(float(apex_cap[2]), float(rings[-1, :, 2].mean()) + 2.5, places=6)
        default = fit_lv_surface(stack, (1.0, 1.0, 5.0))
        self.assertFalse(np.allclose(default.vertices[ring_count:], fit.vertices[ring_count:]))


class FitTest(unittest.TestCase):
    def test_fit_produces_a_closed_surface(self):
        fit = fit_lv_surface(_synthetic_stack(), (1.5, 1.5, 10.0))
        report = surface_topology_report(
            fit.vertices, lv_topology_faces(fit.n_rings, fit.n_angles),
            compute_self_intersections=False,
        )
        self.assertTrue(report.closed)
        self.assertEqual(report.nonmanifold_edges, 0)

    def test_vertex_count_is_fixed_across_inputs(self):
        """Correspondence is the point: a shape basis needs identical ordering."""

        first = fit_lv_surface(_synthetic_stack(n_slices=9), (1.5, 1.5, 10.0))
        second = fit_lv_surface(_synthetic_stack(n_slices=6, radius=12.0), (1.5, 1.5, 10.0))
        self.assertEqual(first.n_vertices, second.n_vertices)
        self.assertEqual(first.n_vertices, first.n_rings * first.n_angles + 2)

    def test_too_few_cavity_slices_returns_none(self):
        volume = np.zeros((32, 32, 6), dtype=np.uint8)
        volume[14:18, 14:18, 2] = 3
        self.assertIsNone(fit_lv_surface(volume, (1.5, 1.5, 10.0)))

    def test_empty_slice_contour_is_none(self):
        self.assertIsNone(
            slice_cavity_contour(np.zeros((32, 32), dtype=bool), 32, (1.0, 1.0))
        )

    def test_contour_has_the_requested_angular_samples(self):
        rows, columns = np.mgrid[0:64, 0:64]
        mask = np.hypot(rows - 32, columns - 32) < 15
        contour = slice_cavity_contour(mask, 48, (1.0, 1.0))
        self.assertEqual(contour.shape, (48, 2))
        radii = np.linalg.norm(contour - contour.mean(axis=0), axis=1)
        self.assertLess(float(np.std(radii) / np.mean(radii)), 0.1)


class ShapeModelTest(unittest.TestCase):
    def _fits(self, n: int = 8):
        from rooted4d.active.acdc import LVFit

        fits = []
        for index in range(n):
            for frame, radius in ((1, 18.0 + index), (12, 13.0 + index)):
                volume = _synthetic_stack(radius=radius)
                fit = fit_lv_surface(
                    volume, (1.5, 1.5, 10.0), patient=f"p{index:03d}", frame=frame
                )
                fits.append(fit)
        return fits

    def test_mean_shape_beats_at_a_physiological_rate(self):
        """Centring the motion basis away left a heart with EF exactly zero."""

        model = build_acdc_shape_model(self._fits(), n_identity=4, n_motion=2)
        zero_id = np.zeros(model.n_identity)
        zero_motion = np.zeros(model.n_motion)
        end_diastolic = abs(
            signed_enclosed_volume(model.shape_at(zero_id, zero_motion, 0.0), model.faces)
        )
        end_systolic = abs(
            signed_enclosed_volume(model.shape_at(zero_id, zero_motion, 1.0), model.faces)
        )
        self.assertGreater(end_diastolic, end_systolic)
        self.assertGreater((end_diastolic - end_systolic) / end_diastolic, 0.05)

    def test_projection_round_trips_a_training_shape(self):
        fits = self._fits()
        model = build_acdc_shape_model(fits, n_identity=6, n_motion=3)
        first = fits[0].vertices
        second = fits[1].vertices
        centre = first.mean(axis=0)
        identity, motion = model.project(first - centre, second - centre)
        reconstructed = model.shape_at(identity, motion, 0.0)
        error = np.linalg.norm(reconstructed - (first - centre), axis=1)
        self.assertLess(float(np.median(error)), 2.0)

    def test_too_few_patients_is_rejected(self):
        with self.assertRaises(ValueError):
            build_acdc_shape_model(self._fits(n=2), n_identity=2, n_motion=1)


if __name__ == "__main__":
    unittest.main()
