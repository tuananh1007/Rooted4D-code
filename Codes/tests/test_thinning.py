"""Thinning a problem's visible evidence (E113) leaves its held-out evidence alone."""

import unittest

import numpy as np

from rooted4d.experiments.benchmark_eval import thin_observations
from rooted4d.inference.joint import Observations


def _observations(cmr_frames=16, tee_frames=24, anchors=False):
    rng = np.random.default_rng(0)
    cmr_vertices = np.arange(10)
    tee_vertices = np.arange(5, 15)
    extra = {}
    if anchors:
        extra = {"tee_anchor_vertices": np.array([5, 6]),
                 "tee_anchor_points": rng.normal(size=(tee_frames, 2, 3))}
    return Observations(
        cmr_times=np.linspace(0.0, 1.0, cmr_frames, endpoint=False),
        cmr_vertices=cmr_vertices,
        cmr_points=rng.normal(size=(cmr_frames, len(cmr_vertices), 3)),
        tee_times=np.linspace(0.0, 1.0, tee_frames, endpoint=False),
        tee_vertices=tee_vertices,
        tee_points=rng.normal(size=(tee_frames, len(tee_vertices), 3)),
        heldout_cmr_times=np.array([0.1, 0.7]),
        heldout_cmr_vertices=cmr_vertices,
        heldout_cmr_points=rng.normal(size=(2, len(cmr_vertices), 3)),
        heldout_tee_times=np.linspace(0.0, 1.0, tee_frames, endpoint=False),
        heldout_tee_vertices=np.array([7, 9]),
        heldout_tee_points=rng.normal(size=(tee_frames, 2, 3)),
        **extra,
    )


class ThinningTest(unittest.TestCase):
    def test_factors_of_one_change_nothing(self):
        full = _observations()
        thinned = thin_observations(full, cmr_keep_every=1, tee_keep_every=1)
        np.testing.assert_array_equal(thinned.cmr_points, full.cmr_points)
        np.testing.assert_array_equal(thinned.tee_points, full.tee_points)

    def test_every_kth_frame_is_kept_and_the_heldout_is_untouched(self):
        full = _observations(anchors=True)
        thinned = thin_observations(full, cmr_keep_every=4, tee_keep_every=8)
        self.assertEqual(len(thinned.cmr_times), 4)
        self.assertEqual(len(thinned.tee_times), 3)
        np.testing.assert_array_equal(thinned.cmr_times, full.cmr_times[::4])
        np.testing.assert_array_equal(thinned.cmr_points, full.cmr_points[::4])
        np.testing.assert_array_equal(thinned.tee_points, full.tee_points[::8])
        np.testing.assert_array_equal(thinned.tee_anchor_points, full.tee_anchor_points[::8])
        for name in ("heldout_cmr_times", "heldout_cmr_points", "heldout_tee_times",
                     "heldout_tee_vertices", "heldout_tee_points", "cmr_vertices", "tee_vertices"):
            np.testing.assert_array_equal(getattr(thinned, name), getattr(full, name))

    def test_a_zero_factor_is_refused(self):
        with self.assertRaises(ValueError):
            thin_observations(_observations(), cmr_keep_every=0)


if __name__ == "__main__":
    unittest.main()

    def test_the_added_three_frame_level_keeps_the_first_seventh_and_thirteenth(self):
        full = _observations()
        thinned = thin_observations(full, cmr_keep_every=6)
        np.testing.assert_array_equal(thinned.cmr_times, full.cmr_times[[0, 6, 12]])
        self.assertEqual(len(thinned.tee_times), 24)
