"""Connected-component cleanup of a predicted cavity label (E86)."""

import unittest

import numpy as np

from rooted4d.segmentation.cleanup import component_labels, keep_largest_cavity_component, largest_component

CAVITY = 3


def _volume():
    """A cavity column through five slices, with myocardium around it."""

    labels = np.zeros((40, 40, 5), dtype=np.uint8)
    labels[8:24, 8:24, :] = 2
    labels[10:22, 10:22, :] = CAVITY
    return labels


class CleanupTest(unittest.TestCase):
    def test_single_component_is_unchanged(self):
        labels = _volume()
        np.testing.assert_array_equal(keep_largest_cavity_component(labels), labels)
        np.testing.assert_array_equal(keep_largest_cavity_component(labels, per_slice=True), labels)

    def test_a_distant_stray_is_removed_and_nothing_else_changes(self):
        labels = _volume()
        stray = labels.copy()
        stray[35:37, 35:37, 2] = CAVITY  # four voxels far from the cavity, on a slice that holds cavity
        cleaned = keep_largest_cavity_component(stray)
        np.testing.assert_array_equal(cleaned, labels)

    def test_only_the_cavity_label_is_touched(self):
        labels = _volume()
        labels[0:3, 0:3, 0] = 2  # a stray of another label stays
        labels[35:37, 35:37, 2] = CAVITY
        cleaned = keep_largest_cavity_component(labels)
        self.assertTrue((cleaned[0:3, 0:3, 0] == 2).all())
        self.assertFalse((cleaned[35:37, 35:37, 2] == CAVITY).any())

    def test_per_slice_removes_an_in_plane_island_joined_through_another_slice(self):
        labels = _volume()
        labels[26:30, 26:30, 1:3] = CAVITY  # an island on slices 1 and 2
        labels[22:26, 22:26, 2] = CAVITY    # joined to the cavity on slice 2 only
        three_d = keep_largest_cavity_component(labels)
        self.assertTrue((three_d[26:30, 26:30, 1] == CAVITY).all())  # connected in 3D, so kept
        per_slice = keep_largest_cavity_component(labels, per_slice=True)
        self.assertFalse((per_slice[26:30, 26:30, 1] == CAVITY).any())  # separate in slice 1's plane
        self.assertTrue((per_slice[22:26, 22:26, 2] == CAVITY).all())  # still attached in slice 2

    def test_connectivity(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 0] = mask[1, 1] = True
        self.assertEqual(component_labels(mask, full_connectivity=True)[1], [2])
        self.assertEqual(sorted(component_labels(mask, full_connectivity=False)[1]), [1, 1])

    def test_empty_and_ties(self):
        empty = np.zeros((4, 4, 2), dtype=np.uint8)
        np.testing.assert_array_equal(keep_largest_cavity_component(empty), empty)
        mask = np.zeros((5, 5), dtype=bool)
        mask[0, 0] = mask[4, 4] = True
        kept = largest_component(mask)
        self.assertTrue(kept[0, 0] and not kept[4, 4])  # equal sizes: the first in C order

    def test_matches_scipy_when_available(self):
        try:
            from scipy import ndimage
        except ImportError:
            self.skipTest("SciPy is not installed here")
        rng = np.random.default_rng(0)
        for _ in range(5):
            mask = rng.random((12, 12, 4)) < 0.3
            _, sizes = component_labels(mask, full_connectivity=True)
            reference, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
            expected = sorted(int((reference == i).sum()) for i in range(1, n + 1))
            self.assertEqual(sorted(sizes), expected)


if __name__ == "__main__":
    unittest.main()
