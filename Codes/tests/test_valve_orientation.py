"""Valve orientation in composite construction.

Every benchmark built before ``orient_valves`` existed carries its valves upside
down relative to the chamber, because the two donor pools order their rings with
opposite handedness (Docs/COMPOSITE_ORIENTATION_FINDING_2026-09-28.md). The
option must fix that without touching anything built without it.
"""

import unittest

import numpy as np

from rooted4d.synthetic.config import DonorConfig, SemisyntheticConfig
from rooted4d.synthetic.dataset_builder import _portable_config_record
from rooted4d.synthetic.donors import Donor, combine_donors, make_toy_complex

N = 24


def _donor(kind: str, vertices: np.ndarray) -> Donor:
    return Donor(f"{kind}-0", f"person-{kind}", kind, vertices, "test", None)


def _free_edge_height(vertices: np.ndarray, groups: dict) -> float:
    """Mean free-edge height along the axis from the annulus centroid to the apex."""

    ring = vertices[groups["annulus"]].mean(axis=0)
    apex = vertices[np.setdiff1d(groups["cmr_visible"], groups["annulus"])[-1]]
    axis = (apex - ring) / np.linalg.norm(apex - ring)
    return float(((vertices[groups["leaflet_free"]] - ring) @ axis).mean())


class ValveOrientationTest(unittest.TestCase):
    def setUp(self):
        self.fixture = make_toy_complex(n_annulus=N)
        self.groups = self.fixture.groups
        self.heart = _donor("whole_heart", self.fixture.global_vertices.copy())
        # A valve like MVSeg2023's: the same anatomy with its ring and free edge
        # indexed in the opposite direction, keeping each leaflet on its arc.
        reverse = (N // 2 - 1 - np.arange(N)) % N
        flipped = self.fixture.global_vertices.copy()
        annulus, free = self.groups["annulus"], self.groups["leaflet_free"]
        flipped[annulus] = self.fixture.global_vertices[annulus[reverse]]
        flipped[free] = self.fixture.global_vertices[free[reverse]]
        self.reversed_valve = _donor("valve", flipped)
        self.matching_valve = _donor("valve", self.fixture.global_vertices.copy())

    def test_the_template_hangs_its_leaflets_into_the_ventricle(self):
        self.assertGreater(_free_edge_height(self.fixture.global_vertices, self.groups), 0.0)

    def test_opposite_handedness_turns_the_valve_over_unless_oriented(self):
        as_built = combine_donors(self.heart, self.reversed_valve, N)
        oriented = combine_donors(self.heart, self.reversed_valve, N, orient_valve=True)
        self.assertLess(_free_edge_height(as_built, self.groups), 0.0)
        self.assertGreater(_free_edge_height(oriented, self.groups), 0.0)
        # Orienting recovers the template exactly: same anatomy, re-indexed back.
        np.testing.assert_allclose(oriented, self.fixture.global_vertices, atol=1e-9)

    def test_matching_handedness_is_left_alone(self):
        np.testing.assert_array_equal(
            combine_donors(self.heart, self.matching_valve, N, orient_valve=True),
            combine_donors(self.heart, self.matching_valve, N))

    def test_the_default_record_is_unchanged(self):
        config = SemisyntheticConfig(donors=DonorConfig())
        self.assertNotIn("orient_valves", _portable_config_record(config)["donors"])
        on = SemisyntheticConfig(donors=DonorConfig(orient_valves=True))
        self.assertTrue(_portable_config_record(on)["donors"]["orient_valves"])


if __name__ == "__main__":
    unittest.main()
