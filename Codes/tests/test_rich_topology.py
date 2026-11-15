"""The richer procedural instance (plan v2, W2.6), and the reduced one left untouched."""

import unittest

import numpy as np

from rooted4d.constraints import AttachmentMap
from rooted4d.geometry import make_toy_complex, sample_topology, toy_complex_for_vertices
from rooted4d.geometry.toy import ToyComplex


class _Reference:
    """``make_toy_complex`` as it was before the levels and rows existed, verbatim."""

    @staticmethod
    def reference_toy_complex(n_annulus: int = 24) -> ToyComplex:
        if n_annulus < 12 or n_annulus % 2:
            raise ValueError("n_annulus must be an even integer of at least 12")

        theta = np.linspace(0.0, 2.0 * np.pi, n_annulus, endpoint=False)
        annulus = np.stack(
            (18.0 * np.cos(theta), 14.0 * np.sin(theta), 2.0 * np.cos(2.0 * theta)),
            axis=-1,
        )
        lower_ring = np.stack(
            (23.0 * np.cos(theta), 19.0 * np.sin(theta), np.full_like(theta, -24.0)),
            axis=-1,
        )
        apex = np.array([[0.0, 0.0, -56.0]])

        half = n_annulus // 2
        anterior_root = np.arange(0, half, dtype=np.int64)
        posterior_root = np.arange(half, n_annulus, dtype=np.int64)

        anterior_free_xyz = annulus[anterior_root].copy()
        anterior_free_xyz[:, :2] *= 0.48
        anterior_free_xyz[:, 2] -= 8.0
        posterior_free_xyz = annulus[posterior_root].copy()
        posterior_free_xyz[:, :2] *= 0.52
        posterior_free_xyz[:, 2] -= 7.0

        blocks = [annulus, lower_ring, apex, anterior_free_xyz, posterior_free_xyz]
        global_vertices = np.concatenate(blocks, axis=0)

        annulus_ids = np.arange(n_annulus, dtype=np.int64)
        lower_ids = np.arange(n_annulus, 2 * n_annulus, dtype=np.int64)
        apex_id = np.array([2 * n_annulus], dtype=np.int64)
        anterior_free = np.arange(2 * n_annulus + 1, 2 * n_annulus + 1 + half, dtype=np.int64)
        posterior_free = np.arange(anterior_free[-1] + 1, anterior_free[-1] + 1 + half, dtype=np.int64)

        local_to_global = {
            "annulus": annulus_ids,
            "chamber": np.concatenate((annulus_ids, lower_ids, apex_id)),
            "anterior_leaflet": np.concatenate((anterior_root, anterior_free)),
            "posterior_leaflet": np.concatenate((posterior_root, posterior_free)),
        }
        attachment_map = AttachmentMap(local_to_global, n_global_vertices=len(global_vertices))

        chamber_faces: list[tuple[int, int, int]] = []
        for index in range(n_annulus):
            following = (index + 1) % n_annulus
            chamber_faces.extend(
                [
                    (int(annulus_ids[index]), int(lower_ids[index]), int(lower_ids[following])),
                    (int(annulus_ids[index]), int(lower_ids[following]), int(annulus_ids[following])),
                    (int(lower_ids[index]), int(apex_id[0]), int(lower_ids[following])),
                ]
            )

        def leaflet_strip(root: np.ndarray, free: np.ndarray) -> np.ndarray:
            triangles: list[tuple[int, int, int]] = []
            for index in range(len(root) - 1):
                triangles.extend(
                    [
                        (int(root[index]), int(free[index]), int(free[index + 1])),
                        (int(root[index]), int(free[index + 1]), int(root[index + 1])),
                    ]
                )
            return np.asarray(triangles, dtype=np.int64)

        anterior_faces = leaflet_strip(anterior_root, anterior_free)
        posterior_faces = leaflet_strip(posterior_root, posterior_free)
        faces = {
            "chamber": np.asarray(chamber_faces, dtype=np.int64),
            "anterior_leaflet": anterior_faces,
            "posterior_leaflet": posterior_faces,
            "cmr_surface": np.asarray(chamber_faces, dtype=np.int64),
            "tee_surface": np.concatenate((anterior_faces, posterior_faces), axis=0),
        }
        groups = {
            "annulus": annulus_ids,
            "chamber_unique": np.concatenate((lower_ids, apex_id)),
            "leaflet_free": np.concatenate((anterior_free, posterior_free)),
            "cmr_visible": np.concatenate((annulus_ids, lower_ids, apex_id)),
            "tee_visible": np.concatenate((annulus_ids, anterior_free, posterior_free)),
        }
        return ToyComplex(
            global_vertices=global_vertices,
            attachment_map=attachment_map,
            groups=groups,
            faces=faces,
        )


class ReducedInstanceUnchangedTest(unittest.TestCase):
    def test_default_is_the_reduced_instance_bit_for_bit(self):
        for n in (12, 24, 48, 96):
            new, old = make_toy_complex(n), _Reference.reference_toy_complex(n)
            np.testing.assert_array_equal(new.global_vertices, old.global_vertices)
            self.assertEqual(sorted(new.groups), sorted(old.groups))
            for key in old.groups:
                np.testing.assert_array_equal(new.groups[key], old.groups[key])
            self.assertEqual(sorted(new.faces), sorted(old.faces))
            for key in old.faces:
                np.testing.assert_array_equal(new.faces[key], old.faces[key])
            for name, rows in old.attachment_map.local_to_global.items():
                np.testing.assert_array_equal(new.attachment_map.local_to_global[name], rows)

    def test_vertex_count_lookup_is_unchanged(self):
        self.assertEqual(len(toy_complex_for_vertices(289).groups["annulus"]), 96)


class RicherInstanceTest(unittest.TestCase):
    def setUp(self):
        self.n, self.levels, self.rows = 24, 6, 3
        self.complex = make_toy_complex(self.n, chamber_levels=self.levels, leaflet_rows=self.rows)

    def test_counts(self):
        n, half = self.n, self.n // 2
        groups = self.complex.groups
        self.assertEqual(len(self.complex.global_vertices), (1 + self.levels + self.rows) * n + 1)
        self.assertEqual(groups["chamber_unique"].size, self.levels * n + 1)
        self.assertEqual(groups["leaflet_free"].size, n)
        self.assertEqual(groups["leaflet_interior"].size, (self.rows - 1) * n)
        self.assertEqual(groups["tee_visible"].size, (1 + self.rows) * n)
        local = self.complex.attachment_map.local_to_global
        self.assertEqual(local["anterior_leaflet"].size, half * (1 + self.rows))
        self.assertEqual(local["chamber"].size, (1 + self.levels) * n + 1)

    def test_free_edge_is_the_last_row_and_the_layout_ends_with_it(self):
        total = len(self.complex.global_vertices)
        np.testing.assert_array_equal(np.sort(self.complex.groups["leaflet_free"]),
                                      np.arange(total - self.n, total))

    def test_attachment_is_shared_by_construction(self):
        local = self.complex.attachment_map.local_to_global
        annulus = set(self.complex.groups["annulus"].tolist())
        roots = set(local["anterior_leaflet"][: self.n // 2].tolist()) | set(local["posterior_leaflet"][: self.n // 2].tolist())
        self.assertEqual(roots, annulus)
        self.assertTrue(annulus <= set(local["chamber"].tolist()))

    def test_faces_index_their_components(self):
        n_vertices = len(self.complex.global_vertices)
        for name, faces in self.complex.faces.items():
            self.assertTrue(np.all((faces >= 0) & (faces < n_vertices)), name)
        self.assertEqual(len(self.complex.faces["chamber"]), self.n * (2 * self.levels + 1))
        self.assertEqual(len(self.complex.faces["anterior_leaflet"]), 2 * self.rows * (self.n // 2 - 1))

    def test_the_vertex_count_is_ambiguous_without_the_topology(self):
        count = len(self.complex.global_vertices)
        self.assertEqual(count, 241)
        self.assertEqual(len(toy_complex_for_vertices(count).groups["annulus"]), 80)
        rich = toy_complex_for_vertices(count, chamber_levels=self.levels, leaflet_rows=self.rows)
        np.testing.assert_array_equal(rich.global_vertices, self.complex.global_vertices)

    def test_sample_topology(self):
        self.assertEqual(sample_topology({}), {"chamber_levels": 1, "leaflet_rows": 1})
        stored = {"topology__chamber_levels": np.array(6), "topology__leaflet_rows": np.array(3)}
        self.assertEqual(sample_topology(stored), {"chamber_levels": 6, "leaflet_rows": 3})


class RicherCompositeTest(unittest.TestCase):
    """Composites and generated samples on the richer instance."""

    def setUp(self):
        from rooted4d.synthetic.donors import Donor

        self.n, self.levels, self.rows = 24, 6, 3
        self.complex = make_toy_complex(self.n, chamber_levels=self.levels, leaflet_rows=self.rows)
        base = self.complex.global_vertices
        rng = np.random.default_rng(1)
        self.heart = Donor("h", "hp", "whole_heart", base + rng.normal(0, 0.5, base.shape), "procedural", "0" * 64)
        self.valve = Donor("v", "vp", "valve", base * 1.1 + rng.normal(0, 0.5, base.shape), "procedural", "0" * 64)

    def combine(self, valve, orient=False):
        from rooted4d.synthetic.donors import combine_donors

        return combine_donors(self.heart, valve, self.n, orient_valve=orient,
                              chamber_levels=self.levels, leaflet_rows=self.rows)

    def test_every_leaflet_row_is_carried_and_the_chamber_kept(self):
        combined = self.combine(self.valve)
        groups = self.complex.groups
        np.testing.assert_array_equal(combined[groups["cmr_visible"]], self.heart.vertices[groups["cmr_visible"]])
        rows = np.setdiff1d(groups["tee_visible"], groups["annulus"])
        self.assertFalse(np.allclose(combined[rows], self.heart.vertices[rows]))

    def test_interior_rows_are_reindexed_with_the_free_edge(self):
        """Orienting a valve equals carrying it after re-indexing its ring and every row by hand."""

        from rooted4d.synthetic.donors import Donor, _ring_handedness

        n = self.n
        order = (n // 2 - 1 - np.arange(n)) % n
        groups = self.complex.groups
        rows = np.setdiff1d(groups["tee_visible"], groups["annulus"]).reshape(self.rows, n)

        def reindexed(vertices):
            out = vertices.copy()
            out[groups["annulus"]] = vertices[groups["annulus"][order]]
            for row in rows:
                out[row] = vertices[row[order]]
            return out

        flipped = Donor("f", "fp", "valve", reindexed(self.valve.vertices), "procedural", "0" * 64)
        chamber = np.setdiff1d(groups["cmr_visible"], groups["annulus"])
        self.assertNotEqual(_ring_handedness(self.heart.vertices, groups["annulus"], chamber),
                            _ring_handedness(flipped.vertices, groups["annulus"], groups["leaflet_free"]))
        by_hand = Donor("b", "bp", "valve", reindexed(flipped.vertices), "procedural", "0" * 64)
        np.testing.assert_allclose(self.combine(flipped, orient=True), self.combine(by_hand), atol=1e-12)
        self.assertFalse(np.allclose(self.combine(flipped)[rows], self.combine(flipped, orient=True)[rows]))

    def test_generated_samples_declare_their_topology(self):
        from rooted4d.synthetic.benchmark import generate_case

        case = generate_case(seed=3, n_annulus=self.n, template_vertices=self.complex.global_vertices,
                             chamber_levels=self.levels, leaflet_rows=self.rows)
        arrays = case.arrays()
        self.assertEqual(sample_topology(arrays), {"chamber_levels": self.levels, "leaflet_rows": self.rows})
        self.assertTrue(set(case.tee_indices.tolist()) <= set(self.complex.groups["tee_visible"].tolist()))
        interior = self.complex.groups["leaflet_interior"]
        self.assertTrue(np.all(case.motion_coefficients[2, interior, 2] < 0.0))
        reduced = generate_case(seed=3).arrays()
        self.assertNotIn("topology__chamber_levels", reduced)


if __name__ == "__main__":
    unittest.main()
