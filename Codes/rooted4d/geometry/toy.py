"""Procedural heterogeneous chamber–annulus–leaflet geometry.

This is intentionally simple and exists only to test exact interface algebra and
known-truth synthetic inference variables before clinical templates are ready.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rooted4d.constraints import AttachmentMap


@dataclass(frozen=True)
class ToyComplex:
    global_vertices: np.ndarray
    attachment_map: AttachmentMap
    groups: dict[str, np.ndarray]
    faces: dict[str, np.ndarray]


def make_toy_complex(n_annulus: int = 24, chamber_levels: int = 1, leaflet_rows: int = 1) -> ToyComplex:
    """The procedural chamber--annulus--leaflet complex.

    The reduced instance, the default, has one ring between the annulus and the
    apex and one free-edge row per leaflet. ``chamber_levels`` and
    ``leaflet_rows`` give the interiors more capacity (plan v2, W2.6): that many
    chamber rings between the annulus and the apex, and that many leaflet rows
    beyond the root, the last of which is the free edge. Vertices are laid out as
    annulus, chamber rings from the annulus down, apex, then the leaflet rows from
    the root outward, each row anterior half then posterior half. With the
    defaults this is exactly the reduced instance every earlier result used.
    """

    if n_annulus < 12 or n_annulus % 2:
        raise ValueError("n_annulus must be an even integer of at least 12")
    if chamber_levels < 1 or leaflet_rows < 1:
        raise ValueError("chamber_levels and leaflet_rows must be at least one")

    theta = np.linspace(0.0, 2.0 * np.pi, n_annulus, endpoint=False)
    annulus = np.stack(
        (18.0 * np.cos(theta), 14.0 * np.sin(theta), 2.0 * np.cos(2.0 * theta)),
        axis=-1,
    )
    if chamber_levels == 1:
        levels = [np.stack(
            (23.0 * np.cos(theta), 19.0 * np.sin(theta), np.full_like(theta, -24.0)),
            axis=-1,
        )]
    else:
        # Rings bulging below the annulus and closing toward the apex.
        levels = []
        for k in range(1, chamber_levels + 1):
            depth = k / chamber_levels
            widen = 1.0 + 0.3 * np.sin(np.pi * depth) - 0.45 * depth
            levels.append(np.stack(
                (18.0 * widen * np.cos(theta), 14.0 * widen * np.sin(theta),
                 np.full_like(theta, -48.0 * depth)),
                axis=-1,
            ))
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
    rows_xyz = []
    for row in range(1, leaflet_rows + 1):
        if row == leaflet_rows:
            rows_xyz.append((anterior_free_xyz, posterior_free_xyz))
        else:
            fraction = row / leaflet_rows
            rows_xyz.append((
                (1.0 - fraction) * annulus[anterior_root] + fraction * anterior_free_xyz,
                (1.0 - fraction) * annulus[posterior_root] + fraction * posterior_free_xyz,
            ))

    blocks = [annulus, *levels, apex] + [part for pair in rows_xyz for part in pair]
    global_vertices = np.concatenate(blocks, axis=0)

    annulus_ids = np.arange(n_annulus, dtype=np.int64)
    level_ids = [np.arange((1 + k) * n_annulus, (2 + k) * n_annulus, dtype=np.int64)
                 for k in range(chamber_levels)]
    apex_id = np.array([(1 + chamber_levels) * n_annulus], dtype=np.int64)
    first_row = int(apex_id[0]) + 1
    anterior_rows, posterior_rows = [], []
    for row in range(leaflet_rows):
        start = first_row + row * n_annulus
        anterior_rows.append(np.arange(start, start + half, dtype=np.int64))
        posterior_rows.append(np.arange(start + half, start + n_annulus, dtype=np.int64))
    anterior_free, posterior_free = anterior_rows[-1], posterior_rows[-1]
    lower_ids = level_ids[-1]

    local_to_global = {
        "annulus": annulus_ids,
        "chamber": np.concatenate((annulus_ids, *level_ids, apex_id)),
        "anterior_leaflet": np.concatenate((anterior_root, *anterior_rows)),
        "posterior_leaflet": np.concatenate((posterior_root, *posterior_rows)),
    }
    attachment_map = AttachmentMap(local_to_global, n_global_vertices=len(global_vertices))

    rings = [annulus_ids, *level_ids]
    chamber_faces: list[tuple[int, int, int]] = []
    for index in range(n_annulus):
        following = (index + 1) % n_annulus
        for upper, lower in zip(rings[:-1], rings[1:]):
            chamber_faces.extend(
                [
                    (int(upper[index]), int(lower[index]), int(lower[following])),
                    (int(upper[index]), int(lower[following]), int(upper[following])),
                ]
            )
        chamber_faces.append((int(lower_ids[index]), int(apex_id[0]), int(lower_ids[following])))

    def leaflet_strip(rows_in_order: list[np.ndarray]) -> np.ndarray:
        triangles: list[tuple[int, int, int]] = []
        for root, free in zip(rows_in_order[:-1], rows_in_order[1:]):
            for index in range(len(root) - 1):
                triangles.extend(
                    [
                        (int(root[index]), int(free[index]), int(free[index + 1])),
                        (int(root[index]), int(free[index + 1]), int(root[index + 1])),
                    ]
                )
        return np.asarray(triangles, dtype=np.int64)

    anterior_faces = leaflet_strip([anterior_root, *anterior_rows])
    posterior_faces = leaflet_strip([posterior_root, *posterior_rows])
    faces = {
        "chamber": np.asarray(chamber_faces, dtype=np.int64),
        "anterior_leaflet": anterior_faces,
        "posterior_leaflet": posterior_faces,
        "cmr_surface": np.asarray(chamber_faces, dtype=np.int64),
        "tee_surface": np.concatenate((anterior_faces, posterior_faces), axis=0),
    }
    leaflet_vertices = [part for pair in zip(anterior_rows, posterior_rows) for part in pair]
    groups = {
        "annulus": annulus_ids,
        "chamber_unique": np.concatenate((*level_ids, apex_id)),
        "leaflet_free": np.concatenate((anterior_free, posterior_free)),
        "cmr_visible": np.concatenate((annulus_ids, *level_ids, apex_id)),
        "tee_visible": np.concatenate((annulus_ids, *leaflet_vertices)),
    }
    if chamber_levels > 1 or leaflet_rows > 1:
        # Only on richer instances, so the reduced instance's groups are unchanged.
        groups["leaflet_interior"] = np.concatenate(
            leaflet_vertices[:-2] or [np.empty(0, dtype=np.int64)])
    return ToyComplex(
        global_vertices=global_vertices,
        attachment_map=attachment_map,
        groups=groups,
        faces=faces,
    )


def toy_complex_for_vertices(n_vertices: int, chamber_levels: int = 1, leaflet_rows: int = 1) -> ToyComplex:
    """The procedural complex that has exactly ``n_vertices`` physical vertices.

    The reduced complex has ``3 n + 1`` vertices for an annulus of ``n`` nodes:
    the annulus, the lower ring, the leaflet free edge and the apex; a richer one
    has ``(1 + chamber_levels + leaflet_rows) n + 1``. A vertex count alone is
    ambiguous between the two (241 is 24 nodes richly and 80 reduced), so a richer
    instance's levels and rows must be passed (``sample_topology``). Model
    builders used to hard-code ``n = 24`` and check the count afterwards, which
    made every other resolution fail even when the data were built correctly;
    this derives ``n`` from the data and fails loudly only when no ``n`` fits.
    """

    count = int(n_vertices)
    per_node = 1 + int(chamber_levels) + int(leaflet_rows)
    if (count - 1) % per_node or (count - 1) // per_node < 12:
        raise ValueError(f"no procedural complex has {count} vertices "
                         f"(need {per_node}n + 1, n >= 12)")
    n_annulus = (count - 1) // per_node
    complex_ = make_toy_complex(n_annulus=n_annulus, chamber_levels=chamber_levels,
                                leaflet_rows=leaflet_rows)
    if len(complex_.global_vertices) != count:
        raise ValueError(
            f"a {n_annulus}-node complex has {len(complex_.global_vertices)} "
            f"vertices, not {count}")
    return complex_


#: Sample keys that declare a richer procedural instance (W2.6); absent on the reduced one.
TOPOLOGY_KEYS = ("topology__chamber_levels", "topology__leaflet_rows")


def sample_topology(sample) -> dict[str, int]:
    """The procedural topology a stored sample was generated on.

    ``sample`` is a mapping or an open ``.npz``; a sample without the keys is
    the reduced instance.
    """

    keys = sample.files if hasattr(sample, "files") else sample
    return {
        "chamber_levels": int(sample["topology__chamber_levels"]) if "topology__chamber_levels" in keys else 1,
        "leaflet_rows": int(sample["topology__leaflet_rows"]) if "topology__leaflet_rows" in keys else 1,
    }
