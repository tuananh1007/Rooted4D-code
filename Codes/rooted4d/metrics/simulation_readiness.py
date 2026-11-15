r"""Is the reconstructed geometry usable by a downstream mesher or solver?

Section 7.1 asks whether the result is simulation-ready, and section 7.4 lists
what to measure. This is the question on which exact coupling should earn its
keep: a soft penalty reaches a small interface gap, but a mesh generator does not
care how small a crack is in the abstract, only whether it falls below the
tolerance at which it welds coincident vertices.

The check performed here is the one a meshing pipeline actually performs. Weld
vertices within a tolerance, then ask whether the result is a single conforming
manifold. If duplicated interface occurrences are farther apart than the
tolerance they survive as distinct vertices, the interface becomes a crack, and
the chamber stops being watertight.

Volumetric element quality is deliberately absent. The procedural template is a
surface complex with no tetrahedra, and section 8 is explicit that a surface-only
model cannot support Jacobian claims. Those metrics belong with the production
template, which is still an open gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rooted4d.metrics.topology import edge_incidence_counts

#: Weld tolerances a mesh pipeline plausibly uses, relative to bounding-box
#: diagonal. Gmsh's geometric tolerance and common repair tools sit in this band.
DEFAULT_WELD_TOLERANCES = (1e-8, 1e-6, 1e-4)


def weld_vertices(
    vertices: np.ndarray, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Merge vertices closer than ``tolerance``; return unique points and a relabel map.

    Grid snapping is what mesh tools actually do, and it is what makes the test
    honest: two occurrences separated by more than the tolerance stay distinct no
    matter how physically close they look.
    """

    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("vertices must have shape (vertices, 3)")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    keys = np.round(points / tolerance).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return points[np.sort(first)], _relabel(first, inverse)


def _relabel(first: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[inverse]


def connected_components(n_vertices: int, faces: np.ndarray) -> int:
    """Number of connected components of the face graph, by union-find."""

    parent = list(range(n_vertices))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for triangle in np.asarray(faces, dtype=np.int64):
        roots = [find(int(v)) for v in triangle]
        for other in roots[1:]:
            if other != roots[0]:
                parent[other] = roots[0]
    used = np.unique(np.asarray(faces, dtype=np.int64))
    return len({find(int(v)) for v in used}) if len(used) else 0


@dataclass(frozen=True)
class SimulationReadinessReport:
    """Whether one assembled geometry would survive a meshing pipeline."""

    weld_tolerance_mm: float
    n_input_vertices: int
    n_welded_vertices: int
    n_expected_vertices: int
    n_faces: int
    n_components: int
    nonmanifold_edges: int
    degenerate_faces: int
    max_interface_gap_mm: float

    @property
    def interface_welded(self) -> bool:
        """Did every duplicated occurrence merge back to one physical vertex?"""

        return self.n_welded_vertices == self.n_expected_vertices

    @property
    def parts_still_joined(self) -> bool:
        """One component means the leaflets are still attached to the chamber."""

        return self.n_components == 1

    @property
    def simulation_ready(self) -> bool:
        return (
            self.interface_welded
            and self.parts_still_joined
            and self.nonmanifold_edges == 0
            and self.degenerate_faces == 0
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "weld_tolerance_mm": self.weld_tolerance_mm,
            "n_input_vertices": self.n_input_vertices,
            "n_welded_vertices": self.n_welded_vertices,
            "n_expected_vertices": self.n_expected_vertices,
            "n_faces": self.n_faces,
            "n_components": self.n_components,
            "nonmanifold_edges": self.nonmanifold_edges,
            "degenerate_faces": self.degenerate_faces,
            "max_interface_gap_mm": self.max_interface_gap_mm,
            "interface_welded": self.interface_welded,
            "parts_still_joined": self.parts_still_joined,
            "simulation_ready": self.simulation_ready,
        }


def assess_simulation_readiness(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    weld_tolerance_mm: float,
    n_expected_vertices: int,
    max_interface_gap_mm: float = 0.0,
    minimum_area_mm2: float = 1e-12,
) -> SimulationReadinessReport:
    """Weld at the given tolerance, then report whether the complex held together."""

    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    welded, relabel = weld_vertices(points, weld_tolerance_mm)
    remapped = relabel[triangles]

    collapsed = (
        (remapped[:, 0] == remapped[:, 1])
        | (remapped[:, 1] == remapped[:, 2])
        | (remapped[:, 2] == remapped[:, 0])
    )
    degenerate = int(np.count_nonzero(collapsed))
    kept = remapped[~collapsed]
    xyz = welded[kept]
    areas = 0.5 * np.linalg.norm(
        np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
    )
    degenerate += int(np.count_nonzero(areas < minimum_area_mm2))

    _, counts = edge_incidence_counts(kept)
    return SimulationReadinessReport(
        weld_tolerance_mm=float(weld_tolerance_mm),
        n_input_vertices=len(points),
        n_welded_vertices=len(welded),
        n_expected_vertices=int(n_expected_vertices),
        n_faces=len(kept),
        n_components=connected_components(len(welded), kept),
        nonmanifold_edges=int(np.count_nonzero(counts > 2)),
        degenerate_faces=degenerate,
        max_interface_gap_mm=float(max_interface_gap_mm),
    )


def interface_welds_at(gap_mm: float, tolerance_mm: float) -> bool:
    """Would duplicated interface occurrences merge at this tolerance?

    This is the whole practical question. An exact parameterization answers yes
    at every tolerance because the gap is zero; a penalty answers yes only above
    a weight the user must know in advance.
    """

    if tolerance_mm <= 0.0:
        raise ValueError("tolerance_mm must be positive")
    return bool(gap_mm <= tolerance_mm)
