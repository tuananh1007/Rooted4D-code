"""Reference-metric membrane and hinge-bending regularizers for leaflets.

These energies apply to Rooted4D's explicit anterior and posterior leaflet
manifolds.  They regularize a fitted surface relative to a declared rest
configuration; they do not claim material-property or stress identification.
Continuous collision handling remains a separate production gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _surface_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("vertices must have finite shape (vertices, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise ValueError("faces must have non-empty shape (faces, 3)")
    if triangles.min() < 0 or triangles.max() >= len(points):
        raise ValueError("faces contain an out-of-range vertex index")
    if np.any(
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 2] == triangles[:, 0])
    ):
        raise ValueError("faces must contain three distinct vertex indices")
    return points, triangles


def _triangle_crosses(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    return np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])


def _signed_dihedral_angles(
    vertices: np.ndarray,
    faces: np.ndarray,
    hinge_edges: np.ndarray,
    hinge_faces: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    crosses = _triangle_crosses(vertices, faces)
    lengths = np.linalg.norm(crosses, axis=1)
    if np.any(lengths <= epsilon):
        raise ValueError("surface contains a degenerate triangle")
    normals = crosses / lengths[:, None]
    normal_0 = normals[hinge_faces[:, 0]]
    normal_1 = normals[hinge_faces[:, 1]]
    edge_vectors = vertices[hinge_edges[:, 1]] - vertices[hinge_edges[:, 0]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    if np.any(edge_lengths <= epsilon):
        raise ValueError("surface contains a collapsed hinge edge")
    edge_directions = edge_vectors / edge_lengths[:, None]
    sine = np.einsum(
        "ij,ij->i",
        edge_directions,
        np.cross(normal_0, normal_1),
    )
    cosine = np.clip(np.einsum("ij,ij->i", normal_0, normal_1), -1.0, 1.0)
    return np.arctan2(sine, cosine)


@dataclass(frozen=True)
class ShellReference:
    """Precomputed rest geometry for one fixed-connectivity leaflet sheet."""

    n_vertices: int
    faces: np.ndarray
    inverse_rest_coordinates: np.ndarray
    rest_areas: np.ndarray
    hinge_edges: np.ndarray
    hinge_faces: np.ndarray
    rest_dihedral_angles: np.ndarray
    hinge_weights: np.ndarray


def build_shell_reference(
    rest_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> ShellReference:
    """Build rest metrics and manifold hinge topology for a leaflet surface."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    points, triangles = _surface_arrays(rest_vertices, faces)
    xyz = points[triangles]
    first = xyz[:, 1] - xyz[:, 0]
    second = xyz[:, 2] - xyz[:, 0]
    first_length = np.linalg.norm(first, axis=1)
    crosses = np.cross(first, second)
    doubled_area = np.linalg.norm(crosses, axis=1)
    if np.any(first_length <= epsilon) or np.any(doubled_area <= 2.0 * epsilon):
        raise ValueError("rest surface contains a degenerate triangle")

    basis_first = first / first_length[:, None]
    normals = crosses / doubled_area[:, None]
    basis_second = np.cross(normals, basis_first)
    rest_coordinates = np.zeros((len(triangles), 2, 2), dtype=np.float64)
    rest_coordinates[:, 0, 0] = first_length
    rest_coordinates[:, 0, 1] = np.einsum("ij,ij->i", second, basis_first)
    rest_coordinates[:, 1, 1] = np.einsum("ij,ij->i", second, basis_second)
    inverse_rest = np.linalg.inv(rest_coordinates)
    rest_areas = 0.5 * doubled_area

    raw_edges = np.concatenate(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ],
        axis=0,
    )
    raw_face_ids = np.tile(np.arange(len(triangles), dtype=np.int64), 3)
    keys = np.sort(raw_edges, axis=1)
    unique_edges, inverse, counts = np.unique(
        keys,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    if np.any(counts > 2):
        raise ValueError("leaflet surface contains a non-manifold edge")
    interior_edge_ids = np.flatnonzero(counts == 2)
    hinge_edges = unique_edges[interior_edge_ids]
    hinge_faces = np.empty((len(interior_edge_ids), 2), dtype=np.int64)
    for output_index, edge_id in enumerate(interior_edge_ids):
        incident = np.sort(raw_face_ids[inverse == edge_id])
        hinge_faces[output_index] = incident

    if len(hinge_edges):
        rest_dihedral = _signed_dihedral_angles(
            points,
            triangles,
            hinge_edges,
            hinge_faces,
            epsilon=epsilon,
        )
        edge_length = np.linalg.norm(
            points[hinge_edges[:, 1]] - points[hinge_edges[:, 0]],
            axis=1,
        )
        adjacent_area = (
            rest_areas[hinge_faces[:, 0]] + rest_areas[hinge_faces[:, 1]]
        )
        hinge_weights = edge_length**2 / np.maximum(adjacent_area, epsilon)
    else:
        rest_dihedral = np.empty(0, dtype=np.float64)
        hinge_weights = np.empty(0, dtype=np.float64)

    return ShellReference(
        n_vertices=len(points),
        faces=triangles.copy(),
        inverse_rest_coordinates=inverse_rest,
        rest_areas=rest_areas,
        hinge_edges=hinge_edges,
        hinge_faces=hinge_faces,
        rest_dihedral_angles=rest_dihedral,
        hinge_weights=hinge_weights,
    )


def shell_membrane_energy(
    vertices: np.ndarray,
    reference: ShellReference,
    *,
    lame_lambda: float = 1.0,
    lame_mu: float = 1.0,
) -> float:
    r"""St. Venant--Kirchhoff membrane regularizer on each rest triangle.

    The deformation gradient maps the two-dimensional rest metric into 3D.
    This is stronger and more explicit than an edge-length-only penalty.
    """

    if lame_lambda < 0.0 or lame_mu <= 0.0:
        raise ValueError("lame_lambda must be non-negative and lame_mu positive")
    points = np.asarray(vertices, dtype=np.float64)
    if points.shape != (reference.n_vertices, 3) or not np.all(np.isfinite(points)):
        raise ValueError("vertices do not match the shell reference")
    xyz = points[reference.faces]
    deformed_edges = np.stack(
        [xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]],
        axis=-1,
    )
    deformation = np.matmul(deformed_edges, reference.inverse_rest_coordinates)
    metric = np.einsum("fki,fkj->fij", deformation, deformation)
    strain = 0.5 * (metric - np.eye(2, dtype=np.float64)[None, :, :])
    trace = np.trace(strain, axis1=1, axis2=2)
    squared_norm = np.sum(strain * strain, axis=(1, 2))
    density = lame_mu * squared_norm + 0.5 * lame_lambda * trace**2
    return float(np.sum(reference.rest_areas * density))


def shell_bending_energy(
    vertices: np.ndarray,
    reference: ShellReference,
    *,
    stiffness: float = 1.0,
    epsilon: float = 1e-12,
) -> float:
    """Wrapped signed-dihedral bending regularizer around the rest sheet."""

    if stiffness < 0.0:
        raise ValueError("stiffness must be non-negative")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    points = np.asarray(vertices, dtype=np.float64)
    if points.shape != (reference.n_vertices, 3) or not np.all(np.isfinite(points)):
        raise ValueError("vertices do not match the shell reference")
    if not len(reference.hinge_edges) or stiffness == 0.0:
        return 0.0
    angles = _signed_dihedral_angles(
        points,
        reference.faces,
        reference.hinge_edges,
        reference.hinge_faces,
        epsilon=epsilon,
    )
    raw_difference = angles - reference.rest_dihedral_angles
    difference = np.arctan2(np.sin(raw_difference), np.cos(raw_difference))
    return float(0.5 * stiffness * np.sum(reference.hinge_weights * difference**2))
