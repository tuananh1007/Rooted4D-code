r"""A NumPy reference deformation solver that actually uses the validity barrier.

``tetrahedron_jacobian_barrier`` has existed as a reference implementation with
unit tests, but a caller search finds it invoked from nowhere else: it was a term
in no objective. That left the mandated ``no_collision_or_jacobian_barriers``
ablation with nothing to ablate, and left the claim that the project guards
element validity resting on an unused function.

This module supplies the host. It fits vertex positions to positional targets
subject to the barrier, which is the smallest problem in which the barrier can
either earn its place or fail to. It is deliberately a reference rather than the
production optimizer: NumPy only, analytic gradients, backtracking line search,
and no collision handling beyond what the barrier itself provides. The PyTorch
port named in the roadmap is expected to replace it, and this is the thing that
port should be checked against.

Gradients are analytic rather than finite-differenced. For a tetrahedron with
vertices :math:`a,b,c,d`,

.. math::
   V=\tfrac{1}{6}\det[b-a,\;c-a,\;d-a],

so :math:`\partial V/\partial b=\tfrac16 (c-a)\times(d-a)` and cyclically, with
:math:`\partial V/\partial a` the negated sum of the other three. The barrier is
a function of the ratio :math:`r=V/V_{\mathrm{ref}}`, so its gradient follows by
the chain rule. Finite differences over a few hundred vertices would have made
the ablation slow enough to shrink, and the closed form is short.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tetrahedra import signed_tetrahedron_volumes

__all__ = [
    "DeformationResult",
    "barrier_value_and_gradient",
    "boundary_faces",
    "collision_value_and_gradient",
    "elastic_value_and_gradient",
    "fit_deformation",
    "lattice_mesh",
    "tetrahedron_edges",
]


def lattice_mesh(shape: tuple[int, int, int], spacing: float = 1.0):
    """A structured box lattice split into tetrahedra, five per cube.

    A fixture, not anatomy. The barrier is a statement about element validity,
    which does not care whether the elements came from a ventricle, and a
    lattice makes it possible to drive a known number of elements toward
    inversion on purpose.
    """

    nx, ny, nz = (int(v) for v in shape)
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError("shape must be at least 2 in every direction")
    grid = np.stack(
        np.meshgrid(
            np.arange(nx) * spacing,
            np.arange(ny) * spacing,
            np.arange(nz) * spacing,
            indexing="ij",
        ),
        axis=-1,
    )
    vertices = grid.reshape(-1, 3).astype(np.float64)

    def index(i, j, k):
        return (i * ny + j) * nz + k

    cells = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                corner = [
                    index(i, j, k),
                    index(i + 1, j, k),
                    index(i + 1, j + 1, k),
                    index(i, j + 1, k),
                    index(i, j, k + 1),
                    index(i + 1, j, k + 1),
                    index(i + 1, j + 1, k + 1),
                    index(i, j + 1, k + 1),
                ]
                v0, v1, v2, v3, v4, v5, v6, v7 = corner
                # Alternating the split keeps shared faces conforming between
                # neighbouring cubes; a fixed split leaves cracks.
                if (i + j + k) % 2 == 0:
                    cells.extend([
                        (v0, v1, v3, v4), (v1, v2, v3, v6),
                        (v1, v4, v5, v6), (v3, v4, v6, v7),
                        (v1, v3, v4, v6),
                    ])
                else:
                    cells.extend([
                        (v0, v1, v2, v5), (v0, v2, v3, v7),
                        (v0, v4, v5, v7), (v2, v5, v6, v7),
                        (v0, v2, v5, v7),
                    ])
    tetrahedra = np.asarray(cells, dtype=np.int64)
    volumes = signed_tetrahedron_volumes(vertices, tetrahedra)
    flip = volumes < 0.0
    if np.any(flip):
        tetrahedra[flip] = tetrahedra[flip][:, [0, 1, 3, 2]]
    return vertices, tetrahedra


def boundary_faces(tetrahedra: np.ndarray) -> np.ndarray:
    """Triangles belonging to exactly one tetrahedron."""

    cells = np.asarray(tetrahedra, dtype=np.int64)
    corners = np.concatenate(
        [
            cells[:, [0, 2, 1]],
            cells[:, [0, 1, 3]],
            cells[:, [0, 3, 2]],
            cells[:, [1, 2, 3]],
        ],
        axis=0,
    )
    keys = np.sort(corners, axis=1)
    _, index, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    return corners[index[counts == 1]]


def tetrahedron_edges(tetrahedra: np.ndarray) -> np.ndarray:
    """Unique undirected edges of a tetrahedral mesh."""

    cells = np.asarray(tetrahedra, dtype=np.int64)
    pairs = np.concatenate(
        [
            cells[:, [0, 1]], cells[:, [0, 2]], cells[:, [0, 3]],
            cells[:, [1, 2]], cells[:, [1, 3]], cells[:, [2, 3]],
        ],
        axis=0,
    )
    return np.unique(np.sort(pairs, axis=1), axis=0)


def elastic_value_and_gradient(
    reference: np.ndarray,
    deformed: np.ndarray,
    edges: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Reference-length spring energy on mesh edges, with its gradient.

    This exists so the barrier ablation is a single-variable comparison. Without
    it, a vertex carrying no positional target has no gradient at all and simply
    stays where it started, so the arm with the barrier disabled inverts elements
    for want of *any* term coupling the interior rather than for want of a
    barrier. Measuring that would have flattered the barrier by comparing it
    against nothing.
    """

    points = np.asarray(deformed, dtype=np.float64)
    pairs = np.asarray(edges, dtype=np.int64)
    rest = np.linalg.norm(
        np.asarray(reference, dtype=np.float64)[pairs[:, 0]]
        - np.asarray(reference, dtype=np.float64)[pairs[:, 1]],
        axis=1,
    )
    delta = points[pairs[:, 0]] - points[pairs[:, 1]]
    length = np.linalg.norm(delta, axis=1)
    stretch = length - rest
    value = float(np.sum(stretch**2))
    unit = delta / np.maximum(length, 1e-12)[:, None]
    contribution = (2.0 * stretch)[:, None] * unit
    gradient = np.zeros_like(points)
    np.add.at(gradient, pairs[:, 0], contribution)
    np.add.at(gradient, pairs[:, 1], -contribution)
    return value, gradient


def collision_value_and_gradient(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    thickness: float,
) -> tuple[float, np.ndarray]:
    r"""Vertex-triangle repulsion barrier, with its gradient.

    The Jacobian barrier is a statement about each element in isolation and says
    nothing about whether distant parts of the surface pass through one another.
    Measured in E21: with the Jacobian barrier alone, every one of 135 elements
    stayed valid while the boundary recorded 96 self-intersections. Local
    invertibility and global non-penetration are different properties, and only
    the first was implemented.

    This adds the second, in the form the mandated ablation names. For every
    vertex and every triangle it does not belong to, a distance ``d`` below
    ``thickness`` contributes :math:`-(d/h-1)^2\log(d/h)`, the same truncated
    log barrier used for element validity, so contact is repelled smoothly and
    penetration is infinitely penalized.

    This is a reference: it tests all vertex-triangle pairs, which is quadratic
    and only acceptable for meshes of the size used here, and it handles
    vertex-triangle proximity but not edge-edge crossings, which can interpenetrate
    with every vertex still far from every face. A production implementation
    needs spatial hashing and continuous collision detection over both primitive
    pairs.
    """

    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if thickness <= 0.0:
        raise ValueError("thickness must be positive")

    gradient = np.zeros_like(points)
    total = 0.0
    corners = points[triangles]
    for index in range(len(points)):
        incident = np.any(triangles == index, axis=1)
        candidate = np.flatnonzero(~incident)
        if candidate.size == 0:
            continue
        closest, weights = _closest_point_on_triangle(points[index], corners[candidate])
        offset = points[index] - closest
        distance = np.linalg.norm(offset, axis=1)
        near = distance < thickness
        if not np.any(near):
            continue
        if np.any(distance[near] <= 0.0):
            return float("inf"), np.zeros_like(points)
        ratio = distance[near] / thickness
        total += float(np.sum(-((ratio - 1.0) ** 2) * np.log(ratio)))
        derivative = -2.0 * (ratio - 1.0) * np.log(ratio) - ((ratio - 1.0) ** 2) / ratio
        direction = offset[near] / distance[near][:, None]
        contribution = (derivative / thickness)[:, None] * direction
        gradient[index] += contribution.sum(axis=0)
        # The reaction goes to the triangle corners with the barycentric weights
        # of the closest point. Splitting it evenly instead would make the
        # gradient inconsistent with the energy, and a line search that is handed
        # a direction which is not a descent direction fails quietly rather than
        # loudly.
        picked = weights[near]
        for corner in range(3):
            np.add.at(
                gradient,
                triangles[candidate[near], corner],
                -picked[:, corner][:, None] * contribution,
            )
    return total, gradient


def _closest_point_on_triangle(
    point: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Closest point on each triangle, with its barycentric weights."""

    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    ap = point[None, :] - a
    d1 = np.einsum("sk,sk->s", ap, ab)
    d2 = np.einsum("sk,sk->s", ap, ac)
    aa = np.einsum("sk,sk->s", ab, ab)
    bb = np.einsum("sk,sk->s", ab, ac)
    cc = np.einsum("sk,sk->s", ac, ac)
    denominator = np.maximum(aa * cc - bb * bb, 1e-30)
    u = (cc * d1 - bb * d2) / denominator
    v = (aa * d2 - bb * d1) / denominator
    outside = (u < 0.0) | (v < 0.0) | (u + v > 1.0)
    if np.any(outside):
        def clamp(numerator, denom):
            return np.clip(numerator / np.maximum(denom, 1e-30), 0.0, 1.0)

        bc = c - b
        t_ab = clamp(d1, aa)
        t_ac = clamp(d2, cc)
        t_bc = clamp(np.einsum("sk,sk->s", point[None, :] - b, bc),
                     np.einsum("sk,sk->s", bc, bc))
        options = np.stack([
            np.stack([t_ab, np.zeros_like(t_ab)], axis=-1),
            np.stack([np.zeros_like(t_ac), t_ac], axis=-1),
            np.stack([1.0 - t_bc, t_bc], axis=-1),
        ], axis=0)
        candidates = (a[None] + options[..., 0][..., None] * ab[None]
                      + options[..., 1][..., None] * ac[None])
        best = np.argmin(((candidates - point[None, None, :]) ** 2).sum(axis=-1), axis=0)
        picked = np.take_along_axis(options, best[None, :, None], axis=0)[0]
        u = np.where(outside, picked[:, 0], u)
        v = np.where(outside, picked[:, 1], v)
    closest = a + u[:, None] * ab + v[:, None] * ac
    weights = np.stack([1.0 - u - v, u, v], axis=1)
    return closest, weights


def _volume_gradients(vertices: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Per-tetrahedron gradient of signed volume, shape (cells, 4, 3)."""

    simplex = vertices[cells]
    a, b, c, d = simplex[:, 0], simplex[:, 1], simplex[:, 2], simplex[:, 3]
    grad_b = np.cross(c - a, d - a) / 6.0
    grad_c = np.cross(d - a, b - a) / 6.0
    grad_d = np.cross(b - a, c - a) / 6.0
    grad_a = -(grad_b + grad_c + grad_d)
    return np.stack([grad_a, grad_b, grad_c, grad_d], axis=1)


def barrier_value_and_gradient(
    reference: np.ndarray,
    deformed: np.ndarray,
    tetrahedra: np.ndarray,
    *,
    minimum_determinant: float = 0.05,
    activation_determinant: float = 0.20,
) -> tuple[float, np.ndarray]:
    r"""Barrier energy and its gradient with respect to deformed vertices.

    Matches ``interior_log_barrier`` in value: with
    :math:`s=(r-r_{\min})/(r_{\mathrm{act}}-r_{\min})` the contribution is
    :math:`-(s-1)^2\log s` while :math:`0<s<1`, zero above, and infinite at or
    below the feasibility boundary. An infeasible configuration returns an
    infinite value and a zero gradient, because the gradient is meaningless
    there and the caller must reject the step rather than follow it.
    """

    cells = np.asarray(tetrahedra, dtype=np.int64)
    reference_volume = signed_tetrahedron_volumes(reference, cells)
    if np.any(reference_volume <= 0.0):
        raise ValueError("reference tetrahedra must be positively oriented")
    deformed_volume = signed_tetrahedron_volumes(deformed, cells)
    ratio = deformed_volume / reference_volume

    span = activation_determinant - minimum_determinant
    if span <= 0.0:
        raise ValueError("activation must exceed minimum")
    scaled = (ratio - minimum_determinant) / span
    gradient = np.zeros_like(np.asarray(deformed, dtype=np.float64))
    if np.any(scaled <= 0.0):
        return float("inf"), gradient

    active = scaled < 1.0
    if not np.any(active):
        return 0.0, gradient

    s = scaled[active]
    value = float(np.sum(-((s - 1.0) ** 2) * np.log(s)))
    # d/ds of -(s-1)^2 log s
    dvalue_ds = -2.0 * (s - 1.0) * np.log(s) - ((s - 1.0) ** 2) / s
    chain = dvalue_ds / (span * reference_volume[active])

    grads = _volume_gradients(deformed, cells[active])
    contribution = chain[:, None, None] * grads
    np.add.at(gradient, cells[active].reshape(-1), contribution.reshape(-1, 3))
    return value, gradient


@dataclass(frozen=True)
class DeformationResult:
    vertices: np.ndarray
    iterations: int
    converged: bool
    data_rmse_mm: float
    barrier_value: float
    minimum_determinant: float
    inverted_cells: int
    history: np.ndarray


def fit_deformation(
    reference: np.ndarray,
    tetrahedra: np.ndarray,
    target_indices: np.ndarray,
    target_positions: np.ndarray,
    *,
    barrier_weight: float = 1.0,
    elastic_weight: float = 0.0,
    collision_weight: float = 0.0,
    collision_faces: np.ndarray | None = None,
    collision_thickness: float = 1.0,
    minimum_determinant: float = 0.05,
    activation_determinant: float = 0.20,
    max_iterations: int = 400,
    step: float = 1.0,
    tolerance: float = 1e-10,
) -> DeformationResult:
    """Fit vertices to positional targets, optionally guarded by the barrier.

    ``barrier_weight = 0`` disables the barrier entirely and is the ablation
    arm. With the barrier active, a step that would leave the feasible set is
    rejected by the line search rather than clamped, so the iterate never passes
    through an inverted configuration.
    """

    vertices = np.asarray(reference, dtype=np.float64).copy()
    cells = np.asarray(tetrahedra, dtype=np.int64)
    indices = np.asarray(target_indices, dtype=np.int64)
    targets = np.asarray(target_positions, dtype=np.float64)
    if indices.shape[0] != targets.shape[0]:
        raise ValueError("one target position is required per target index")
    if barrier_weight < 0.0 or elastic_weight < 0.0 or collision_weight < 0.0:
        raise ValueError("weights cannot be negative")
    if collision_weight > 0.0 and collision_faces is None:
        raise ValueError("collision_weight requires collision_faces")
    edges = tetrahedron_edges(cells) if elastic_weight > 0.0 else None

    def energy_and_gradient(points):
        residual = points[indices] - targets
        data = float(np.sum(residual**2))
        gradient = np.zeros_like(points)
        np.add.at(gradient, indices, 2.0 * residual)
        total = data
        if elastic_weight > 0.0:
            elastic, elastic_gradient = elastic_value_and_gradient(
                reference, points, edges
            )
            total = total + elastic_weight * elastic
            gradient = gradient + elastic_weight * elastic_gradient
        if collision_weight > 0.0:
            contact, contact_gradient = collision_value_and_gradient(
                points, collision_faces, thickness=collision_thickness
            )
            if not np.isfinite(contact):
                return float("inf"), gradient, data
            total = total + collision_weight * contact
            gradient = gradient + collision_weight * contact_gradient
        if barrier_weight > 0.0:
            value, barrier_gradient = barrier_value_and_gradient(
                reference,
                points,
                cells,
                minimum_determinant=minimum_determinant,
                activation_determinant=activation_determinant,
            )
            if not np.isfinite(value):
                return float("inf"), gradient, data
            total = total + barrier_weight * value
            gradient = gradient + barrier_weight * barrier_gradient
        return total, gradient, data

    total, gradient, data = energy_and_gradient(vertices)
    history = [total]
    scale = step
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        norm = float(np.linalg.norm(gradient))
        if norm < 1e-12:
            converged = True
            break
        direction = gradient / norm
        accepted = False
        trial_scale = scale
        for _ in range(40):
            candidate = vertices - trial_scale * direction
            candidate_total, candidate_gradient, candidate_data = energy_and_gradient(
                candidate
            )
            if np.isfinite(candidate_total) and candidate_total < total:
                vertices = candidate
                improvement = total - candidate_total
                total, gradient, data = (
                    candidate_total,
                    candidate_gradient,
                    candidate_data,
                )
                scale = trial_scale * 1.5
                accepted = True
                history.append(total)
                if improvement <= tolerance * max(abs(total), 1.0):
                    converged = True
                break
            trial_scale *= 0.5
        if not accepted or converged:
            converged = True
            break

    volumes = signed_tetrahedron_volumes(vertices, cells)
    reference_volume = signed_tetrahedron_volumes(reference, cells)
    ratio = volumes / reference_volume
    if barrier_weight > 0.0:
        barrier, _ = barrier_value_and_gradient(
            reference,
            vertices,
            cells,
            minimum_determinant=minimum_determinant,
            activation_determinant=activation_determinant,
        )
    else:
        barrier = float("nan")
    return DeformationResult(
        vertices=vertices,
        iterations=iteration,
        converged=converged,
        data_rmse_mm=float(np.sqrt(data / max(len(indices), 1))),
        barrier_value=float(barrier),
        minimum_determinant=float(ratio.min()),
        inverted_cells=int(np.count_nonzero(ratio <= 0.0)),
        history=np.asarray(history, dtype=np.float64),
    )
