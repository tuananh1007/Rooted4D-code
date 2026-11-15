r"""Interface-conditioned component bases: the Craig--Bampton split.

The current shape space is a single global basis over every physical vertex, so
each mode displaces the chamber and the leaflets together. That is efficient
when the anatomy really does move together and a liability when it does not:
the representation floors on the v3 test donors are 1.35 mm over CMR-visible
vertices and 2.14 mm over TEE-visible ones, close enough to the fusion errors
that some of the accuracy attributed to coupling may be compression instead.

This module provides the alternative decomposition, one block per component:

.. math::

    u_m = H_m a + N_m z_m

with

* ``a`` the displacement of the shared interface, the annular ring, which every
  component sees identically. Attachment is therefore exact by construction for
  any coefficients, exactly as in the assembled model: the interface is written
  once.
* ``H_m`` the **constraint modes**, the harmonic extension of an interface
  displacement into component ``m``. Column ``j`` is the field that equals one
  at interface vertex ``j`` and is harmonic in the interior, so interface motion
  propagates smoothly inward instead of being resisted by the interior basis.
* ``N_m`` the **interior modes**, Dirichlet eigenmodes of the component
  Laplacian with the interface held fixed. Each is exactly zero on the
  interface, so the interior can deform without moving the attachment, and each
  component can carry a different number of them.

This representation is not new and is not claimed as a contribution: it is
component-mode synthesis, standard in structural dynamics, and the shared/private
split has a long statistical history under names like JIVE. What is open here is
the *inference* -- how much interface motion the observations of each modality
actually support once the interiors, pose, phase and session are free to adjust.
This module exists so that question can be asked.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "cotangent_laplacian",
    "constraint_modes",
    "interior_modes",
    "InterfaceConditionedBasis",
    "interface_conditioned_basis",
]


def cotangent_laplacian(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Symmetric cotangent Laplacian of a triangle mesh, as a dense matrix.

    Returns ``L`` with positive diagonal and non-positive off-diagonal entries,
    so ``x^T L x`` is the Dirichlet energy. Degenerate triangles contribute
    nothing rather than infinities: a zero-area triangle has no well-defined
    cotangent weight, and clamping is preferable to propagating a NaN into an
    eigensolve where it would surface as an unrelated failure much later.

    The weights are clamped below at zero. Obtuse triangles can otherwise give
    negative cotangents and an indefinite operator, which would make the
    Dirichlet eigenproblem below meaningless.
    """

    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("vertices must have shape (vertices, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("faces must have shape (faces, 3)")

    n = points.shape[0]
    laplacian = np.zeros((n, n), dtype=np.float64)
    for triangle in triangles:
        i, j, k = int(triangle[0]), int(triangle[1]), int(triangle[2])
        for a, b, opposite in ((i, j, k), (j, k, i), (k, i, j)):
            first = points[a] - points[opposite]
            second = points[b] - points[opposite]
            cross = np.linalg.norm(np.cross(first, second))
            if cross <= 1e-12:
                continue
            weight = 0.5 * float(np.dot(first, second)) / cross
            weight = max(weight, 0.0)
            laplacian[a, b] -= weight
            laplacian[b, a] -= weight
            laplacian[a, a] += weight
            laplacian[b, b] += weight
    return laplacian


def _partition(component: np.ndarray, interface: np.ndarray):
    """Split a component's vertices into interface and interior, in order."""

    component = np.asarray(component, dtype=np.int64)
    interface = np.asarray(interface, dtype=np.int64)
    on_interface = np.isin(component, interface)
    boundary = component[on_interface]
    interior = component[~on_interface]
    if boundary.size == 0:
        raise ValueError("component does not touch the interface")
    return boundary, interior


def constraint_modes(
    vertices: np.ndarray,
    faces: np.ndarray,
    component: np.ndarray,
    interface: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Harmonic extension of interface displacement into a component.

    Returns ``(modes, boundary, interior)`` where ``modes`` has one row per
    vertex of ``component`` (interface rows first, then interior, in the order
    of the returned index arrays) and one column per interface vertex. The
    interface block is the identity, so an interface displacement is reproduced
    exactly, and the interior block is ``-L_II^{-1} L_IB``.
    """

    boundary, interior = _partition(component, interface)
    laplacian = cotangent_laplacian(vertices, faces)
    if interior.size == 0:
        return np.eye(boundary.size), boundary, interior

    inner = laplacian[np.ix_(interior, interior)]
    coupling = laplacian[np.ix_(interior, boundary)]
    # A pinned Laplacian is positive definite on a connected interior, but a
    # component whose interior touches the interface only weakly can still be
    # badly conditioned; solve in the least-squares sense rather than inverting.
    extension = -np.linalg.lstsq(inner, coupling, rcond=None)[0]
    modes = np.vstack([np.eye(boundary.size), extension])
    return modes, boundary, interior


def interior_modes(
    vertices: np.ndarray,
    faces: np.ndarray,
    component: np.ndarray,
    interface: np.ndarray,
    n_modes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dirichlet eigenmodes of a component with its interface held fixed.

    Returns ``(modes, boundary, interior)`` with ``modes`` shaped
    ``(len(component), n_modes)``, ordered interface rows then interior rows.
    The interface rows are exactly zero, which is the property the whole
    decomposition rests on: an interior mode cannot move the attachment.
    """

    if n_modes < 0:
        raise ValueError("n_modes must be non-negative")
    boundary, interior = _partition(component, interface)
    if n_modes == 0 or interior.size == 0:
        return np.zeros((boundary.size + interior.size, 0)), boundary, interior

    laplacian = cotangent_laplacian(vertices, faces)
    inner = laplacian[np.ix_(interior, interior)]
    inner = 0.5 * (inner + inner.T)
    values, vectors = np.linalg.eigh(inner)
    order = np.argsort(values)
    keep = min(n_modes, interior.size)
    selected = vectors[:, order[:keep]]
    modes = np.zeros((boundary.size + interior.size, keep), dtype=np.float64)
    modes[boundary.size :, :] = selected
    return modes, boundary, interior


class InterfaceConditionedBasis:
    """Per-component displacement basis sharing one interface block.

    ``assemble`` maps interface coefficients and per-component interior
    coefficients to a displacement over the physical template. The interface is
    written from ``a`` alone, so every component agrees there by construction
    rather than by penalty.
    """

    def __init__(self, n_physical: int, interface: np.ndarray, blocks: dict):
        self.n_physical = int(n_physical)
        self.interface = np.asarray(interface, dtype=np.int64)
        self.blocks = blocks

    @property
    def interface_dimension(self) -> int:
        return 3 * self.interface.size

    @property
    def interior_dimensions(self) -> dict:
        return {name: 3 * block["interior_modes"].shape[1]
                for name, block in self.blocks.items()}

    @property
    def total_dimension(self) -> int:
        return self.interface_dimension + sum(self.interior_dimensions.values())

    def assemble(self, interface_coefficients: np.ndarray,
                 interior_coefficients: dict | None = None) -> np.ndarray:
        """Displacement field over the physical template, shape (n_physical, 3).

        Vertices belonging to no component are left at zero; the caller decides
        whether that is a template defect or an intentionally unmodelled region.
        """

        a = np.asarray(interface_coefficients, dtype=np.float64).reshape(-1, 3)
        if a.shape[0] != self.interface.size:
            raise ValueError(
                "interface coefficients describe %d vertices, interface has %d"
                % (a.shape[0], self.interface.size))
        interior_coefficients = interior_coefficients or {}

        displacement = np.zeros((self.n_physical, 3), dtype=np.float64)
        for name, block in self.blocks.items():
            boundary, interior = block["boundary"], block["interior"]
            rows = np.concatenate([boundary, interior])
            # Interface coefficients are indexed by position in self.interface,
            # so reorder them into this component's boundary order.
            position = {int(v): i for i, v in enumerate(self.interface)}
            take = np.array([position[int(v)] for v in boundary], dtype=np.int64)
            field = block["constraint_modes"] @ a[take]
            z = interior_coefficients.get(name)
            if z is not None and block["interior_modes"].shape[1]:
                z = np.asarray(z, dtype=np.float64).reshape(-1, 3)
                if z.shape[0] != block["interior_modes"].shape[1]:
                    raise ValueError(
                        "component %r expects %d interior modes, received %d"
                        % (name, block["interior_modes"].shape[1], z.shape[0]))
                field = field + block["interior_modes"] @ z
            displacement[rows] += field
        # An interface vertex is written by every component that touches it, and
        # they all write the same value, so divide by the multiplicity rather
        # than accumulating it.
        multiplicity = np.zeros(self.n_physical, dtype=np.float64)
        for block in self.blocks.values():
            rows = np.concatenate([block["boundary"], block["interior"]])
            multiplicity[rows] += 1.0
        touched = multiplicity > 0
        displacement[touched] /= multiplicity[touched, None]
        return displacement


def interface_conditioned_basis(
    vertices: np.ndarray,
    component_faces: dict,
    component_vertices: dict,
    interface: np.ndarray,
    interior_mode_counts: dict,
) -> InterfaceConditionedBasis:
    """Build the per-component basis blocks.

    ``interior_mode_counts`` may differ per component, which is the point: the
    chamber and the leaflets need not carry the same deformation capacity, and
    under a single global basis they necessarily do.
    """

    points = np.asarray(vertices, dtype=np.float64)
    blocks = {}
    for name, faces in component_faces.items():
        component = np.asarray(component_vertices[name], dtype=np.int64)
        constraints, boundary, interior = constraint_modes(
            points, faces, component, interface)
        interiors, boundary_check, interior_check = interior_modes(
            points, faces, component, interface,
            int(interior_mode_counts.get(name, 0)))
        if not (np.array_equal(boundary, boundary_check)
                and np.array_equal(interior, interior_check)):
            raise RuntimeError("partition disagreed between the two mode builders")
        blocks[name] = {
            "constraint_modes": constraints,
            "interior_modes": interiors,
            "boundary": boundary,
            "interior": interior,
        }
    return InterfaceConditionedBasis(points.shape[0], interface, blocks)
