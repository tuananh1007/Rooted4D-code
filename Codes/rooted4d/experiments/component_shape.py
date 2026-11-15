r"""Shape spaces on the real component topologies, for the real-image stage (Stage C).

The fusion benchmark's structured shape space (``structured_model.py``) is built
on the procedural complex. Stage C applies the selected estimator to the real
chamber and valve topologies that the image-input pipeline resamples into, so
the same construction is needed on an arbitrary component mesh with a ring
interface:

* each training deviation is split into the ring displacement and, per
  component, the interior residual left after the ring has been propagated
  inward by harmonic extension (the component's cotangent-Laplacian Dirichlet
  problem, on the training mean);
* a principal basis is learned per block, the ring modes propagated into every
  component and the interior modes (zero on the ring) span the space, whose
  columns are orthonormalized in physical coordinates and rotated onto the
  eigenvectors of the training coefficient covariance;
* the per-coefficient training scales give the covariance-aware prior, with the
  multiplier the joint solver uses (floored, normalized to unit mean square).

The global comparator is principal components at the same rank. Both return a
:class:`ShapeSpace`, and :func:`map_estimate` is the regularized fit: the MAP
estimate of a centred input under the prior, in closed form because the basis is
orthonormal, ``z = (I + beta M^2)^{-1} B^T (u - mean)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rooted4d.geometry.interface_modes import constraint_modes

RANK_TOLERANCE = 1e-8


@dataclass(frozen=True)
class ShapeSpace:
    mean: np.ndarray            # (V, 3), centred training mean
    basis: np.ndarray           # (3V, r), orthonormal columns
    scales: np.ndarray          # (r,), descending training standard deviations
    kind: str
    #: Training standard deviation per coordinate outside the span (the
    #: probabilistic-PCA residual), used when a fit keeps the complement.
    residual_scale: float = 0.0

    @property
    def rank(self) -> int:
        return self.basis.shape[1]


def covariance_multiplier(scales: np.ndarray, floor: float = 0.005) -> np.ndarray:
    """The joint solver's covariance-aware factor: 1/max(s_k/s_1, floor), unit mean square."""

    scales = np.asarray(scales, dtype=np.float64)
    relative = np.maximum(scales / float(scales[0]), max(floor, 1e-12))
    multiplier = 1.0 / relative
    return multiplier / float(np.sqrt(np.mean(multiplier**2)))


def _outside_multiplier(space: ShapeSpace, floor: float) -> float:
    """The complement's factor on the same scale as :func:`covariance_multiplier`."""

    scales = np.asarray(space.scales, dtype=np.float64)
    raw = 1.0 / np.maximum(scales / float(scales[0]), max(floor, 1e-12))
    outside = 1.0 / max(space.residual_scale / float(scales[0]), max(floor, 1e-12))
    return float(outside / np.sqrt(np.mean(raw**2)))


def map_estimate(shape: np.ndarray, space: ShapeSpace, beta: float, floor: float = 0.005,
                 complement: bool = False) -> np.ndarray:
    """Regularized fit of one resampled surface: centre, shrink in the space, restore.

    Without the complement, ``beta = 0`` is the orthogonal projection onto the
    span and everything outside it is discarded. With it, the prior is the
    probabilistic-PCA one: the part of the input outside the span is shrunk with
    the training residual scale instead of discarded, so ``beta = 0`` returns the
    input unchanged. Either way, large ``beta`` tends to the training mean at the
    input's centroid.
    """

    points = np.asarray(shape, dtype=np.float64)
    centroid = points.mean(axis=0)
    deviation = (points - centroid - space.mean).reshape(-1)
    coefficients = space.basis.T @ deviation
    outside = deviation - space.basis @ coefficients
    if beta > 0.0:
        coefficients = coefficients / (1.0 + beta * covariance_multiplier(space.scales, floor) ** 2)
        outside = outside / (1.0 + beta * _outside_multiplier(space, floor) ** 2)
    fitted = space.basis @ coefficients + (outside if complement else 0.0)
    return space.mean + fitted.reshape(space.mean.shape) + centroid


class ComponentStructure:
    """Ring interface, components, and harmonic constraint modes on a real topology."""

    def __init__(self, template: np.ndarray, faces_by_component: dict, components: dict,
                 interface: np.ndarray) -> None:
        self.template = np.asarray(template, dtype=np.float64)
        self.interface = np.sort(np.asarray(interface, dtype=np.int64))
        self.components = {k: np.sort(np.asarray(v, dtype=np.int64)) for k, v in components.items()}
        self.n_physical = len(self.template)
        self.position = {int(v): i for i, v in enumerate(self.interface)}
        self.blocks = {}
        for name, component in self.components.items():
            modes, boundary, interior = constraint_modes(
                self.template, np.asarray(faces_by_component[name], dtype=np.int64), component,
                self.interface)
            self.blocks[name] = {"constraint_modes": modes, "boundary": boundary, "interior": interior}

    def _take(self, boundary: np.ndarray) -> np.ndarray:
        return np.array([self.position[int(v)] for v in boundary], dtype=np.int64)

    def split(self, fields: np.ndarray):
        """Ring values, and each component's interior residual after propagating the ring."""

        ring = fields[:, self.interface, :].reshape(len(fields), -1)
        residuals = {}
        for name, block in self.blocks.items():
            boundary, interior = block["boundary"], block["interior"]
            propagated = np.einsum("ij,njk->nik", block["constraint_modes"],
                                   fields[:, self.interface, :][:, self._take(boundary), :])
            actual = fields[:, np.concatenate([boundary, interior]), :]
            residuals[name] = (actual - propagated)[:, boundary.size:, :].reshape(len(fields), -1)
        return ring, residuals

    def assemble(self, ring: np.ndarray, interiors: dict) -> np.ndarray:
        """Propagate the ring, average overlapping components, add the interiors."""

        count = len(ring)
        total = np.zeros((count, self.n_physical, 3))
        multiplicity = np.zeros(self.n_physical)
        for name, block in self.blocks.items():
            boundary, interior = block["boundary"], block["interior"]
            propagated = np.einsum("ij,njk->nik", block["constraint_modes"],
                                   ring.reshape(count, -1, 3)[:, self._take(boundary), :])
            rows = np.concatenate([boundary, interior])
            contribution = propagated.copy()
            if interiors.get(name) is not None:
                contribution[:, boundary.size:, :] += interiors[name].reshape(count, interior.size, 3)
            total[:, rows, :] += contribution
            multiplicity[rows] += 1.0
        touched = multiplicity > 0
        total[:, touched, :] /= multiplicity[touched, None]
        return total

    def span(self, ring_basis: np.ndarray, interior_bases: dict) -> np.ndarray:
        columns = []
        zero_ring = np.zeros((1, self.interface.size * 3))
        none: dict = dict.fromkeys(self.components)
        for j in range(ring_basis.shape[1]):
            columns.append(self.assemble(ring_basis[:, j][None, :], none).ravel())
        for name, block_basis in interior_bases.items():
            for j in range(block_basis.shape[1]):
                interiors = dict.fromkeys(self.components)
                interiors[name] = block_basis[:, j][None, :]
                columns.append(self.assemble(zero_ring, interiors).ravel())
        return np.column_stack(columns)


def _principal(rows: np.ndarray, n: int) -> np.ndarray:
    _, _, right = np.linalg.svd(rows, full_matrices=False)
    return right[: min(n, right.shape[0])].T.copy()


def _finish(basis: np.ndarray, deviations: np.ndarray, rotate: bool, kind: str, mean: np.ndarray) -> ShapeSpace:
    """Covariance axes (optional), descending scales, and a fixed sign convention."""

    flat = deviations.reshape(len(deviations), -1)
    coefficients = flat @ basis
    if rotate:
        covariance = coefficients.T @ coefficients / max(len(flat) - 1, 1)
        values, vectors = np.linalg.eigh(covariance)
        basis = basis @ vectors[:, np.argsort(values)[::-1]]
        coefficients = flat @ basis
    scales = np.sqrt((coefficients**2).sum(axis=0) / max(len(flat) - 1, 1))
    order = np.argsort(scales)[::-1]
    basis, scales = basis[:, order], scales[order]
    pivots = np.abs(basis).argmax(axis=0)
    basis = basis * np.sign(basis[pivots, np.arange(basis.shape[1])])
    # Per-coordinate variance left outside the span; centred shapes carry no
    # translation, so the complement has three dimensions fewer than it appears.
    residual = flat - (flat @ basis) @ basis.T
    freedom = max(flat.shape[1] - basis.shape[1] - 3, 1)
    residual_scale = float(np.sqrt((residual**2).sum() / (max(len(flat) - 1, 1) * freedom)))
    return ShapeSpace(mean=mean, basis=basis, scales=scales, kind=kind, residual_scale=residual_scale)


def chamber_structure(n_rings: int = 8, n_angles: int = 32) -> dict:
    """The ACDC chamber as one component whose interface is the basal ring.

    Arguments for :func:`fit_structured_space`. Vertex correspondence needs ring
    0 to be basal in every surface, so fit with ``fit_lv_surface(...,
    base_first=True)``: ACDC stacks all come with slice 0 basal, and the
    radius test the fitter otherwise applies is wrong on a few frames.
    """

    from rooted4d.active.acdc import lv_topology_faces

    count = n_rings * n_angles + 2
    return {"faces_by_component": {"chamber": lv_topology_faces(n_rings, n_angles)},
            "components": {"chamber": np.arange(count)}, "interface": np.arange(n_angles)}


def centre(shapes: np.ndarray) -> np.ndarray:
    shapes = np.asarray(shapes, dtype=np.float64)
    return shapes - shapes.mean(axis=1, keepdims=True)


def fit_structured_space(shapes: np.ndarray, faces_by_component: dict, components: dict,
                         interface: np.ndarray, ring_dim: int, interior_dim: int) -> ShapeSpace:
    """The structured space on a real topology, learned from centred training shapes."""

    centred = centre(shapes)
    mean = centred.mean(axis=0)
    deviations = centred - mean
    structure = ComponentStructure(mean, faces_by_component, components, interface)
    ring, residuals = structure.split(deviations)
    ring_basis = _principal(ring, ring_dim)
    interior_bases = {name: _principal(rows, interior_dim) for name, rows in residuals.items()}
    assembled = structure.span(ring_basis, interior_bases)
    # A uniform ring displacement propagates to a rigid translation of the whole
    # component, so the assembled span is not translation-free, while every
    # training shape is centred. Removing each column's centroid keeps exactly
    # the part of the span that centred shapes can use, and makes the fit's
    # centre-then-project step consistent with the space.
    columns = assembled.reshape(len(mean), 3, -1)
    assembled = (columns - columns.mean(axis=0, keepdims=True)).reshape(assembled.shape)
    left, singular, _ = np.linalg.svd(assembled, full_matrices=False)
    rank = int(np.count_nonzero(singular > RANK_TOLERANCE * singular[0]))
    return _finish(left[:, :rank], deviations, rotate=True, kind="structured", mean=mean)


def fit_global_space(shapes: np.ndarray, rank: int) -> ShapeSpace:
    """Principal components of the centred training shapes at a given rank."""

    centred = centre(shapes)
    mean = centred.mean(axis=0)
    deviations = centred - mean
    _, _, right = np.linalg.svd(deviations.reshape(len(deviations), -1), full_matrices=False)
    return _finish(right[:rank].T.copy(), deviations, rotate=False, kind="global", mean=mean)
