r"""Train-only builder for the interface-conditioned structured shape space.

E53 compared a structured shape space against global PCA and reported the
structured one ahead. Two defects made that comparison unusable: the structured
arm was scored by a blockwise projection while the global arm received an
optimal one, and the allocations were chosen by looking at test. The corrected
diagnostic ``e53b_projection_independent`` fixed the projection but still reads
test geometry and still inherits test-selected allocations, so it remains a
diagnostic rather than a result.

This module extracts the construction from that diagnostic and nothing else:

* everything is learned from the **train** identities, deduplicated, so 90
  identities cannot masquerade as 180 shapes;
* the assembled span is orthonormalized in physical displacement coordinates,
  the same coordinates the global PCA basis lives in, so the identity prior
  penalises the same physical quantity in both arms;
* the retained dimension is the **numerical rank** of the assembled span, not
  the number of columns appended to it -- the interface block and the interior
  blocks overlap on the boundary ring and the propagate-average-add operator
  is not injective, so columns and rank are different numbers;
* motion and session directions are relearned in the new latent coordinates
  from the same training fields, at matched ranks and temporal orders.

The product is an ordinary :class:`SceneModel`, so the structured arm can be
put through the identical solver, prior and evaluation harness as the global
one. That is the only way the comparison means anything: E53 measured static
projection RMS of known geometry, which is not the quantity the method is
judged on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rooted4d.experiments.benchmark_eval import BENCHMARK_ROOT, _identity_geometry, benchmark_topology
from rooted4d.geometry import toy_complex_for_vertices
from rooted4d.geometry.interface_modes import interface_conditioned_basis
from rooted4d.inference.scene import SceneModel
from rooted4d.priors import GaugedLatentBases

#: Singular values below this fraction of the leading one are treated as null
#: directions of the assembly operator rather than as basis members.
RANK_TOLERANCE = 1e-8


def _principal_basis(rows: np.ndarray, n: int) -> np.ndarray:
    """Leading ``n`` right singular vectors of ``rows``, as columns."""

    if n <= 0 or rows.size == 0:
        return np.zeros((rows.shape[1], 0))
    _, _, right = np.linalg.svd(rows, full_matrices=False)
    return right[: min(n, right.shape[0])].T.copy()


class _Structure:
    """The interface ring, the component partition, and the constraint modes.

    Built on the mean **training** template rather than on the procedural
    complex's own vertices. The constraint modes are harmonic with respect to
    whatever surface they are built on, and the surface this model deforms is
    the train mean, which is the template the resulting ``SceneModel`` carries.
    Nothing here reads validation or test geometry.
    """

    def __init__(self, template: np.ndarray, topology: dict[str, int] | None = None) -> None:
        complex_ = toy_complex_for_vertices(len(template), **(topology or {}))
        interface = np.sort(complex_.groups["annulus"])
        local = complex_.attachment_map.local_to_global
        # Each leaflet's own vertices beyond its root: the free edge, and on a
        # richer instance (W2.6) the rows between.
        components = {
            "chamber": np.sort(np.union1d(complex_.groups["chamber_unique"], interface)),
            "anterior_leaflet": np.sort(np.union1d(np.setdiff1d(local["anterior_leaflet"], interface), interface)),
            "posterior_leaflet": np.sort(np.union1d(np.setdiff1d(local["posterior_leaflet"], interface), interface)),
        }
        faces = {name: complex_.faces[name] for name in components}
        capacity = {
            name: int(np.setdiff1d(rows, interface).size)
            for name, rows in components.items()
        }
        self.complex = complex_
        self.interface = interface
        self.components = components
        self.n_physical = len(template)
        self.basis = interface_conditioned_basis(
            template, faces, components, interface, capacity
        )
        self.position = {int(v): i for i, v in enumerate(interface)}

    def _take(self, boundary: np.ndarray) -> np.ndarray:
        return np.array([self.position[int(v)] for v in boundary])

    def split(self, fields: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Interface ring values, and each component's interior residual.

        The residual is what the ring cannot already explain: the constraint
        modes propagate the ring inward, and what is left over is the interior's
        own deformation. Splitting this way is what lets the components carry
        different capacity.
        """

        ring = fields[:, self.interface, :].reshape(len(fields), -1)
        residuals = {}
        for name, block in self.basis.blocks.items():
            boundary, interior = block["boundary"], block["interior"]
            propagated = np.einsum(
                "ij,njk->nik",
                block["constraint_modes"],
                fields[:, self.interface, :][:, self._take(boundary), :],
            )
            actual = fields[:, np.concatenate([boundary, interior]), :]
            residuals[name] = (actual - propagated)[:, boundary.size :, :].reshape(
                len(fields), -1
            )
        return ring, residuals

    def assemble(
        self, ring: np.ndarray, interiors: dict[str, np.ndarray | None]
    ) -> np.ndarray:
        """Propagate the ring, average the overlaps, add the interiors."""

        count = len(ring)
        total = np.zeros((count, self.n_physical, 3))
        multiplicity = np.zeros(self.n_physical)
        for name, block in self.basis.blocks.items():
            boundary, interior = block["boundary"], block["interior"]
            propagated = np.einsum(
                "ij,njk->nik", block["constraint_modes"], ring[:, self._take(boundary), :]
            )
            rows = np.concatenate([boundary, interior])
            contribution = propagated.copy()
            if interiors.get(name) is not None:
                contribution[:, boundary.size :, :] += interiors[name].reshape(
                    count, interior.size, 3
                )
            total[:, rows, :] += contribution
            multiplicity[rows] += 1.0
        touched = multiplicity > 0
        total[:, touched, :] /= multiplicity[touched, None]
        return total

    def span(self, ring_basis: np.ndarray, interior_bases: dict[str, np.ndarray]):
        """Push each unit coefficient through ``assemble`` to get its field.

        Deriving the span from the assembly operator itself, rather than
        rebuilding the harmonic matrix by hand, guarantees the columns span
        exactly the set of fields this construction can produce.
        """

        columns = []
        zero_ring = np.zeros((1, self.interface.size, 3))
        none_interiors: dict[str, np.ndarray | None] = dict.fromkeys(self.components)
        for j in range(ring_basis.shape[1]):
            field = ring_basis[:, j].reshape(1, self.interface.size, 3)
            columns.append(self.assemble(field, none_interiors).ravel())
        for name, block_basis in interior_bases.items():
            for j in range(block_basis.shape[1]):
                interiors: dict[str, np.ndarray | None] = dict.fromkeys(self.components)
                interiors[name] = block_basis[:, j][None, :]
                columns.append(self.assemble(zero_ring, interiors).ravel())
        if not columns:
            raise ValueError("the structured span has no columns")
        return np.column_stack(columns)


def fit_structured_benchmark_model(
    root: Path | str = BENCHMARK_ROOT,
    *,
    interface_dim: int,
    interior_dim: int,
    n_motion: int = 4,
    n_session: int = 2,
    n_phase_nodes: int = 8,
    coordinates: str = "assembly",
    exclude_identities: frozenset[str] | set[str] | None = None,
) -> tuple[SceneModel, dict[str, Any]]:
    """Learn the structured shape space from the train split only, then freeze it.

    ``interface_dim`` is the rank kept for the shared annular ring, and
    ``interior_dim`` the rank kept for each component's interior residual. The
    resulting model's ``n_modes`` is the numerical rank of the assembled span,
    which is generally smaller than ``interface_dim + 3 * interior_dim``; the
    caller has to read it back rather than assume it.

    ``coordinates`` chooses the axes inside that span, which the geometry does
    not care about and a diagonal prior very much does:

    * ``"assembly"`` keeps the left singular vectors of the assembled span and
      records the marginal training scale along each. Those axes do not
      diagonalise the training covariance -- on the rank-48 allocation the
      largest coefficient correlation is 0.94 -- so a diagonal prior over them
      is an *approximation* that discards the cross-covariance. This is the
      estimator E57 measured, kept so it stays reproducible and so it can be
      compared explicitly.
    * ``"covariance"`` rotates inside the same span onto the eigenvectors of the
      training coefficient covariance. The coefficients are then uncorrelated on
      train, a diagonal prior over them *is* the full empirical-covariance
      prior, and the structured arm receives the treatment the global PCA arm
      has had all along, since PCA coordinates diagonalise their own covariance
      by construction.

    The span, the mean, the rank and every attachment property are identical in
    the two; only the coordinates differ. Motion and session directions are
    learned after the rotation, by projecting the same training fields onto the
    final basis, so they are the same physical subspaces expressed in the new
    coordinates rather than something relearned differently.
    """

    if coordinates not in {"assembly", "covariance"}:
        raise ValueError("coordinates must be 'assembly' or 'covariance'")

    root = Path(root)
    templates, motions, sessions = _identity_geometry(root, "train", exclude_identities)
    mean_template = templates.mean(axis=0)
    deviations = templates - mean_template

    structure = _Structure(mean_template, benchmark_topology(root))
    ring, residuals = structure.split(deviations)
    ring_basis = _principal_basis(ring, interface_dim)
    interior_bases = {
        name: _principal_basis(rows, interior_dim) for name, rows in residuals.items()
    }

    assembled = structure.span(ring_basis, interior_bases)
    left, singular, _ = np.linalg.svd(assembled, full_matrices=False)
    rank = int(np.count_nonzero(singular > RANK_TOLERANCE * singular[0]))
    if rank < 1:
        raise ValueError("the assembled span collapsed to rank zero")
    shape_basis = left[:, :rank].copy()

    flat = deviations.reshape(len(deviations), -1)
    coefficients = flat @ shape_basis
    covariance = coefficients.T @ coefficients / max(len(deviations) - 1, 1)
    marginal = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(marginal, marginal)
    np.fill_diagonal(correlation, 0.0)
    offdiagonal = covariance - np.diag(np.diag(covariance))
    discarded = {
        "max_abs_offdiagonal_correlation": float(np.abs(correlation).max()),
        "offdiagonal_covariance_fraction": float(
            np.linalg.norm(offdiagonal) / np.linalg.norm(covariance)
        ),
    }
    if coordinates == "covariance":
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
        shape_basis = shape_basis @ eigenvectors
        coefficients = flat @ shape_basis
    # The same quantity the global arm records: the standard deviation of the
    # training identities' coefficient along each direction. For a PCA basis
    # that is the singular value over sqrt(N-1); for this basis the directions
    # are not the data's own principal axes, so it has to be measured.
    scales = np.sqrt((coefficients**2).sum(axis=0) / max(len(deviations) - 1, 1))
    # The identity prior floors each scale relative to ``scales[0]``, a
    # convention that assumes a descending spectrum. A PCA basis supplies one
    # for free and this one does not, so impose it here rather than teach the
    # solver a second convention.
    order = np.argsort(scales)[::-1]
    shape_basis = shape_basis[:, order]
    scales = scales[order]
    coefficients = coefficients[:, order]
    # Every column's sign is arbitrary: the assembly SVD's and the eigenvectors'
    # alike, and LAPACK builds disagree on both. Fix it on the final physical
    # columns, so the largest-magnitude entry of each is positive. No result
    # depends on it -- a sign flip is an exact floating-point reflection of the
    # solver's path -- but a reproducible basis should not either. Done before
    # motion and session are learned, so they see the same columns.
    pivots = np.abs(shape_basis).argmax(axis=0)
    signs = np.sign(shape_basis[pivots, np.arange(shape_basis.shape[1])])
    shape_basis = shape_basis * signs
    coefficients = coefficients * signs

    n_vertices = templates.shape[1]

    def project(fields: np.ndarray) -> np.ndarray:
        return fields.reshape(-1, n_vertices * 3) @ shape_basis

    bases = GaugedLatentBases.from_samples(
        project(motions), project(sessions), n_motion=n_motion, n_session=n_session
    )

    residual = flat - coefficients @ shape_basis.T
    total = float((flat**2).sum())
    model = SceneModel(
        template_physical=mean_template,
        attachment_map=structure.complex.attachment_map,
        groups=structure.complex.groups,
        shape_basis=shape_basis,
        bases=bases,
        n_phase_nodes=n_phase_nodes,
        shape_scales=scales.copy(),
    )
    provenance = {
        "builder": "structured_interface_conditioned",
        "coordinates": coordinates,
        # Measured in the assembly axes, whichever coordinates were kept: this is
        # what a diagonal prior over those axes throws away, and it is zero by
        # construction once the covariance rotation has been applied.
        "assembly_axes": discarded,
        "training_identities": int(len(templates)),
        "interface_dim": int(interface_dim),
        "interior_dim_per_component": int(interior_dim),
        "columns": int(assembled.shape[1]),
        "numerical_rank": rank,
        "rank_deficiency": int(assembled.shape[1] - rank),
        "variance_captured": float(1.0 - (residual**2).sum() / max(total, 1e-30)),
        "scale_head": [float(v) for v in scales[: min(16, rank)]],
        "relative_smallest_scale": float(scales[-1] / scales[0]),
    }
    return model, provenance
