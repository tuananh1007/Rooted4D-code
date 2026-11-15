r"""The constrained scene model that joint inference optimizes.

This assembles the pieces the earlier work packages produced into one object:

* exact attachment, because geometry is built as ``x = A q`` from unique physical
  coordinates (Theorem 1), so no attachment loss appears anywhere below;
* a reduced shape basis ``B`` so the inverse problem stays low dimensional, as
  section 6.5 requires;
* the section 4.2 decomposition ``zeta_m(tau) = zeta_id + U_h h(tau) + U_s r_m(tau)``
  with block-orthogonal bases and both centering gauges;
* periodic Fourier motion (Proposition 4) and a monotone circular phase warp.

The shape basis is analytic rather than learned. Learning identity and motion
bases from registered training geometries is work package 3 and needs donors that
have been fitted to a frozen template, which remains a blocked readiness gate.
A deterministic smooth basis lets the solver be built and tested now without
pretending a population prior exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rooted4d.constraints import AttachmentMap, constraint_matrix_from_equalities
from rooted4d.geometry import RigidTransform, make_toy_complex
from rooted4d.motion import MonotoneCircularPhaseWarp
from rooted4d.priors import GaugedLatentBases
from rooted4d.theory import spanning_tree_equalities


def smooth_shape_basis(vertices: np.ndarray, n_modes: int) -> np.ndarray:
    """Deterministic low-frequency displacement modes over the template.

    Each mode displaces vertices along one axis by a product of low-order
    trigonometric functions of the normalized coordinates. The result is
    orthonormalized, so the basis is well conditioned regardless of how the
    template is scaled.
    """

    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("vertices must have shape (vertices, 3)")
    if n_modes < 1:
        raise ValueError("n_modes must be at least one")
    lower = points.min(axis=0)
    extent = np.maximum(points.max(axis=0) - lower, 1e-9)
    normalized = (points - lower) / extent

    columns = []
    frequency = 0
    while len(columns) < n_modes:
        for axis in range(3):
            for kind in range(2):
                if len(columns) >= n_modes:
                    break
                argument = np.pi * (frequency + 1) * normalized[:, (axis + kind) % 3]
                weight = np.cos(argument) if kind == 0 else np.sin(argument)
                field = np.zeros_like(points)
                field[:, axis] = weight
                columns.append(field.reshape(-1))
        frequency += 1
    basis = np.stack(columns, axis=1)
    orthonormal, _ = np.linalg.qr(basis)
    return orthonormal[:, :n_modes]


@dataclass(frozen=True)
class SceneModel:
    """Template, exact assembly map, reduced shape space, and gauged subspaces."""

    template_physical: np.ndarray
    attachment_map: AttachmentMap
    groups: dict[str, np.ndarray]
    shape_basis: np.ndarray
    bases: GaugedLatentBases
    n_phase_nodes: int
    sessions: tuple[str, ...] = ("cmr", "tee")
    #: Per-mode anatomical standard deviation over the training identities, in
    #: the units of the shape coefficients. Optional, and ``None`` on any model
    #: whose basis was not learned from data -- the procedural fixture, and the
    #: detached model, whose basis is the lifted coupled block plus separation
    #: modes and therefore a different space where these scales do not apply.
    #: A solver may use it to shrink weak shape directions more than strong
    #: ones; without it the identity prior is isotropic.
    shape_scales: np.ndarray | None = None
    #: Set only on a detached model: the equality system a soft penalty must
    #: fight, and the local row each physical vertex is observed through.
    soft_constraint: np.ndarray | None = None
    observation_index: np.ndarray | None = None
    #: Number of trailing shape modes that break attachment. Zero for a
    #: coupled model; the penalty acts on exactly these coordinates.
    separation_dimension: int = 0
    #: Set only on a detached model built with ``matched=True``: how many
    #: leading coordinates are the coupled model's own coefficients, and that
    #: model's scales, so the solver can apply the coupled identity prior to
    #: exactly those coordinates. None on every other model.
    matched_lifted_dimension: int | None = None
    parent_shape_scales: np.ndarray | None = None

    @property
    def is_detached(self) -> bool:
        """True for the soft-attachment baseline, where components are separate."""

        return self.soft_constraint is not None

    def observation_rows(self, physical_indices: np.ndarray) -> np.ndarray:
        """Template rows through which the given physical vertices are observed."""

        indices = np.asarray(physical_indices, dtype=np.int64)
        if self.observation_index is None:
            return indices
        return self.observation_index[indices]

    def detached(
        self, separation_modes: int | None = None, *, matched: bool = False
    ) -> "SceneModel":
        """Return the same anatomy with the interface free rather than shared.

        This is the ``joint_soft_attachment`` baseline. Making it a *fair*
        baseline is subtler than it looks, and the obvious constructions both
        fail:

        A single positional basis over duplicated vertices gives two coincident
        occurrences identical displacements, so attachment holds for free and the
        arm tests nothing. Per-component bases fix that but change the shape
        space as well as the coupling, so any accuracy difference confounds the
        two.

        The construction used here changes only the coupling. Theorem 1 gives
        ``im(A) = ker(C)``, so lifting the coupled basis through the assembly map
        yields ``A B`` with ``C(AB) = 0``: exactly the coupled model's shape
        space, expressed on duplicated coordinates, still perfectly attached.
        Detachment is then *added* as separation modes drawn from the row space
        of ``C``, which is orthogonal to ``ker(C)`` and therefore to the lifted
        block. The consequences are what make the comparison clean:

        * every geometry the coupled model can express, this one can too;
        * the attachment residual depends on the separation coefficients alone,
          because ``CA = 0`` kills the rest, so the penalty is well conditioned
          instead of fighting the whole parameter vector;
        * as the penalty weight grows the separation block is driven to zero and
          the arm converges to the exact model, which is the behaviour a fair
          baseline must have.

        By default that last property does not quite hold. The lifted columns
        are renormalised by ``||A B_j||`` (about 1.04 to 1.55 on the benchmark
        models), which changes the coordinates everything else is expressed in:

        * an isotropic prior on the renormalised coefficients penalises some
          anatomical directions up to ~2.4 times harder than the coupled model;
        * a covariance-aware prior is lost entirely, since no scales are kept;
        * motion and session reuse the coupled bases unscaled, so their physical
          subspaces become ``A B diag(1/||A B_j||) U`` rather than ``A B U`` --
          not the subspaces the coupled model has, despite sharing the matrices;
        * the gauges are imposed in the renormalised coordinates.

        ``matched=True`` skips the renormalisation. The lifted coordinates are
        then the coupled coefficients themselves, so the motion and session
        subspaces, the gauges, and -- through ``parent_shape_scales`` -- the
        coupled identity prior all carry over unchanged, and a stiff penalty
        reproduces the exact model. The separation modes get no identity prior:
        they have no anatomical distribution to shrink toward, and the
        attachment penalty is this arm's designed restraint on them. The default
        stays as it was so earlier results reproduce.
        """

        if self.is_detached:
            raise ValueError("model is already detached")
        order = list(self.attachment_map.local_to_global.values())
        local_to_physical = np.concatenate(order).astype(np.int64)
        n_local = len(local_to_physical)
        local_template = self.template_physical[local_to_physical]

        observation_index = np.full(self.n_physical, -1, dtype=np.int64)
        for row, physical in enumerate(local_to_physical.tolist()):
            if observation_index[physical] < 0:
                observation_index[physical] = row

        constraint = constraint_matrix_from_equalities(
            n_local, spanning_tree_equalities(self.attachment_map)
        )
        identity_map = AttachmentMap(
            local_to_global={"detached": np.arange(n_local, dtype=np.int64)},
            n_global_vertices=n_local,
        )

        # Lift the coupled basis: A B, which satisfies C(AB) = 0 by Theorem 1.
        assembly = self.attachment_map.coordinate_assembly_matrix()
        lifted = assembly @ self.shape_basis
        if not matched:
            lifted = lifted / np.maximum(np.linalg.norm(lifted, axis=0), 1e-12)

        # Separation modes span the attachment-violating directions.
        count = self.n_modes if separation_modes is None else int(separation_modes)
        if count < 1:
            raise ValueError("a detached model needs at least one separation mode")
        _, singular, right = np.linalg.svd(constraint, full_matrices=False)
        rank = int(np.count_nonzero(singular > 1e-10 * singular[0]))
        if count > rank:
            raise ValueError(
                f"only {rank} independent separation directions exist; {count} requested"
            )
        separation = right[:count].T.copy()
        basis = np.concatenate([lifted, separation], axis=1)

        # Motion and session act on the coupled block only, so both arms share
        # exactly the same subspaces and differ solely by the separation DOFs.
        pad = np.zeros((count, self.bases.n_motion))
        motion = np.concatenate([self.bases.motion, pad], axis=0)
        pad = np.zeros((count, self.bases.n_session))
        session = np.concatenate([self.bases.session, pad], axis=0)

        return SceneModel(
            template_physical=local_template,
            attachment_map=identity_map,
            groups=self.groups,
            shape_basis=basis,
            bases=GaugedLatentBases(motion=motion, session=session),
            n_phase_nodes=self.n_phase_nodes,
            sessions=self.sessions,
            soft_constraint=constraint,
            observation_index=observation_index,
            separation_dimension=count,
            matched_lifted_dimension=self.n_modes if matched else None,
            parent_shape_scales=(
                None if not matched or self.shape_scales is None
                else np.asarray(self.shape_scales, dtype=np.float64).copy()
            ),
        )

    def detached_attachment_residual(self, latent: np.ndarray) -> float:
        """Max interface violation of a detached fit, in millimetres."""

        if self.soft_constraint is None:
            raise ValueError("only a detached model has a soft constraint to violate")
        coordinates = self.physical_coordinates(latent)
        flat = coordinates.reshape(*coordinates.shape[:-2], -1)
        violation = flat @ self.soft_constraint.T
        return float(np.max(np.abs(violation)))

    @classmethod
    def procedural(
        cls,
        n_annulus: int = 24,
        n_modes: int = 12,
        n_motion: int = 4,
        n_session: int = 2,
        n_phase_nodes: int = 8,
        seed: int = 0,
    ) -> "SceneModel":
        """Build a scene over the procedural test template.

        Named ``procedural`` for the same reason the dataset builder is: this is
        a test fixture, not a frozen clinical template.
        """

        complex_ = make_toy_complex(n_annulus=n_annulus)
        basis = smooth_shape_basis(complex_.global_vertices, n_modes)
        rng = np.random.default_rng(seed)
        # Motion occupies the leading modes; the session subspace is deflated off
        # it so assumption A3 holds by construction rather than by assertion.
        motion_samples = rng.normal(size=(6 * n_modes, n_modes))
        motion_samples[:, n_motion:] *= 0.05
        session_samples = rng.normal(size=(6 * n_modes, n_modes))
        bases = GaugedLatentBases.from_samples(
            motion_samples, session_samples, n_motion=n_motion, n_session=n_session
        )
        return cls(
            template_physical=complex_.global_vertices,
            attachment_map=complex_.attachment_map,
            groups=complex_.groups,
            shape_basis=basis,
            bases=bases,
            n_phase_nodes=n_phase_nodes,
        )

    @property
    def n_physical(self) -> int:
        return len(self.template_physical)

    @property
    def n_modes(self) -> int:
        return self.shape_basis.shape[1]

    def physical_coordinates(self, latent: np.ndarray) -> np.ndarray:
        """Map reduced latent coordinates to physical vertices ``q``."""

        values = np.asarray(latent, dtype=np.float64)
        if values.shape[-1] != self.n_modes:
            raise ValueError(f"latent must end with dimension {self.n_modes}")
        displacement = values @ self.shape_basis.T
        return self.template_physical + displacement.reshape(
            *values.shape[:-1], self.n_physical, 3
        )

    def assemble(self, latent: np.ndarray) -> dict[str, np.ndarray]:
        """Local component coordinates; attachment holds exactly by construction."""

        return self.attachment_map.assemble(self.physical_coordinates(latent))

    def attachment_residual(self, latent: np.ndarray) -> float:
        """The primary interface endpoint: max duplicated-vertex disagreement."""

        return self.attachment_map.max_residual(self.assemble(latent))


@dataclass(frozen=True)
class SceneParameters:
    """Every latent quantity the joint solver estimates."""

    pose_axis_angle: np.ndarray
    pose_translation: np.ndarray
    phase_offset: float
    phase_logits: np.ndarray
    identity: np.ndarray
    motion: np.ndarray
    session: dict[str, np.ndarray]

    def transform(self) -> RigidTransform:
        return RigidTransform.from_axis_angle(self.pose_axis_angle, self.pose_translation)

    def warp(self) -> MonotoneCircularPhaseWarp:
        return MonotoneCircularPhaseWarp(self.phase_logits, offset=self.phase_offset)
