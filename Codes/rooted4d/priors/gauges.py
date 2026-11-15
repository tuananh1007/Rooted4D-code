r"""Block-orthogonal latent bases for the shared-motion and session subspaces.

Specification section 4.2 requires two fixed learned subspaces with

.. math::
   U_h^\top U_h = I,\qquad
   U_s^\top U_s = I,\qquad
   U_h^\top U_s = 0.

Assumption A3 of the theorem stack calls these bases *whitened* and
*block-orthogonal*.  Theorem 3 fails without them: if the two column spaces
overlap, a shared-motion component and a session-discrepancy component become
algebraically interchangeable and the decomposition is no longer unique.

This module constructs and validates those bases.  It does not claim that the
learned directions correspond to identifiable physiology; the session subspace
is an explicitly regularized discrepancy space, as stated in section 4.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sample_matrix(samples: np.ndarray, name: str, latent_dimension: int) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != latent_dimension:
        raise ValueError(f"{name} must have shape (samples, {latent_dimension})")
    if array.shape[0] == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be non-empty and finite")
    return array


def _leading_orthonormal_basis(
    samples: np.ndarray,
    n_components: int,
    *,
    name: str,
    rtol: float,
) -> np.ndarray:
    """Return the leading ``n_components`` right singular vectors as columns."""

    _, singular_values, right = np.linalg.svd(samples, full_matrices=False)
    available = int(
        np.count_nonzero(singular_values > rtol * max(float(singular_values[0]), 0.0))
    ) if singular_values.size else 0
    if available < n_components:
        raise ValueError(
            f"{name} spans only {available} numerically independent directions; "
            f"{n_components} were requested"
        )
    return right[:n_components].T.copy()


@dataclass(frozen=True)
class BasisValidation:
    """Orthonormality and block-orthogonality residuals of a basis pair."""

    motion_orthonormality: float
    session_orthonormality: float
    block_orthogonality: float
    tolerance: float

    @property
    def valid(self) -> bool:
        return max(
            self.motion_orthonormality,
            self.session_orthonormality,
            self.block_orthogonality,
        ) <= self.tolerance

    def raise_if_invalid(self) -> None:
        if not self.valid:
            raise ValueError(
                "Assumption A3 violated: "
                f"motion orthonormality {self.motion_orthonormality:.3e}, "
                f"session orthonormality {self.session_orthonormality:.3e}, "
                f"block orthogonality {self.block_orthogonality:.3e} "
                f"(tolerance {self.tolerance:.3e})"
            )

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "motion_orthonormality": self.motion_orthonormality,
            "session_orthonormality": self.session_orthonormality,
            "block_orthogonality": self.block_orthogonality,
            "tolerance": self.tolerance,
            "valid": self.valid,
        }


@dataclass(frozen=True)
class GaugedLatentBases:
    """Whitened, mutually orthogonal shared-motion and session subspaces."""

    motion: np.ndarray
    session: np.ndarray

    def __post_init__(self) -> None:
        motion = np.asarray(self.motion, dtype=np.float64)
        session = np.asarray(self.session, dtype=np.float64)
        for array, name in ((motion, "motion"), (session, "session")):
            if array.ndim != 2 or array.shape[1] == 0:
                raise ValueError(f"{name} basis must have shape (latent, components >= 1)")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} basis must be finite")
        if motion.shape[0] != session.shape[0]:
            raise ValueError("both bases must share the same ambient latent dimension")
        if motion.shape[1] + session.shape[1] > motion.shape[0]:
            raise ValueError(
                "the two blocks cannot be mutually orthogonal: their combined rank "
                f"{motion.shape[1] + session.shape[1]} exceeds the latent dimension "
                f"{motion.shape[0]}"
            )
        object.__setattr__(self, "motion", motion)
        object.__setattr__(self, "session", session)

    @classmethod
    def from_samples(
        cls,
        motion_samples: np.ndarray,
        session_samples: np.ndarray,
        *,
        n_motion: int,
        n_session: int,
        rtol: float = 1e-10,
    ) -> "GaugedLatentBases":
        """Learn ``U_h`` from motion samples, then ``U_s`` from deflated session samples.

        The session samples are projected off ``im(U_h)`` before their own
        subspace is extracted.  Section 4.2 requires this ordering: shared
        periodic motion is the reference block, and the session subspace only
        receives what shared motion cannot already express.
        """

        if n_motion < 1 or n_session < 1:
            raise ValueError("n_motion and n_session must both be at least one")
        if rtol <= 0.0:
            raise ValueError("rtol must be positive")
        motion_array = np.asarray(motion_samples, dtype=np.float64)
        if motion_array.ndim != 2:
            raise ValueError("motion_samples must have shape (samples, latent)")
        latent_dimension = motion_array.shape[1]
        motion_array = _sample_matrix(motion_array, "motion_samples", latent_dimension)
        session_array = _sample_matrix(session_samples, "session_samples", latent_dimension)

        motion_basis = _leading_orthonormal_basis(
            motion_array, n_motion, name="motion_samples", rtol=rtol
        )
        deflated = session_array - (session_array @ motion_basis) @ motion_basis.T
        session_basis = _leading_orthonormal_basis(
            deflated, n_session, name="session_samples after motion deflation", rtol=rtol
        )
        return cls(motion=motion_basis, session=session_basis)

    @property
    def latent_dimension(self) -> int:
        return self.motion.shape[0]

    @property
    def n_motion(self) -> int:
        return self.motion.shape[1]

    @property
    def n_session(self) -> int:
        return self.session.shape[1]

    def validate(self, tolerance: float = 1e-9) -> BasisValidation:
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        motion_gram = self.motion.T @ self.motion - np.eye(self.n_motion)
        session_gram = self.session.T @ self.session - np.eye(self.n_session)
        cross = self.motion.T @ self.session
        return BasisValidation(
            motion_orthonormality=float(np.max(np.abs(motion_gram))),
            session_orthonormality=float(np.max(np.abs(session_gram))),
            block_orthogonality=float(np.max(np.abs(cross))) if cross.size else 0.0,
            tolerance=float(tolerance),
        )
