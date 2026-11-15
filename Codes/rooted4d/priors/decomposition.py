r"""Gauge-fixed identity / shared-motion / session-discrepancy decomposition.

Specification section 4.2 writes each session trajectory in the constrained
latent space as

.. math::
   z_{im}(\tau)=z_i^{\mathrm{id}}+U_h h_i(\tau)+U_s r_{im}(\tau),

subject to the two centering gauges

.. math::
   \int_0^1 h_i(\tau)\,d\tau=0,
   \qquad
   \sum_m w_{im} r_{im}(\tau)=0 \ \ \text{for every } \tau .

Theorem 3 states that these gauges make ``(z^id, h, r_m)`` unique given the
trajectories and the fixed block-orthogonal bases.  This module implements the
canonical decomposition constructively, in the order used by the proof:

1. the weighted cross-session mean removes the session block entirely, because
   the session gauge annihilates it;
2. the cycle mean of that average is the identity, because the motion gauge
   removes ``U_h h``;
3. the motion coefficients are the ``U_h`` projection of the remainder;
4. each session coefficient is the ``U_s`` projection of that session's
   deviation from the cross-session mean, which satisfies the session gauge by
   construction.

Real trajectories need not lie in ``span(U_h) ⊕ span(U_s)``.  What the two
projections cannot represent is returned explicitly as unmodelled residual
rather than silently absorbed, so a subspace that is too small is visible in
the report instead of appearing as a good fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .gauges import GaugedLatentBases


def _normalized_weights(
    weights: Mapping[str, float] | None,
    sessions: tuple[str, ...],
) -> dict[str, float]:
    if weights is None:
        return {name: 1.0 / len(sessions) for name in sessions}
    if set(weights) != set(sessions):
        raise ValueError(
            f"weights must cover exactly the sessions {sorted(sessions)}; "
            f"received {sorted(weights)}"
        )
    values = {name: float(weights[name]) for name in sessions}
    if any(value <= 0.0 for value in values.values()):
        raise ValueError("assumption A3 requires strictly positive session weights")
    total = sum(values.values())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("session weights must sum to a positive finite value")
    return {name: value / total for name, value in values.items()}


def _trajectory_stack(
    trajectories: Mapping[str, np.ndarray],
    latent_dimension: int,
) -> tuple[tuple[str, ...], dict[str, np.ndarray], int]:
    if len(trajectories) < 2:
        raise ValueError("at least two sessions are required for a cross-session gauge")
    sessions = tuple(sorted(trajectories))
    stacked: dict[str, np.ndarray] = {}
    n_phases: int | None = None
    for name in sessions:
        array = np.asarray(trajectories[name], dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != latent_dimension:
            raise ValueError(
                f"session '{name}' must have shape (phases, {latent_dimension}); "
                f"received {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"session '{name}' contains non-finite values")
        if n_phases is None:
            n_phases = array.shape[0]
        elif array.shape[0] != n_phases:
            raise ValueError(
                "every session must be resampled onto the same phase grid before "
                "the cross-session gauge is applied"
            )
        stacked[name] = array
    assert n_phases is not None
    if n_phases < 2:
        raise ValueError("at least two phase samples are required")
    return sessions, stacked, n_phases


@dataclass(frozen=True)
class GaugedDecomposition:
    """Canonical section 4.2 decomposition of one subject's trajectories."""

    identity: np.ndarray
    motion: np.ndarray
    session: dict[str, np.ndarray]
    weights: dict[str, float]
    unmodelled_motion: np.ndarray
    unmodelled_session: dict[str, np.ndarray]

    @property
    def sessions(self) -> tuple[str, ...]:
        return tuple(sorted(self.session))

    @property
    def max_unmodelled(self) -> float:
        """Largest latent component the two subspaces could not represent."""

        values = [float(np.max(np.abs(self.unmodelled_motion)))]
        values.extend(
            float(np.max(np.abs(residual))) for residual in self.unmodelled_session.values()
        )
        return max(values)

    def gauge_residuals(self) -> dict[str, float]:
        """Numerical violation of the two centering gauges of section 4.2."""

        cycle_mean = np.max(np.abs(self.motion.mean(axis=0)))
        weighted = sum(
            self.weights[name] * self.session[name] for name in self.sessions
        )
        return {
            "motion_cycle_mean": float(cycle_mean),
            "session_cross_session_mean": float(np.max(np.abs(weighted))),
        }

    def as_dict(self) -> dict[str, float | int]:
        residuals = self.gauge_residuals()
        return {
            "n_sessions": len(self.session),
            "n_motion_modes": int(self.motion.shape[1]),
            "n_session_modes": int(next(iter(self.session.values())).shape[1]),
            "max_unmodelled": self.max_unmodelled,
            **residuals,
        }


class SessionDecomposition:
    """Compose and canonically decompose gauge-fixed latent trajectories.

    The phase grid is assumed to sample one complete cycle at uniform spacing,
    ``tau_j = j / T`` for ``j = 0..T-1``.  For that grid the sample mean is an
    exact quadrature of the cycle integral for every Fourier mode below the
    Nyquist harmonic, so the motion gauge of section 4.2 is enforced exactly for
    the periodic bases of section 4.3 rather than only approximately.
    """

    def __init__(
        self,
        bases: GaugedLatentBases,
        *,
        weights: Mapping[str, float] | None = None,
        sessions: tuple[str, ...] = ("cmr", "tee"),
        basis_tolerance: float = 1e-9,
    ) -> None:
        bases.validate(basis_tolerance).raise_if_invalid()
        if len(set(sessions)) != len(sessions) or len(sessions) < 2:
            raise ValueError("sessions must be at least two distinct names")
        self.bases = bases
        self.sessions = tuple(sorted(sessions))
        self.weights = _normalized_weights(weights, self.sessions)

    @property
    def latent_dimension(self) -> int:
        return self.bases.latent_dimension

    def compose(
        self,
        identity: np.ndarray,
        motion: np.ndarray,
        session: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Evaluate ``z_m(tau) = a + U_h h(tau) + U_s r_m(tau)``.

        The arguments need not satisfy the gauges; this is the forward map used
        to build the admissible reparameterizations that Theorem 3 must collapse.
        """

        anatomy = np.asarray(identity, dtype=np.float64)
        if anatomy.shape != (self.latent_dimension,):
            raise ValueError(f"identity must have shape ({self.latent_dimension},)")
        shared = np.asarray(motion, dtype=np.float64)
        if shared.ndim != 2 or shared.shape[1] != self.bases.n_motion:
            raise ValueError(f"motion must have shape (phases, {self.bases.n_motion})")
        if set(session) != set(self.sessions):
            raise ValueError(f"session must cover exactly {list(self.sessions)}")
        common = anatomy + shared @ self.bases.motion.T
        result: dict[str, np.ndarray] = {}
        for name in self.sessions:
            residual = np.asarray(session[name], dtype=np.float64)
            if residual.shape != (shared.shape[0], self.bases.n_session):
                raise ValueError(
                    f"session '{name}' must have shape "
                    f"({shared.shape[0]}, {self.bases.n_session})"
                )
            result[name] = common + residual @ self.bases.session.T
        return result

    def decompose(self, trajectories: Mapping[str, np.ndarray]) -> GaugedDecomposition:
        """Return the unique gauge-satisfying decomposition of Theorem 3."""

        sessions, stacked, _ = _trajectory_stack(trajectories, self.latent_dimension)
        if sessions != self.sessions:
            raise ValueError(
                f"trajectories must cover exactly {list(self.sessions)}; "
                f"received {list(sessions)}"
            )
        # Step 1: the session gauge annihilates U_s in the weighted mean.
        common = sum(self.weights[name] * stacked[name] for name in sessions)
        # Step 2: the motion gauge makes the cycle mean the canonical identity.
        identity = common.mean(axis=0)
        centered = common - identity
        # Step 3: shared motion is the U_h projection of what remains.
        motion = centered @ self.bases.motion
        unmodelled_motion = centered - motion @ self.bases.motion.T
        # Step 4: each session coefficient is the U_s projection of its deviation
        # from the weighted mean, which satisfies the session gauge identically.
        session: dict[str, np.ndarray] = {}
        unmodelled_session: dict[str, np.ndarray] = {}
        for name in sessions:
            deviation = stacked[name] - common
            coefficients = deviation @ self.bases.session
            session[name] = coefficients
            unmodelled_session[name] = deviation - coefficients @ self.bases.session.T
        return GaugedDecomposition(
            identity=identity,
            motion=motion,
            session=session,
            weights=dict(self.weights),
            unmodelled_motion=unmodelled_motion,
            unmodelled_session=unmodelled_session,
        )

    def canonicalize(
        self,
        identity: np.ndarray,
        motion: np.ndarray,
        session: Mapping[str, np.ndarray],
    ) -> GaugedDecomposition:
        """Map any parameterization to the canonical one generating the same trajectories."""

        return self.decompose(self.compose(identity, motion, session))

    def reparameterize(
        self,
        identity: np.ndarray,
        motion: np.ndarray,
        session: Mapping[str, np.ndarray],
        *,
        motion_shift: np.ndarray,
        session_shift: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Apply the two exact gauge ambiguities named in section 4.2.

        ``a -> a + U_h b`` with ``h -> h - b`` is the motion ambiguity removed by
        cycle centering.  ``a -> a + U_s c`` with ``r_m -> r_m - c`` for a
        constant ``c`` is the session ambiguity removed by cross-session
        centering.  Both leave every ``z_m(tau)`` bitwise unchanged up to
        floating-point rounding, so a correct implementation of Theorem 3 must
        return the same canonical decomposition before and after.
        """

        shift_h = np.asarray(motion_shift, dtype=np.float64)
        shift_s = np.asarray(session_shift, dtype=np.float64)
        if shift_h.shape != (self.bases.n_motion,):
            raise ValueError(f"motion_shift must have shape ({self.bases.n_motion},)")
        if shift_s.shape != (self.bases.n_session,):
            raise ValueError(f"session_shift must have shape ({self.bases.n_session},)")
        new_identity = (
            np.asarray(identity, dtype=np.float64)
            + self.bases.motion @ shift_h
            + self.bases.session @ shift_s
        )
        new_motion = np.asarray(motion, dtype=np.float64) - shift_h
        new_session = {
            name: np.asarray(session[name], dtype=np.float64) - shift_s
            for name in self.sessions
        }
        return new_identity, new_motion, new_session
