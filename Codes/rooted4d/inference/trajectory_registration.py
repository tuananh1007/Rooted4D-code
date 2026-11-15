r"""Trajectory registration of two independent reconstructions.

Late fusion as first implemented aligns its two reconstructions at a single
phase, after a separate feature-based temporal alignment, so it never uses the
motion both fits estimate. This module registers the whole trajectories
instead, with the scene's own alignment parameterization: a rigid map, a phase
offset and a monotone circular warp, fitted by the solver's own
Levenberg--Marquardt after the same 24-offset coarse search the joint fit
starts from. It is the registration step of exact repair (``repair.py``),
extended by the warp and run once, and it is reported as a baseline of its own
("late fusion, trajectory registration"): a baseline registered more weakly
than the method's own repair step would make the comparison unfair.

Conventions. The moving trajectory is the TEE fit's session-free anatomy in its
own canonical frame at its own phases ``s``; the reference is the CMR fit's
session-free anatomy in the CMR frame. The registration finds ``G`` (TEE-fit
frame to CMR frame) and ``chi`` (TEE-fit phase to CMR phase) such that
``G x_T(s) ~ x_C(chi(s))``. The TEE acquisition is then placed in the CMR frame
by ``P_T o G^-1``, and its time-to-phase map is ``chi o phi_T``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from rooted4d.geometry.rigid import RigidTransform
from rooted4d.motion import MonotoneCircularPhaseWarp

from .joint import SolverConfig, _levenberg_marquardt, latent_trajectory
from .scene import SceneModel, SceneParameters

#: Phases at which the trajectories are compared, and at which the interface gap
#: is measured, as for every other arm.
REGISTRATION_PHASES = np.linspace(0.0, 1.0, 32, endpoint=False)
#: The coarse phase-offset search of the joint fit's stage A, and its budget.
OFFSET_CANDIDATES = 24
COARSE_ITERATIONS = 8


def session_free_trajectory(
    parameters: SceneParameters,
    model: SceneModel,
    config: SolverConfig,
    phases: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    """Session-free coordinates of ``vertices``, shape (phases, vertices, 3)."""

    shared = replace(config, use_session=False)
    tau = np.mod(np.asarray(phases, dtype=np.float64), 1.0)
    latent = latent_trajectory(parameters, tau, "cmr", model, shared)
    return model.physical_coordinates(latent)[:, model.observation_rows(np.asarray(vertices)), :]


def inverse_phase(warp: MonotoneCircularPhaseWarp, phases: np.ndarray) -> np.ndarray:
    """The source phases that ``warp`` maps onto ``phases``."""

    unwrapped = np.mod(np.asarray(phases, dtype=np.float64) - warp.offset, 1.0)
    return np.interp(unwrapped, warp.target_knots, warp.source_knots)


@dataclass(frozen=True)
class TrajectoryRegistration:
    """``G`` (moving frame to reference frame) and ``chi`` (moving to reference phase)."""

    transform: RigidTransform
    warp: MonotoneCircularPhaseWarp
    #: Root-mean-square point distance after registration, over every
    #: registration vertex at every phase.
    rms_mm: float
    cost: float
    iterations: int

    def place(self, pose: RigidTransform) -> RigidTransform:
        """The moving acquisition's pose in the reference frame, ``pose o G^-1``."""

        inverse = self.transform.inverse()
        return RigidTransform(pose.rotation @ inverse.rotation,
                              pose.rotation @ inverse.translation + pose.translation)


def register_trajectory_arrays(
    reference: Callable[[np.ndarray], np.ndarray],
    moving: np.ndarray,
    config: SolverConfig,
    *,
    knots: int,
    phases: np.ndarray = REGISTRATION_PHASES,
    offset_candidates: int = OFFSET_CANDIDATES,
) -> TrajectoryRegistration:
    """Register ``moving`` (points at ``phases``) onto the ``reference`` trajectory.

    ``reference(tau)`` returns the reference points at arbitrary phases, with
    the same vertex order as ``moving``. The stages mirror the joint fit: a
    coarse search over ``offset_candidates`` phase offsets with the warp shape
    frozen (``COARSE_ITERATIONS`` iterations each), then the rigid map and the
    offset from the best candidate, then the warp shape as well, which is
    penalized for acceleration exactly as the joint fit penalizes its own.
    """

    phases = np.asarray(phases, dtype=np.float64)
    flat_moving = np.asarray(moving, dtype=np.float64).reshape(-1, 3)
    frozen = np.zeros(knots)
    smoothness = np.sqrt(config.phase_smoothness_weight)

    def unpack(vector: np.ndarray) -> tuple[RigidTransform, MonotoneCircularPhaseWarp]:
        logits = vector[7:] if vector.size > 7 else frozen
        return (RigidTransform.from_axis_angle(vector[:3], vector[3:6]),
                MonotoneCircularPhaseWarp(logits, offset=float(vector[6])))

    def data(vector: np.ndarray) -> np.ndarray:
        transform, warp = unpack(vector)
        target = reference(warp.evaluate(phases)).reshape(-1, 3)
        return (transform.apply(flat_moving) - target).reshape(-1)

    def residual(vector: np.ndarray) -> np.ndarray:
        if vector.size == 7:
            return data(vector)
        increments = unpack(vector)[1].increments
        return np.concatenate([data(vector),
                               smoothness * np.diff(increments) * len(increments)])

    coarse = replace(config, max_iterations=COARSE_ITERATIONS)
    best: tuple[float, np.ndarray] | None = None
    for offset in np.linspace(0.0, 1.0, offset_candidates, endpoint=False):
        solution, cost, _ = _levenberg_marquardt(
            residual, np.concatenate([np.zeros(6), [offset]]), coarse)
        if best is None or cost < best[0]:
            best = (cost, solution)
    assert best is not None

    vector, _, first = _levenberg_marquardt(residual, best[1], config)
    vector, cost, second = _levenberg_marquardt(
        residual, np.concatenate([vector, frozen]), config)
    transform, warp = unpack(vector)
    distances = data(vector).reshape(-1, 3)
    return TrajectoryRegistration(
        transform=transform,
        warp=warp,
        rms_mm=float(np.sqrt((distances ** 2).sum(axis=1).mean())),
        cost=float(cost),
        iterations=int(first + second),
    )


def register_trajectories(
    reference: SceneParameters,
    moving: SceneParameters,
    model: SceneModel,
    config: SolverConfig,
    vertices: np.ndarray,
    *,
    phases: np.ndarray = REGISTRATION_PHASES,
) -> TrajectoryRegistration:
    """Register the ``moving`` fit's trajectory onto the ``reference`` fit's."""

    fixed = session_free_trajectory(moving, model, config, phases, vertices)
    return register_trajectory_arrays(
        lambda tau: session_free_trajectory(reference, model, config, tau, vertices),
        fixed, config, knots=model.n_phase_nodes, phases=phases)


def registered_interface_gap(
    reference: SceneParameters,
    moving: SceneParameters,
    registration: TrajectoryRegistration,
    model: SceneModel,
    config: SolverConfig,
    phases: np.ndarray = REGISTRATION_PHASES,
) -> float:
    """Largest annulus coordinate discrepancy over the cycle, in the reference frame.

    Measured at the same 32 reference phases, on the same vertices and with the
    same maximum-coordinate norm as every other arm's interface endpoint; the
    moving fit is read at the phases the registration maps onto them.
    """

    annulus = np.sort(model.groups["annulus"])
    chamber = session_free_trajectory(reference, model, config, phases, annulus)
    valve = registration.transform.apply(session_free_trajectory(
        moving, model, config, inverse_phase(registration.warp, phases), annulus))
    return float(np.abs(valve - chamber).max())
