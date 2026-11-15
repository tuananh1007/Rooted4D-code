r"""Independent fitting followed by interface repair.

The competitor the reassessment names for the unknown-alignment comparison:
reconstruct each modality independently, then make the two reconstructions
agree at the interface. Joint fitting has to beat this, because it keeps what
independent fitting is good at -- each modality's anatomy fitted to its own
evidence -- and still ends attached.

Both arms share the first three steps:

1. **Independent fits.** CMR in the reference frame, TEE with its own pose and
   phase, both session-free and from the shared start. Each modality has its
   own identity-prior weight, which is tuned separately.
2. **Registration of the two rings.** The annulus trajectory of each fit is
   evaluated over the cycle. The TEE ring is brought onto the CMR ring by a
   phase shift and a rigid map, chosen by grid search and Kabsch over every
   ring vertex at every phase. Both inputs are reconstructions; no truth is
   used. This is needed because a TEE-only fit fixes neither its phase origin
   nor its frame relative to CMR.
3. **Consensus.** The equal-weight average of the two registered rings,
   projected onto the ring trajectories the model can represent.

They then differ in how the consensus is imposed.

**Exact repair** refits each modality separately, on its own evidence, with
its shape restricted to the affine set whose ring trajectory *is* the
consensus: ``v = v* + N u``, with ``N`` spanning the shape directions that do
not move the ring. Equality at eight phases is equality over the whole cycle
at Fourier order three. What this leaves free depends on the representation,
and that is a finding, not a defect. The structured basis leaves its 36
interior modes free. Global PCA at rank 48 leaves none -- its ring map has full
rank -- so its exact repair is the ring-determined anatomy, and each
modality's interior evidence is overwritten.

**Penalty repair** makes one joint refit against both modalities' evidence,
with the ring pulled toward the consensus by a penalty. It fixes the pilot's
known defect: the pilot evaluated the ring under the CMR session, whereas here
it is evaluated on the session-free shared anatomy, which is what the
consensus is defined on. The penalty grid must exclude values near zero,
because as the penalty vanishes this arm becomes the joint fit it is supposed
to compete with.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .initialization import StartingAlignment, _matrix_to_axis_angle, start_from
from .joint import (
    Observations,
    SceneFit,
    SolverConfig,
    _levenberg_marquardt,
    _pack,
    _residuals,
    _unpack,
    fit_joint_scene,
    latent_trajectory,
)
from .phase_pose import kabsch
from .scene import SceneModel, SceneParameters

#: Phases at which rings are compared and enforced. Eight exceeds 2 * 3 + 1, so
#: at Fourier order three agreement here is agreement over the whole cycle.
RING_PHASES = np.linspace(0.0, 1.0, 8, endpoint=False)
#: Coarse phase-shift grid for ring registration, refined by golden section.
SHIFT_GRID = np.linspace(0.0, 1.0, 96, endpoint=False)


def _ring_rows(model: SceneModel) -> np.ndarray:
    return model.observation_rows(np.sort(model.groups["annulus"]))


def ring_trajectory(
    parameters: SceneParameters,
    model: SceneModel,
    config: SolverConfig,
    phases: np.ndarray = RING_PHASES,
) -> np.ndarray:
    """Session-free annulus coordinates, shape (phases, ring vertices, 3)."""

    shared = replace(config, use_session=False)
    latent = latent_trajectory(parameters, np.asarray(phases, dtype=np.float64),
                               "cmr", model, shared)
    return model.physical_coordinates(latent)[:, _ring_rows(model), :]


def _shape_vector(parameters: SceneParameters) -> np.ndarray:
    return np.concatenate([parameters.identity, parameters.motion.reshape(-1)])


def _with_shape(parameters: SceneParameters, vector: np.ndarray) -> SceneParameters:
    n = parameters.identity.size
    return replace(parameters, identity=vector[:n].copy(),
                   motion=vector[n:].reshape(parameters.motion.shape).copy())


def ring_map(model: SceneModel, config: SolverConfig, template: SceneParameters):
    """The ring trajectory as an affine function of the shape vector.

    Returns ``(offset, L)`` with ``ring(v) = offset + L v``, flattened over
    phases, ring vertices and coordinates. Exact, because the session-free
    latent trajectory is linear in identity and motion.
    """

    zero = _with_shape(template, np.zeros(_shape_vector(template).size))
    offset = ring_trajectory(zero, model, config).reshape(-1)
    size = _shape_vector(template).size
    columns = np.empty((offset.size, size))
    for j in range(size):
        unit = np.zeros(size)
        unit[j] = 1.0
        columns[:, j] = ring_trajectory(_with_shape(template, unit), model, config).reshape(-1) - offset
    return offset, columns


def _registration_cost(reference: np.ndarray, moving: np.ndarray):
    transform = kabsch(moving.reshape(-1, 3), reference.reshape(-1, 3))
    aligned = transform.apply(moving.reshape(-1, 3)).reshape(moving.shape)
    return float(((aligned - reference) ** 2).sum()), transform, aligned


def register_rings(
    cmr: SceneParameters, tee: SceneParameters, model: SceneModel, config: SolverConfig
):
    """Phase shift and rigid map taking the TEE fit's ring onto the CMR fit's.

    The shift ``delta`` is such that the TEE fit's anatomy at its own phase
    ``tau + delta`` is compared with the CMR fit's anatomy at ``tau``.
    """

    reference = ring_trajectory(cmr, model, config)

    def cost(delta: float) -> float:
        moving = ring_trajectory(tee, model, config, (RING_PHASES + delta) % 1.0)
        return _registration_cost(reference, moving)[0]

    costs = np.array([cost(d) for d in SHIFT_GRID])
    step = SHIFT_GRID[1] - SHIFT_GRID[0]
    lo, hi = SHIFT_GRID[int(np.argmin(costs))] - step, SHIFT_GRID[int(np.argmin(costs))] + step
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = cost(a), cost(b)
    for _ in range(40):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = cost(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = cost(b)
    delta = float(((a + b) / 2.0) % 1.0)
    moving = ring_trajectory(tee, model, config, (RING_PHASES + delta) % 1.0)
    residual, transform, aligned = _registration_cost(reference, moving)
    return delta, transform, reference, aligned, residual


@dataclass(frozen=True)
class Consensus:
    target: np.ndarray            # representable consensus ring, flattened
    shape: np.ndarray             # a shape vector that produces it
    offset: np.ndarray            # ring map offset
    matrix: np.ndarray            # ring map L
    null_space: np.ndarray        # shape directions that do not move the ring
    delta: float
    transform: Any
    registration_rms_mm: float
    projection_rms_mm: float


def build_consensus(cmr_fit, tee_fit, model: SceneModel, config: SolverConfig) -> Consensus:
    delta, transform, reference, aligned, residual = register_rings(
        cmr_fit.parameters, tee_fit.parameters, model, config)
    average = 0.5 * reference + 0.5 * aligned
    offset, matrix = ring_map(model, config, cmr_fit.parameters)
    shape, *_ = np.linalg.lstsq(matrix, average.reshape(-1) - offset, rcond=None)
    target = offset + matrix @ shape
    _, singular, right = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.count_nonzero(singular > 1e-10 * singular[0]))
    null_space = right[rank:].T.copy()
    count = reference.shape[0] * reference.shape[1]
    return Consensus(
        target=target, shape=shape, offset=offset, matrix=matrix, null_space=null_space,
        delta=delta, transform=transform,
        registration_rms_mm=float(np.sqrt(residual / count)),
        projection_rms_mm=float(np.sqrt(((target - average.reshape(-1)) ** 2).sum() / count)),
    )


def independent_fits(
    observations: Observations,
    model: SceneModel,
    config: SolverConfig,
    alignment: StartingAlignment | None,
    modality_overrides: dict | None = None,
):
    """Session-free CMR-only and TEE-only fits, each with its own overrides."""

    overrides = modality_overrides or {}
    cmr_config = replace(config, use_cmr=True, use_tee=False, estimate_pose=False,
                         estimate_phase_warp=False, use_session=False,
                         **overrides.get("cmr", {}))
    tee_config = replace(config, use_cmr=False, use_tee=True, estimate_pose=True,
                         estimate_phase_warp=True, use_session=False,
                         **overrides.get("tee", {}))

    def start(arm_config):
        return None if alignment is None else start_from(alignment, model, arm_config)

    cmr_fit = fit_joint_scene(observations, model, cmr_config, start=start(cmr_config))
    tee_fit = fit_joint_scene(observations, model, tee_config, start=start(tee_config))
    return cmr_fit, tee_fit, cmr_config, tee_config


def _refit_on_affine_set(
    observations, model, config, template: SceneParameters, consensus: Consensus
):
    """Refit every non-shape parameter and the free shape directions ``u``."""

    base = consensus.shape
    null = consensus.null_space
    packed = _pack(template, model, config)
    shape_size = base.size

    def expand(reduced: np.ndarray) -> np.ndarray:
        full = packed.copy()
        full[:shape_size] = base + (null @ reduced[: null.shape[1]] if null.size else 0.0)
        full[shape_size:] = reduced[null.shape[1]:]
        return full

    reduced = np.concatenate([np.zeros(null.shape[1]), packed[shape_size:]])

    def residual_function(value: np.ndarray) -> np.ndarray:
        return _residuals(expand(value), template, observations, model, config)

    if reduced.size:
        reduced, cost, iterations = _levenberg_marquardt(residual_function, reduced, config)
    else:
        cost, iterations = float(residual_function(reduced) @ residual_function(reduced)), 0
    parameters = _unpack(expand(reduced), template, model, config)
    return parameters, float(cost), int(iterations)


def _scene_fit(parameters, observations, model, config, cost, iterations) -> SceneFit:
    return SceneFit(parameters=parameters, cost=cost, iterations=iterations, stage_costs=(),
                    attachment_residual=0.0, packed=_pack(parameters, model, config),
                    model=model, config=config, observations=observations)


def exact_repair(
    observations: Observations,
    model: SceneModel,
    config: SolverConfig,
    *,
    alignment: StartingAlignment | None,
    modality_overrides: dict | None = None,
) -> dict[str, Any]:
    cmr_fit, tee_fit, cmr_config, tee_config = independent_fits(
        observations, model, config, alignment, modality_overrides)
    consensus = build_consensus(cmr_fit, tee_fit, model, cmr_config)

    cmr_parameters, cmr_cost, cmr_iterations = _refit_on_affine_set(
        observations, model, cmr_config, cmr_fit.parameters, consensus)

    # The TEE refit's anatomy is now expressed in the reference frame and phase,
    # so its pose and phase origin are moved accordingly before refitting:
    # canonical_TEE ~ T^-1(reference), so the pose becomes P_T o T^-1, and the
    # TEE fit's phase tau + delta is reference phase tau.
    tee_pose = tee_fit.parameters.transform()
    inverse = consensus.transform.inverse()
    rotation = tee_pose.rotation @ inverse.rotation
    translation = tee_pose.rotation @ inverse.translation + tee_pose.translation
    tee_start = replace(
        tee_fit.parameters,
        pose_axis_angle=_matrix_to_axis_angle(rotation),
        pose_translation=np.asarray(translation, dtype=np.float64),
        phase_offset=float((tee_fit.parameters.phase_offset - consensus.delta) % 1.0),
    )
    tee_parameters, tee_cost, tee_iterations = _refit_on_affine_set(
        observations, model, tee_config, tee_start, consensus)

    phases = np.linspace(0.0, 1.0, 32, endpoint=False)
    gap = float(np.abs(ring_trajectory(cmr_parameters, model, cmr_config, phases)
                       - ring_trajectory(tee_parameters, model, tee_config, phases)).max())
    # Identity directions alone, with motion held, are the count the
    # pre-registration names (0 for global PCA at rank 48, 36 for structured).
    identity = consensus.matrix[:, : cmr_fit.parameters.identity.size]
    singular = np.linalg.svd(consensus.matrix, compute_uv=False)
    identity_rank = np.linalg.matrix_rank(identity, tol=1e-10 * singular[0])
    return {
        "cmr": _scene_fit(cmr_parameters, observations, model, cmr_config, cmr_cost, cmr_iterations),
        "tee": _scene_fit(tee_parameters, observations, model, tee_config, tee_cost, tee_iterations),
        "interface_gap_mm": gap,
        "free_shape_directions": int(consensus.null_space.shape[1]),
        "free_identity_directions": int(identity.shape[1] - identity_rank),
        "registration_rms_mm": consensus.registration_rms_mm,
        "projection_rms_mm": consensus.projection_rms_mm,
        "iterations": int(cmr_fit.iterations + tee_fit.iterations + cmr_iterations + tee_iterations),
    }


def penalty_repair(
    observations: Observations,
    model: SceneModel,
    config: SolverConfig,
    *,
    alignment: StartingAlignment | None,
    ring_weight: float,
    modality_overrides: dict | None = None,
) -> dict[str, Any]:
    cmr_fit, tee_fit, cmr_config, _ = independent_fits(
        observations, model, config, alignment, modality_overrides)
    consensus = build_consensus(cmr_fit, tee_fit, model, cmr_config)

    # One joint refit from the registered TEE alignment, the CMR fit's anatomy
    # and zero session, with the session-free ring pulled to the consensus.
    tee_pose = tee_fit.parameters.transform()
    inverse = consensus.transform.inverse()
    start = start_from(StartingAlignment(
        _matrix_to_axis_angle(tee_pose.rotation @ inverse.rotation),
        tee_pose.rotation @ inverse.translation + tee_pose.translation,
        float((tee_fit.parameters.phase_offset - consensus.delta) % 1.0), 0, 0.0),
        model, config)
    start = replace(start, identity=cmr_fit.parameters.identity.copy(),
                    motion=cmr_fit.parameters.motion.copy(),
                    phase_logits=tee_fit.parameters.phase_logits.copy())
    packed = _pack(start, model, config)
    scale = max(float(np.mean(consensus.target ** 2)), 1e-9)
    shape_size = consensus.shape.size

    def residual_function(value: np.ndarray) -> np.ndarray:
        data = _residuals(value, start, observations, model, config)
        ring = consensus.offset + consensus.matrix @ value[:shape_size]
        return np.concatenate([data, np.sqrt(ring_weight / scale) * (ring - consensus.target)])

    packed, cost, iterations = _levenberg_marquardt(residual_function, packed, config)
    parameters = _unpack(packed, start, model, config)
    fit = _scene_fit(parameters, observations, model, config, float(cost), int(iterations))
    ring = consensus.offset + consensus.matrix @ packed[:shape_size]
    return {
        "fit": fit,
        "ring_deviation_mm": float(np.abs(ring - consensus.target).max()),
        "registration_rms_mm": consensus.registration_rms_mm,
        "projection_rms_mm": consensus.projection_rms_mm,
        "iterations": int(cmr_fit.iterations + tee_fit.iterations + iterations),
    }
