r"""Staged joint pose / phase / anatomy / session inference (work package 5).

This is the MAP half of the Rooted4D inverse problem. It optimizes, in the
staged order section 6.3 prescribes, over

* TEE pose in SE(3), CMR being the reference frame (assumption A4);
* a monotone circular phase warp (Proposition 4);
* patient identity and shared periodic motion;
* a bounded low-rank session discrepancy under both centering gauges.

Two properties hold by construction rather than by penalty, which is the point
of the method: attachment, because geometry is assembled from unique physical
coordinates, and cycle closure, because motion uses an integer Fourier basis.
The objective in section 4.6 accordingly contains no attachment term and no
cycle-consistency term.

Optimization is Levenberg-Marquardt with a central-difference Jacobian. That is
slower than an analytic gradient but keeps the package dependency-light and makes
the same Jacobian available to the Theorem 5 and 6 diagnostics, replacing the
analytic stand-in used before this module existed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from .image_evidence import ImageEvidence

from rooted4d.motion import (
    MonotoneCircularPhaseWarp,
    interpolation_basis,
    periodic_basis,
)

from rooted4d.geometry.rigid import axis_angle_to_matrix

from .scene import SceneModel, SceneParameters

#: Curve-evidence settings (``SolverConfig.curve_evidence``) and the closed
#: curves each one scores by distance rather than vertex by vertex.
CURVE_EVIDENCE = {
    "": (),
    "annulus": ("annulus",),
    "annulus+free_edges": ("annulus", "anterior_free", "posterior_free"),
}


@dataclass(frozen=True)
class Observations:
    """Point evidence with an explicit held-out partition.

    Held-out frames and vertices are never shown to the solver; they exist so the
    first co-primary endpoint of the preregistered protocol -- error on hidden
    CMR planes and hidden TEE regions -- can be measured on every arm with the
    identical mask, as the baseline lock requires.
    """

    cmr_times: np.ndarray
    cmr_vertices: np.ndarray
    cmr_points: np.ndarray
    tee_times: np.ndarray
    tee_vertices: np.ndarray
    tee_points: np.ndarray
    heldout_cmr_times: np.ndarray
    heldout_cmr_vertices: np.ndarray
    heldout_cmr_points: np.ndarray
    heldout_tee_times: np.ndarray
    heldout_tee_vertices: np.ndarray
    heldout_tee_points: np.ndarray
    #: Optional TEE landmarks with known correspondence (E101): the template
    #: vertices the front end's commissures are taken to be, and their observed
    #: positions at every TEE frame. Only curve evidence reads them; every other
    #: configuration ignores them, so a sample that carries them is fitted
    #: exactly as before.
    tee_anchor_vertices: np.ndarray | None = None
    tee_anchor_points: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.cmr_points.shape != (
            len(self.cmr_times),
            len(self.cmr_vertices),
            3,
        ):
            raise ValueError("cmr_points must have shape (frames, vertices, 3)")
        if self.tee_points.shape != (
            len(self.tee_times),
            len(self.tee_vertices),
            3,
        ):
            raise ValueError("tee_points must have shape (frames, vertices, 3)")
        if (self.tee_anchor_vertices is None) != (self.tee_anchor_points is None):
            raise ValueError("tee anchors need both their vertices and their points")
        if self.tee_anchor_vertices is not None and np.shape(self.tee_anchor_points) != (
            len(self.tee_times),
            len(self.tee_anchor_vertices),
            3,
        ):
            raise ValueError("tee_anchor_points must have shape (tee frames, anchors, 3)")


@dataclass(frozen=True)
class SolverConfig:
    """Which parts of the model are active. Baseline arms differ only here."""

    fourier_order: int = 3
    session_order: int = 1
    #: ``"periodic"`` uses the Fourier basis of Proposition 4. ``"per_frame"``
    #: replaces it with free values at ``motion_nodes`` time nodes, linearly
    #: interpolated and *not* closed at the seam, which is the mandated
    #: per-frame ablation. Only shared motion changes; the session residual
    #: stays periodic so the arms differ in one thing.
    motion_model: str = "periodic"
    motion_nodes: int = 16
    #: Give the session residual a constant term as well as its oscillatory
    #: ones. Without it the session block is ``periodic_basis(tau, order)``,
    #: which is pure cosine and sine and therefore has zero mean over the cycle,
    #: so the model has no parameter for a *static* offset between two sessions.
    #: MITEA scan-rescan measures real between-session discrepancy at 2.90 mm rms
    #: static against 1.01 mm dynamic, so the dominant component of the real
    #: effect is exactly the one the original parameterization cannot express.
    #: Defaults to False, which reproduces every result computed before this was
    #: found; True is the corrected model.
    session_includes_offset: bool = False
    use_cmr: bool = True
    use_tee: bool = True
    estimate_pose: bool = True
    estimate_phase_warp: bool = True
    #: Freeze the scalar phase offset at its template value.
    #: ``estimate_phase_warp`` only freezes the warp *shape*: the offset is
    #: packed unconditionally, so a run with the warp disabled still slides
    #: phase. A diagnostic that needs genuinely fixed alignment has to freeze
    #: this as well. Defaults to True, which reproduces every result computed
    #: before the flag existed.
    estimate_phase_offset: bool = True
    use_session: bool = True
    enforce_gauges: bool = True
    identity_prior_weight: float = 1.0
    #: Shrink each shape coefficient by the anatomical variance the training
    #: identities actually show in that direction, instead of penalising every
    #: direction equally. The isotropic prior treats the weakest learned mode as
    #: being as plausible as the strongest, which is the reason added capacity
    #: has to be paid for twice: once in variance and once in a prior that does
    #: not know the modes are ordered. The total strength of the prior is held
    #: equal to the isotropic one, so this flag changes the *distribution* of
    #: shrinkage and not its magnitude, and ``identity_prior_weight`` keeps the
    #: same meaning in both arms. Requires ``model.shape_scales``; without it
    #: the isotropic prior is used and a warning-free fallback keeps every
    #: earlier result reproducible. Defaults to False for that reason.
    covariance_aware_identity_prior: bool = False
    #: Floor on the per-mode scale, as a fraction of the leading mode's scale.
    #: It guards against a spectrum that decays to nearly zero, where the
    #: weakest directions would divide by almost nothing and the prior would
    #: become a hard constraint on exactly the directions least determined by
    #: the data. It is a numerical guard and nothing more: E56b swept it from
    #: 0.005 to 0.20 on the v3 benchmark at 48 modes and held-out error fell
    #: monotonically as the floor fell, by 0.33 mm in total, with the best
    #: setting one that clamped no mode at all. Every mode the floor binds is
    #: a mode whose shrinkage has been made uniform again, so the default is
    #: set below that benchmark's smallest relative scale (0.0148) and binds
    #: nothing there. Raise it only for a spectrum that genuinely decays
    #: further.
    identity_prior_scale_floor: float = 0.005
    motion_prior_weight: float = 0.30
    session_prior_weight: float = 12.0
    phase_smoothness_weight: float = 2.0
    #: Only meaningful on a detached model; Theorem 2 governs what it buys.
    soft_attachment_weight: float = 0.0
    #: Weights on image-domain likelihood evidence. Zero, the default, keeps the
    #: solver on point evidence and every existing result reproducible. Non-zero
    #: requires an ImageEvidence passed to fit_joint_scene, and drops the
    #: assumption that an observation names which template vertex produced it.
    cmr_image_weight: float = 0.0
    tee_image_weight: float = 0.0
    #: E101, curve evidence. Point evidence names the template vertex behind each
    #: observation, so a front end that numbers a curve from a different place
    #: (E89: the valve fit's ring starts at an axis of arbitrary sign) supplies
    #: the right points with the wrong names. Under curve evidence the observed
    #: points of the named curves are scored by their distance to the predicted
    #: closed curve instead of to one vertex: ``"annulus"``, or
    #: ``"annulus+free_edges"`` for the ring and both leaflets' free edges.
    #: Correspondence then enters only through the TEE anchors
    #: (``Observations.tee_anchor_*``, the commissures), added as point
    #: evidence. ``""``, the default, is point evidence and every earlier result.
    curve_evidence: str = ""
    #: Under curve evidence with anchors, the start is first turned about the
    #: template's annular axis to the best of this many equally spaced
    #: rotations under the arm's own objective, because the shared start is a
    #: Kabsch fit that uses the ring's numbering.
    curve_start_rotations: int = 12
    max_iterations: int = 60
    finite_difference_step: float = 1e-5
    initial_damping: float = 1e-2
    convergence_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if self.cmr_image_weight < 0.0 or self.tee_image_weight < 0.0:
            raise ValueError("image weights cannot be negative")
        # A run needs evidence from somewhere, but it need not be point
        # evidence: an image-only arm turns both point modalities off, and
        # forbidding that would make the image term impossible to isolate. A
        # first attempt at the comparison left the point terms on, so the image
        # arm was point-plus-image against point, the thousands of point
        # residuals dominated, and the two arms agreed to within 0.03 mm --
        # which looked like a result and was an artefact.
        has_points = self.use_cmr or self.use_tee
        has_images = self.cmr_image_weight > 0.0 or self.tee_image_weight > 0.0
        if not (has_points or has_images):
            raise ValueError("at least one point modality or image weight must be enabled")
        if self.fourier_order < 1 or self.session_order < 1:
            raise ValueError("fourier_order and session_order must be at least one")
        if self.motion_model not in {"periodic", "per_frame"}:
            raise ValueError("motion_model must be 'periodic' or 'per_frame'")
        if self.motion_model == "per_frame" and self.motion_nodes < 2:
            raise ValueError("per_frame motion needs at least two nodes")
        if self.curve_evidence not in CURVE_EVIDENCE:
            raise ValueError(f"curve_evidence must be one of {sorted(CURVE_EVIDENCE)}")
        if self.curve_start_rotations < 1:
            raise ValueError("curve_start_rotations must be at least one")


def session_rows(config: SolverConfig) -> int:
    """Number of session parameter rows per session under the configured basis."""

    return 2 * config.session_order + (1 if config.session_includes_offset else 0)


def session_design(phases: np.ndarray, config: SolverConfig) -> np.ndarray:
    """Design matrix for the per-session residual at ``phases``."""

    oscillatory = periodic_basis(phases, config.session_order)
    if not config.session_includes_offset:
        return oscillatory
    constant = np.ones(oscillatory.shape[:-1] + (1,), dtype=np.float64)
    return np.concatenate([constant, oscillatory], axis=-1)


def motion_rows(config: SolverConfig) -> int:
    """Number of shared-motion parameter rows under the configured basis."""

    if config.motion_model == "per_frame":
        return config.motion_nodes
    return 2 * config.fourier_order


def motion_design(phases: np.ndarray, config: SolverConfig) -> np.ndarray:
    """Design matrix mapping motion rows to a trajectory at ``phases``."""

    if config.motion_model == "per_frame":
        return interpolation_basis(phases, config.motion_nodes)
    return periodic_basis(phases, config.fourier_order)


def _pack(parameters: SceneParameters, model: SceneModel, config: SolverConfig) -> np.ndarray:
    pieces = [parameters.identity, parameters.motion.reshape(-1)]
    if config.use_session:
        pieces.append(
            np.concatenate([parameters.session[name].reshape(-1) for name in model.sessions])
        )
    if config.estimate_pose:
        pieces.extend([parameters.pose_axis_angle, parameters.pose_translation])
    if config.estimate_phase_offset:
        pieces.append(np.array([parameters.phase_offset]))
    if config.estimate_phase_warp:
        pieces.append(parameters.phase_logits)
    return np.concatenate(pieces)


def _unpack(
    vector: np.ndarray,
    template: SceneParameters,
    model: SceneModel,
    config: SolverConfig,
) -> SceneParameters:
    cursor = 0

    def take(count: int) -> np.ndarray:
        nonlocal cursor
        chunk = vector[cursor : cursor + count]
        cursor += count
        return chunk

    identity = take(model.n_modes)
    motion = take(motion_rows(config) * model.bases.n_motion).reshape(
        motion_rows(config), model.bases.n_motion
    )
    if config.use_session:
        per_session = session_rows(config) * model.bases.n_session
        session = {
            name: take(per_session).reshape(session_rows(config), model.bases.n_session)
            for name in model.sessions
        }
    else:
        session = {
            name: np.zeros((session_rows(config), model.bases.n_session))
            for name in model.sessions
        }
    if config.estimate_pose:
        axis_angle = take(3)
        translation = take(3)
    else:
        axis_angle = template.pose_axis_angle
        translation = template.pose_translation
    phase_offset = (
        float(take(1)[0]) if config.estimate_phase_offset else float(template.phase_offset)
    )
    phase_logits = take(model.n_phase_nodes) if config.estimate_phase_warp else template.phase_logits
    return SceneParameters(
        pose_axis_angle=axis_angle,
        pose_translation=translation,
        phase_offset=phase_offset,
        phase_logits=phase_logits,
        identity=identity,
        motion=motion,
        session=session,
    )


def _apply_gauges(
    parameters: SceneParameters, model: SceneModel, config: SolverConfig
) -> SceneParameters:
    """Project onto the section 4.2 gauges.

    An integer Fourier basis has zero cycle mean already, so the motion gauge is
    automatic. The session gauge is not: it must be imposed by moving the
    weighted cross-session mean of the session coefficients into identity.
    """

    if not (config.enforce_gauges and config.use_session):
        return parameters
    weight = 1.0 / len(model.sessions)
    mean = sum(weight * parameters.session[name] for name in model.sessions)
    centered = {name: parameters.session[name] - mean for name in model.sessions}
    return replace(parameters, session=centered)


def latent_trajectory(
    parameters: SceneParameters,
    phases: np.ndarray,
    session: str,
    model: SceneModel,
    config: SolverConfig,
) -> np.ndarray:
    """Evaluate ``zeta_m(tau) = zeta_id + U_h h(tau) + U_s r_m(tau)``."""

    tau = np.asarray(phases, dtype=np.float64)
    shared = motion_design(tau, config) @ parameters.motion
    latent = parameters.identity + shared @ model.bases.motion.T
    if config.use_session:
        residual = session_design(tau, config) @ parameters.session[session]
        latent = latent + residual @ model.bases.session.T
    return latent


def predict(
    parameters: SceneParameters,
    observations: Observations,
    model: SceneModel,
    config: SolverConfig,
    *,
    cmr_times: np.ndarray | None = None,
    cmr_vertices: np.ndarray | None = None,
    tee_times: np.ndarray | None = None,
    tee_vertices: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Predict CMR and TEE point evidence, optionally on a held-out mask."""

    gauged = _apply_gauges(parameters, model, config)
    cmr_prediction = None
    tee_prediction = None

    times = observations.cmr_times if cmr_times is None else cmr_times
    vertices = observations.cmr_vertices if cmr_vertices is None else cmr_vertices
    if len(times):
        latent = latent_trajectory(gauged, times, "cmr", model, config)
        rows = model.observation_rows(vertices)
        cmr_prediction = model.physical_coordinates(latent)[:, rows, :]

    times = observations.tee_times if tee_times is None else tee_times
    vertices = observations.tee_vertices if tee_vertices is None else tee_vertices
    if len(times):
        warp = gauged.warp()
        latent = latent_trajectory(gauged, warp.evaluate(times), "tee", model, config)
        rows = model.observation_rows(vertices)
        geometry = model.physical_coordinates(latent)[:, rows, :]
        tee_prediction = gauged.transform().apply(geometry)
    return cmr_prediction, tee_prediction


#: Number of leading rows of the most recent ``_residuals`` call that came from
#: observations rather than penalties. Written on every call and read straight
#: afterwards by the Jacobian helpers below.
_OBSERVATION_ROWS: dict[str, int] = {"n": 0}


def _matched_detached_multiplier(
    model: SceneModel, config: SolverConfig, size: int
) -> np.ndarray:
    """The coupled model's identity prior, carried onto a matched detached model.

    The leading coordinates of a matched detached model are the coupled
    coefficients themselves, so the coupled model's per-coefficient factor --
    computed from its own scales by the same function, isotropic or
    covariance-aware alike -- applies to them unchanged. Separation coefficients
    get zero: only the attachment penalty restrains them.
    """

    lifted = int(model.matched_lifted_dimension)
    parent = replace(model, shape_scales=model.parent_shape_scales,
                     matched_lifted_dimension=None, parent_shape_scales=None)
    coupled = identity_prior_multiplier(parent, config, lifted)
    multiplier = np.zeros(size)
    multiplier[:lifted] = 1.0 if coupled is None else coupled
    return multiplier


def identity_prior_multiplier(
    model: SceneModel, config: SolverConfig, size: int
) -> np.ndarray | None:
    """Per-coefficient factor on the identity prior rows, or None if isotropic.

    Factored out of ``_residuals`` so the equivalence of the prior across
    coordinate systems can be tested through the code the solver runs rather
    than through a re-derivation of it.
    """

    if model.matched_lifted_dimension is not None:
        return _matched_detached_multiplier(model, config, size)
    if not (config.covariance_aware_identity_prior and model.shape_scales is not None):
        return None
    # N(0, diag(s^2)) in place of N(0, I): shrink each shape direction by how
    # much the training identities actually vary along it, instead of
    # treating the weakest learned mode as exactly as plausible as the
    # strongest. Scales are taken relative to the leading mode, so the
    # leading direction is penalised exactly as the isotropic prior
    # penalises it and only weaker ones are shrunk harder; using the raw
    # scales would silently redefine what identity_prior_weight means and
    # make this arm incomparable with its own control.
    scales = np.asarray(model.shape_scales, dtype=np.float64)
    scales = scales[:size]
    if scales.size != size or not scales[0] > 0.0:
        return None
    relative = np.maximum(
        scales / float(scales[0]), max(config.identity_prior_scale_floor, 1e-12)
    )
    # Equalise total strength, so the two priors differ in *where* they
    # shrink and not in *how much*. Without this the multipliers 1/rel
    # average 153 to 236 across the mode counts used here, so the aware
    # arm silently ran at an effective weight far outside the grid its
    # control was swept over, and the comparison measured regularisation
    # magnitude rather than covariance awareness. Normalising the mean
    # squared multiplier to one leaves the leading mode penalised a
    # little less than isotropic and the weak tail a great deal more,
    # which is the intended redistribution.
    multiplier = 1.0 / relative
    multiplier = multiplier / float(np.sqrt(np.mean(multiplier ** 2)))
    return multiplier


def _curve_loops(model: SceneModel) -> dict[str, np.ndarray]:
    """The closed curves of the procedural topology, each in its own vertex order.

    The free edges are closed loops because that is how the donors resample
    them (``valve_donor_vertices``).
    """

    free = np.asarray(model.groups["leaflet_free"])
    half = len(free) // 2
    return {
        "annulus": np.asarray(model.groups["annulus"]),
        "anterior_free": free[:half],
        "posterior_free": free[half:],
    }


def _closest_on_loop(points: np.ndarray, loop: np.ndarray) -> np.ndarray:
    """Closest point on each frame's closed polyline ``loop`` to each of ``points``.

    ``points`` is (frames, k, 3) and ``loop`` (frames, m, 3); the result is
    (frames, k, 3).
    """

    start = loop
    edge = np.roll(loop, -1, axis=1) - start
    relative = points[:, :, None, :] - start[:, None, :, :]
    length = np.maximum((edge**2).sum(axis=-1), 1e-12)[:, None, :]
    fraction = np.clip((relative * edge[:, None, :, :]).sum(axis=-1) / length, 0.0, 1.0)
    candidates = start[:, None, :, :] + fraction[..., None] * edge[:, None, :, :]
    nearest = ((points[:, :, None, :] - candidates) ** 2).sum(axis=-1).argmin(axis=2)
    return np.take_along_axis(candidates, nearest[..., None, None], axis=2)[:, :, 0, :]


def _curve_evidence(
    parameters: SceneParameters,
    observations: Observations,
    model: SceneModel,
    config: SolverConfig,
) -> list[np.ndarray]:
    """Evidence residuals under curve evidence (``SolverConfig.curve_evidence``).

    Every observed point keeps its three rows. A point on a named curve is
    scored against the closest point of that curve as predicted in its own
    frame; every other point against its own vertex, as under point evidence.
    The TEE anchors follow as point evidence.
    """

    loops = _curve_loops(model)
    curves = CURVE_EVIDENCE[config.curve_evidence]
    everything = np.arange(len(model.template_physical))
    cmr_full, tee_full = predict(
        parameters, observations, model, config, cmr_vertices=everything, tee_vertices=everything
    )
    pieces: list[np.ndarray] = []
    for active, full, vertices, points in (
        (config.use_cmr, cmr_full, observations.cmr_vertices, observations.cmr_points),
        (config.use_tee, tee_full, observations.tee_vertices, observations.tee_points),
    ):
        if not active or full is None:
            continue
        vertices = np.asarray(vertices)
        points = np.asarray(points, dtype=np.float64)
        difference = full[:, vertices, :] - points
        for name in curves:
            on_curve = np.isin(vertices, loops[name])
            if on_curve.any():
                difference[:, on_curve, :] = (
                    _closest_on_loop(points[:, on_curve, :], full[:, loops[name], :])
                    - points[:, on_curve, :]
                )
        pieces.append(difference.reshape(-1))
    if config.use_tee and tee_full is not None and observations.tee_anchor_vertices is not None:
        anchors = np.asarray(observations.tee_anchor_vertices, dtype=np.int64)
        pieces.append((tee_full[:, anchors, :] - observations.tee_anchor_points).reshape(-1))
    return pieces


def _axis_angle(rotation: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation matrix, stable up to and at half a turn."""

    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-12:
        return np.zeros(3)
    if np.sin(angle) > 1e-6:
        axis = np.array([
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]) / (2.0 * np.sin(angle))
    else:
        # Near half a turn the skew part vanishes; the axis is the dominant
        # column of (R + I) / 2 = a a^T.
        outer = 0.5 * (rotation + np.eye(3))
        column = int(np.argmax(np.diag(outer)))
        axis = outer[:, column] / np.sqrt(max(outer[column, column], 1e-12))
    return axis / np.linalg.norm(axis) * angle


def turned_about_annulus(
    parameters: SceneParameters, model: SceneModel, angle: float
) -> SceneParameters:
    """The same scene with the template turned by ``angle`` about its annular axis.

    The CMR geometry is unchanged; in the TEE frame each vertex lands where the
    turned template puts it, so the TEE evidence is explained by the template
    turned about the axis through the ring's centre.
    """

    ring = model.template_physical[np.asarray(model.groups["annulus"])]
    centre = ring.mean(axis=0)
    axis = np.linalg.svd(ring - centre, full_matrices=False)[2][2]
    turn = axis_angle_to_matrix(axis * float(angle))
    pose = parameters.transform()
    return replace(
        parameters,
        pose_axis_angle=_axis_angle(pose.rotation @ turn),
        pose_translation=pose.translation + pose.rotation @ (centre - turn @ centre),
    )


def _curve_start(
    template: SceneParameters,
    observations: Observations,
    model: SceneModel,
    config: SolverConfig,
    image: "ImageEvidence | None",
) -> SceneParameters:
    """The start turned about the annular axis to the best of the tried rotations."""

    best = (np.inf, template)
    for index in range(config.curve_start_rotations):
        candidate = template if index == 0 else turned_about_annulus(
            template, model, 2.0 * np.pi * index / config.curve_start_rotations
        )
        residual = _residuals(
            _pack(candidate, model, config), candidate, observations, model, config, image=image
        )
        cost = float(residual @ residual)
        if cost < best[0]:
            best = (cost, candidate)
    return best[1]


def _residuals(
    vector: np.ndarray,
    template: SceneParameters,
    observations: Observations,
    model: SceneModel,
    config: SolverConfig,
    image: "ImageEvidence | None" = None,
) -> np.ndarray:
    parameters = _unpack(vector, template, model, config)

    pieces: list[np.ndarray] = []
    if CURVE_EVIDENCE[config.curve_evidence]:
        pieces.extend(_curve_evidence(parameters, observations, model, config))
    else:
        cmr_prediction, tee_prediction = predict(parameters, observations, model, config)
        if config.use_cmr and cmr_prediction is not None:
            pieces.append((cmr_prediction - observations.cmr_points).reshape(-1))
        if config.use_tee and tee_prediction is not None:
            pieces.append((tee_prediction - observations.tee_points).reshape(-1))
    if image is not None and (config.cmr_image_weight or config.tee_image_weight):
        # Imported here because image evidence imports the solver for its
        # trajectory helper, and a module-level import would be circular.
        from .image_evidence import image_residuals

        block = image_residuals(parameters, image, model, config)
        if block.size:
            pieces.append(block)

    # Everything appended from here on is a penalty rather than an observation.
    # Recorded so a caller that needs the evidence alone can slice on it instead
    # of recomputing the count and drifting out of step with this order.
    #
    # Image residuals are counted here, above the priors, because they are
    # evidence. They used to be appended *after* the priors, so the prefix
    # ``evidence_jacobian()`` slices excluded them: an image-only fit recorded
    # zero observation rows and raised, and a mixed fit silently dropped the
    # image information from the Theorem 5 and 6 diagnostics.
    _OBSERVATION_ROWS["n"] = int(sum(piece.size for piece in pieces))

    # Section 4.6 prior terms. There is deliberately no attachment residual and
    # no cycle-closure residual here; both hold by construction.
    identity_rows = np.sqrt(config.identity_prior_weight) * parameters.identity
    multiplier = identity_prior_multiplier(model, config, parameters.identity.size)
    if multiplier is not None:
        identity_rows = identity_rows * multiplier
    pieces.append(identity_rows)
    pieces.append(np.sqrt(config.motion_prior_weight) * parameters.motion.reshape(-1))
    if config.use_session:
        gauged = _apply_gauges(parameters, model, config)
        pieces.append(
            np.sqrt(config.session_prior_weight)
            * np.concatenate([gauged.session[name].reshape(-1) for name in model.sessions])
        )
    if model.is_detached and config.soft_attachment_weight > 0.0:
        # The only place in this package where attachment is a loss rather than
        # a construction. It exists to make the locked baseline comparison real.
        phases = np.linspace(0.0, 1.0, 8, endpoint=False)
        # The gauged geometry, which is the one reported and scored (E120);
        # the raw session mean is otherwise free to absorb the penalty.
        latent = latent_trajectory(
            _apply_gauges(parameters, model, config), phases, "cmr", model, config
        )
        coordinates = model.physical_coordinates(latent)
        flat = coordinates.reshape(coordinates.shape[0], -1)
        pieces.append(
            np.sqrt(config.soft_attachment_weight)
            * (flat @ model.soft_constraint.T).reshape(-1)
        )

    if config.estimate_phase_warp:
        # Penalize local acceleration of the warp only, never its monotonicity,
        # which is already guaranteed by the softplus increments.
        increments = parameters.warp().increments
        pieces.append(
            np.sqrt(config.phase_smoothness_weight)
            * np.diff(increments)
            * len(increments)
        )
    return np.concatenate(pieces)


def _levenberg_marquardt(
    residual_function: Callable[[np.ndarray], np.ndarray],
    start: np.ndarray,
    config: SolverConfig,
) -> tuple[np.ndarray, float, int]:
    current = np.array(start, dtype=np.float64)
    residual = residual_function(current)
    cost = float(residual @ residual)
    damping = config.initial_damping
    iterations = 0

    for iterations in range(1, config.max_iterations + 1):
        jacobian = _numerical_jacobian(residual_function, current, config.finite_difference_step)
        gradient = jacobian.T @ residual
        hessian = jacobian.T @ jacobian
        improved = False
        for _ in range(12):
            try:
                step = np.linalg.solve(
                    hessian + damping * np.diag(np.maximum(np.diag(hessian), 1e-12)),
                    -gradient,
                )
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            candidate = current + step
            candidate_residual = residual_function(candidate)
            candidate_cost = float(candidate_residual @ candidate_residual)
            if np.isfinite(candidate_cost) and candidate_cost < cost:
                improvement = cost - candidate_cost
                current, residual, cost = candidate, candidate_residual, candidate_cost
                damping = max(damping * 0.4, 1e-12)
                improved = True
                if improvement < config.convergence_tolerance * max(cost, 1.0):
                    return current, cost, iterations
                break
            damping *= 10.0
        if not improved:
            break
    return current, cost, iterations


def _numerical_jacobian(
    residual_function: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
    step: float,
) -> np.ndarray:
    columns = []
    for index in range(len(point)):
        offset = np.zeros_like(point)
        offset[index] = step
        forward = residual_function(point + offset)
        backward = residual_function(point - offset)
        columns.append((forward - backward) / (2.0 * step))
    return np.stack(columns, axis=1)


def packed_gauge_directions(
    parameters: SceneParameters, model: SceneModel, config: SolverConfig
) -> np.ndarray:
    r"""Analytic gauge directions of the packed parameterization.

    Theorem 5 requires the latent gauges to be part of ``c(theta)`` so the
    tangent space excludes them. Two exist here, and both are constructions
    rather than defects:

    **Session centering.** ``_apply_gauges`` subtracts the weighted cross-session
    mean, so moving every session's coefficients by the same amount leaves every
    prediction unchanged. This contributes one direction per session coefficient.

    **Phase-increment normalization.** Increments are
    ``s_j = softplus(u_j) + eps`` divided by their sum, so any perturbation with
    ``delta s ∝ s`` leaves the normalized increments fixed. Differentiating,
    that direction is ``v_j = s_j / sigma(u_j)`` with ``sigma`` the logistic
    derivative of softplus. It is evaluated at the current logits because the
    parameterization is nonlinear.

    Returned as rows, matching the ``constraint_jacobian`` convention of
    :func:`rooted4d.theory.constrained_identifiability`.
    """

    dimension = len(_pack(parameters, model, config))
    rows: list[np.ndarray] = []

    offset = model.n_modes + motion_rows(config) * model.bases.n_motion
    if config.use_session:
        per_session = session_rows(config) * model.bases.n_session
        if config.enforce_gauges:
            for slot in range(per_session):
                direction = np.zeros(dimension)
                for index in range(len(model.sessions)):
                    direction[offset + index * per_session + slot] = 1.0
                rows.append(direction / np.linalg.norm(direction))
        # Advance past the session block whether or not its gauge was declared.
        # Keeping this inside the `enforce_gauges` branch left the offset short
        # by the session block, so with gauges disabled the phase gauge was
        # written into the session and pose slots. The true phase gauge then
        # stayed in the tangent space and alpha read zero for the wrong reason.
        offset += per_session * len(model.sessions)

    if config.estimate_pose:
        offset += 6
    if config.estimate_phase_offset:
        # The scalar phase offset is identified rather than a gauge, but it
        # still occupies a slot. Advancing past it while ``_pack`` omits it
        # would write the phase gauge one slot late -- the same class of offset
        # defect recorded above for the session block.
        offset += 1

    if config.estimate_phase_warp:
        logits = np.asarray(parameters.phase_logits, dtype=np.float64)
        softplus = np.logaddexp(0.0, logits) + MonotoneCircularPhaseWarp(
            logits
        ).epsilon
        # Stable logistic: the naive form overflows for strongly negative
        # logits, which the held-out families produce.
        sigmoid = np.where(
            logits >= 0.0,
            1.0 / (1.0 + np.exp(-np.clip(logits, 0.0, None))),
            np.exp(np.clip(logits, None, 0.0)) / (1.0 + np.exp(np.clip(logits, None, 0.0))),
        )
        direction = np.zeros(dimension)
        direction[offset : offset + len(logits)] = softplus / np.maximum(sigmoid, 1e-12)
        norm = float(np.linalg.norm(direction))
        if norm > 0.0:
            rows.append(direction / norm)

    if not rows:
        return np.zeros((0, dimension))
    return np.stack(rows, axis=0)


@dataclass(frozen=True)
class SceneFit:
    """A fitted scene plus the diagnostics the preregistered protocol needs."""

    parameters: SceneParameters
    cost: float
    iterations: int
    stage_costs: tuple[float, ...]
    attachment_residual: float
    packed: np.ndarray
    model: SceneModel
    config: SolverConfig
    observations: Observations
    #: Retained so the diagnostic Jacobian is the Jacobian of the objective that
    #: was actually minimized. Omitting it would make alpha describe a
    #: point-evidence problem the solver never solved.
    image: "ImageEvidence | None" = None

    def objective_jacobian(self) -> np.ndarray:
        """Jacobian of the full penalized residual that was minimized.

        Appropriate for posterior curvature, where the prior is part of the
        model. Not the evidence Jacobian: it carries the identity, motion and
        session prior rows, plus any enabled penalty, so its smallest eigenvalue
        can be positive because a prior is informative rather than because the
        observations are.
        """

        def residual_function(vector: np.ndarray) -> np.ndarray:
            return _residuals(
                vector, self.parameters, self.observations, self.model, self.config,
                image=self.image,
            )

        return _numerical_jacobian(
            residual_function, self.packed, self.config.finite_difference_step
        )

    def evidence_jacobian(self) -> np.ndarray:
        """``J_E`` for the Theorem 5 and 6 diagnostics: observations only.

        This previously differentiated the whole penalized residual and was
        described as the evidence Jacobian anyway. On a small synthetic probe
        that returned 3,959 rows against 3,924 observation residuals, and the
        reported alpha moved from about 0.713 to 3.231 once the prior and
        smoothness rows were included. A positive alpha then describes
        regularized stability rather than information supplied by the data,
        which is precisely the interpretation Theorem 5 forbids.
        """

        full = self.objective_jacobian()
        rows = _OBSERVATION_ROWS["n"]
        if rows <= 0 or rows > full.shape[0]:
            raise RuntimeError(
                "observation row count (%d) is not consistent with the residual "
                "of %d rows; the split is stale" % (rows, full.shape[0])
            )
        return full[:rows]

    def gauge_directions(self) -> np.ndarray:
        """Rows spanning the analytic gauge subspace at this solution."""

        return packed_gauge_directions(self.parameters, self.model, self.config)

    def heldout_errors(self) -> dict[str, float]:
        """First co-primary endpoint: error on evidence the solver never saw."""

        result: dict[str, float] = {}
        observations = self.observations
        if len(observations.heldout_cmr_times):
            prediction, _ = predict(
                self.parameters,
                observations,
                self.model,
                self.config,
                cmr_times=observations.heldout_cmr_times,
                cmr_vertices=observations.heldout_cmr_vertices,
                tee_times=np.empty(0),
            )
            distances = np.linalg.norm(
                prediction - observations.heldout_cmr_points, axis=-1
            )
            result["heldout_cmr_mm"] = float(np.mean(distances))
            result["heldout_cmr_p95_mm"] = float(np.percentile(distances, 95))
        if len(observations.heldout_tee_times):
            _, prediction = predict(
                self.parameters,
                observations,
                self.model,
                self.config,
                cmr_times=np.empty(0),
                tee_times=observations.heldout_tee_times,
                tee_vertices=observations.heldout_tee_vertices,
            )
            distances = np.linalg.norm(
                prediction - observations.heldout_tee_points, axis=-1
            )
            result["heldout_tee_mm"] = float(np.mean(distances))
            result["heldout_tee_p95_mm"] = float(np.percentile(distances, 95))
        return result


def initial_parameters(model: SceneModel, config: SolverConfig) -> SceneParameters:
    return SceneParameters(
        pose_axis_angle=np.zeros(3),
        pose_translation=np.zeros(3),
        phase_offset=0.0,
        phase_logits=np.zeros(model.n_phase_nodes),
        identity=np.zeros(model.n_modes),
        motion=np.zeros((motion_rows(config), model.bases.n_motion)),
        session={
            name: np.zeros((session_rows(config), model.bases.n_session))
            for name in model.sessions
        },
    )


def _attachment_residual(
    parameters: SceneParameters, model: SceneModel, config: SolverConfig
) -> float:
    """Max ``|Cx|_inf`` over the cycle, which the protocol wants per case."""

    phases = np.linspace(0.0, 1.0, 32, endpoint=False)
    latent = latent_trajectory(parameters, phases, "cmr", model, config)
    if model.is_detached:
        return model.detached_attachment_residual(latent)
    return model.attachment_residual(latent)


def fit_joint_scene(
    observations: Observations,
    model: SceneModel,
    config: SolverConfig = SolverConfig(),
    *,
    start: SceneParameters | None = None,
    phase_offset_candidates: int = 24,
    image: "ImageEvidence | None" = None,
) -> SceneFit:
    """Run the staged optimization of work package 5.

    Stages unlock parameters progressively rather than releasing everything at
    once, because a joint fit started cold on pose, phase, anatomy and session
    together reliably finds a local minimum in which the session term has
    absorbed the pose error.
    """

    template = initial_parameters(model, config) if start is None else start
    if (
        CURVE_EVIDENCE[config.curve_evidence]
        and config.use_tee
        and observations.tee_anchor_vertices is not None
        and config.curve_start_rotations > 1
    ):
        template = _curve_start(template, observations, model, config, image)

    # Stage A: a coarse circular phase-offset search before anything is unlocked,
    # so the later stages start in the right basin of the periodic objective.
    tee_evidence = config.use_tee or config.tee_image_weight > 0.0
    # Skipped when the offset is frozen: searching a parameter the caller has
    # declared fixed would still choose it, just by enumeration instead of by
    # gradient, and a fixed-alignment diagnostic would silently not be fixed.
    if (
        tee_evidence
        and config.estimate_phase_warp
        and config.estimate_phase_offset
        and phase_offset_candidates > 0
    ):
        frozen = replace(
            config,
            estimate_pose=config.estimate_pose,
            estimate_phase_warp=False,
            use_session=False,
            max_iterations=8,
        )
        best: tuple[float, SceneParameters] | None = None
        for offset in np.linspace(0.0, 1.0, phase_offset_candidates, endpoint=False):
            candidate = replace(template, phase_offset=float(offset))
            vector = _pack(candidate, model, frozen)
            solution, cost, _ = _levenberg_marquardt(
                lambda value, base=candidate: _residuals(
                    value, base, observations, model, frozen, image=image
                ),
                vector,
                frozen,
            )
            if best is None or cost < best[0]:
                best = (cost, _unpack(solution, candidate, model, frozen))
        assert best is not None
        template = best[1]

    stages = [
        replace(config, use_session=False, estimate_phase_warp=False),
        replace(config, use_session=False),
        config,
    ]
    stage_costs = []
    current = template
    cost = float("inf")
    iterations = 0
    packed = _pack(current, model, config)

    for stage in stages:
        vector = _pack(current, model, stage)
        solution, cost, stage_iterations = _levenberg_marquardt(
            lambda value, base=current, active=stage: _residuals(
                value, base, observations, model, active, image=image
            ),
            vector,
            stage,
        )
        current = _unpack(solution, current, model, stage)
        stage_costs.append(cost)
        iterations += stage_iterations
        packed = _pack(current, model, config)

    current = _apply_gauges(current, model, config)
    return SceneFit(
        parameters=current,
        cost=cost,
        iterations=iterations,
        stage_costs=tuple(stage_costs),
        attachment_residual=_attachment_residual(current, model, config),
        packed=packed,
        model=model,
        config=config,
        observations=observations,
        image=image,
    )
