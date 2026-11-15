r"""The six comparison arms fixed in ``baselines/BASELINE_LOCK.yaml``.

Every arm sees identical observations and identical held-out masks, and none may
tune anything on the hidden evidence. The arms differ only in which parts of the
model are active, which is what isolates the contribution being claimed:

===========================  ==================================================
Arm                          What removing it tests
===========================  ==================================================
``cmr_only``                 information available without TEE
``tee_only``                 information available without CMR
``late_fusion``              benefit of joint inference over post-hoc registration
``joint_soft_attachment``    shared degrees of freedom versus a tuned penalty
``joint_exact_no_gauges``    identifiability, not generic fusion
``rooted4d_full``            the proposed method
===========================  ==================================================

A failed fit is reported as a failure, never silently dropped, because the lock
states that failed cases count as failures.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from rooted4d.motion import circular_distance
from rooted4d.registration import align_periodic_features, register_point_sets
from rooted4d.theory import constrained_identifiability

from .initialization import StartingAlignment, start_from
from .joint import Observations, SceneFit, SolverConfig, fit_joint_scene, predict
from .problem import JointProblem, pose_errors
from .scene import SceneModel, SceneParameters

BASELINE_IDS = (
    "cmr_only",
    "tee_only",
    "late_fusion",
    "joint_soft_attachment",
    "joint_exact_no_gauges",
    "rooted4d_full",
)


@dataclass(frozen=True)
class ArmResult:
    """One arm's scored outcome on one problem."""

    baseline_id: str
    succeeded: bool
    failure_reason: str | None
    attachment_residual_mm: float
    heldout_cmr_mm: float | None
    heldout_tee_mm: float | None
    rotation_deg: float | None
    translation_mm: float | None
    phase_error_cycles: float | None
    cost: float
    iterations: int
    #: Smallest non-gauge eigenvalue of this arm's own evidence Jacobian. When it
    #: vanishes the arm has an unidentified direction, and any quantity that
    #: direction moves is not a reproducible number.
    alpha: float | None = None
    identified: bool | None = None
    #: Tails and the composed phase map, for the unknown-alignment comparison.
    #: The phase offset alone is not an endpoint: most of its direction lies in
    #: the span of the warp, so what is scored is the whole estimated map from
    #: TEE acquisition time to cardiac phase against the true one, as a median
    #: circular distance over the TEE frames.
    heldout_cmr_p95_mm: float | None = None
    heldout_tee_p95_mm: float | None = None
    phase_map_cycles: float | None = None
    #: Interface repair only: how well the two independently fitted rings
    #: registered, how far their average lay from what the model can represent,
    #: and, for exact repair, how many shape directions it left free to refit
    #: (identity alone, and identity with motion).
    repair_registration_rms_mm: float | None = None
    repair_projection_rms_mm: float | None = None
    repair_free_identity_directions: int | None = None
    repair_free_shape_directions: int | None = None
    #: Fitted parameters, when a variant asks for them (``store_parameters``),
    #: so geometry can be measured later without refitting. Left out of the
    #: record otherwise, which keeps every earlier row unchanged.
    extras: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        record = {
            "baseline_id": self.baseline_id,
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
            "attachment_residual_mm": self.attachment_residual_mm,
            "heldout_cmr_mm": self.heldout_cmr_mm,
            "heldout_tee_mm": self.heldout_tee_mm,
            "rotation_deg": self.rotation_deg,
            "translation_mm": self.translation_mm,
            "phase_error_cycles": self.phase_error_cycles,
            "cost": self.cost,
            "iterations": self.iterations,
            "alpha": self.alpha,
            "identified": self.identified,
            "heldout_cmr_p95_mm": self.heldout_cmr_p95_mm,
            "heldout_tee_p95_mm": self.heldout_tee_p95_mm,
            "phase_map_cycles": self.phase_map_cycles,
            "repair_registration_rms_mm": self.repair_registration_rms_mm,
            "repair_projection_rms_mm": self.repair_projection_rms_mm,
            "repair_free_identity_directions": self.repair_free_identity_directions,
            "repair_free_shape_directions": self.repair_free_shape_directions,
        }
        if self.extras is not None:
            record["extras"] = self.extras
        return record


def parameters_record(parameters: SceneParameters) -> dict[str, Any]:
    """Every field of a parameter set, as arrays, for ``ArmResult.extras``."""

    return {
        "pose_axis_angle": np.asarray(parameters.pose_axis_angle, dtype=np.float64),
        "pose_translation": np.asarray(parameters.pose_translation, dtype=np.float64),
        "phase_offset": float(parameters.phase_offset),
        "phase_logits": np.asarray(parameters.phase_logits, dtype=np.float64),
        "identity": np.asarray(parameters.identity, dtype=np.float64),
        "motion": np.asarray(parameters.motion, dtype=np.float64),
        "session": {name: np.asarray(value, dtype=np.float64)
                    for name, value in parameters.session.items()},
    }


def parameters_from_record(record: dict[str, Any]) -> SceneParameters:
    """The inverse of :func:`parameters_record`, for records read back from JSON."""

    return SceneParameters(
        pose_axis_angle=np.asarray(record["pose_axis_angle"], dtype=np.float64),
        pose_translation=np.asarray(record["pose_translation"], dtype=np.float64),
        phase_offset=float(record["phase_offset"]),
        phase_logits=np.asarray(record["phase_logits"], dtype=np.float64),
        identity=np.asarray(record["identity"], dtype=np.float64),
        motion=np.asarray(record["motion"], dtype=np.float64),
        session={name: np.asarray(value, dtype=np.float64)
                 for name, value in record["session"].items()},
    )


def phase_map_error(parameters: SceneParameters, problem: JointProblem) -> float | None:
    """Median circular distance between estimated and true TEE time-to-phase maps."""

    times = np.asarray(problem.observations.tee_times, dtype=np.float64)
    if times.size == 0:
        return None
    estimated = np.asarray(parameters.warp().evaluate(times), dtype=np.float64)
    truth = np.asarray(problem.truth.warp().evaluate(times), dtype=np.float64)
    return float(np.median(circular_distance(estimated, truth)))


def _score(
    baseline_id: str,
    fit: SceneFit,
    problem: JointProblem,
    *,
    report_pose: bool = True,
    with_identifiability: bool = True,
) -> ArmResult:
    errors = fit.heldout_errors()
    alpha: float | None = None
    identified: bool | None = None
    if with_identifiability:
        # Restricting to the non-gauge tangent space is what makes a vanishing
        # alpha meaningful rather than an artefact of the parameterization.
        diagnostics = constrained_identifiability(
            fit.evidence_jacobian(), constraint_jacobian=fit.gauge_directions()
        )
        alpha = diagnostics.smallest_tangent_eigenvalue
        identified = bool(diagnostics.identifiable and alpha > 1e-8)
    pose = pose_errors(fit.parameters, problem.truth) if report_pose else {}
    phase = (
        float(
            circular_distance(
                np.array([fit.parameters.phase_offset]),
                np.array([problem.truth.phase_offset]),
            )[0]
        )
        if report_pose
        else None
    )
    return ArmResult(
        baseline_id=baseline_id,
        succeeded=bool(np.isfinite(fit.cost)),
        failure_reason=None if np.isfinite(fit.cost) else "non_finite_cost",
        attachment_residual_mm=fit.attachment_residual,
        heldout_cmr_mm=errors.get("heldout_cmr_mm"),
        heldout_tee_mm=errors.get("heldout_tee_mm"),
        rotation_deg=pose.get("rotation_deg"),
        translation_mm=pose.get("translation_mm"),
        phase_error_cycles=phase,
        cost=float(fit.cost),
        iterations=int(fit.iterations),
        alpha=alpha,
        identified=identified,
        heldout_cmr_p95_mm=errors.get("heldout_cmr_p95_mm"),
        heldout_tee_p95_mm=errors.get("heldout_tee_p95_mm"),
        phase_map_cycles=phase_map_error(fit.parameters, problem) if report_pose else None,
    )


def _unimodal(
    baseline_id: str,
    problem: JointProblem,
    config: SolverConfig,
    *,
    use_cmr: bool,
    use_tee: bool,
    alignment: StartingAlignment | None = None,
) -> ArmResult:
    arm_config = replace(
        config,
        use_cmr=use_cmr,
        use_tee=use_tee,
        # Pose and phase relate the two modalities; with only one there is no
        # evidence for them, so estimating them would be fitting noise.
        estimate_pose=use_tee,
        estimate_phase_warp=use_tee,
        use_session=False,
    )
    fit = fit_joint_scene(problem.observations, problem.model, arm_config,
                          start=_start(alignment, problem.model, arm_config))
    return _score(baseline_id, fit, problem, report_pose=use_tee)


def _start(
    alignment: StartingAlignment | None, model: SceneModel, config: SolverConfig
) -> SceneParameters | None:
    """The shared start for one arm's model, or None for the historical zero start."""

    return None if alignment is None else start_from(alignment, model, config)


def _late_fusion_fits(
    problem: JointProblem,
    config: SolverConfig,
    alignment: StartingAlignment | None,
    modality_overrides: dict | None,
) -> tuple[SceneFit, SceneFit, SolverConfig, SolverConfig]:
    """The two independent reconstructions both late-fusion arms register."""

    observations = problem.observations
    model = problem.model

    # Retuned independent fitting gives each modality its own solver settings;
    # with no overrides both fits use the shared config, as they always did.
    overrides = modality_overrides or {}
    cmr_config = replace(
        config, use_tee=False, estimate_pose=False, estimate_phase_warp=False, use_session=False,
        **overrides.get("cmr", {}),
    )
    cmr_fit = fit_joint_scene(observations, model, cmr_config)

    # The TEE reconstruction places itself in its own frame, which is what an
    # independent TEE pipeline produces; a shape basis cannot absorb a rigid
    # offset, so withholding pose here would fail the arm for the wrong reason.
    tee_config = replace(config, use_cmr=False, use_session=False, **overrides.get("tee", {}))
    tee_fit = fit_joint_scene(observations, model, tee_config,
                              start=_start(alignment, model, tee_config))
    return cmr_fit, tee_fit, cmr_config, tee_config


def run_late_fusion(
    problem: JointProblem,
    config: SolverConfig,
    *,
    alignment: StartingAlignment | None = None,
    modality_overrides: dict | None = None,
) -> ArmResult:
    """Independent reconstructions, then staged robust SE(3) and circular DTW.

    This is the arm the literature already occupies, so it is the one joint
    inference has to beat. The two reconstructions never inform each other: CMR
    anatomy is fitted from CMR evidence alone and predicts the hidden CMR
    planes, TEE anatomy is fitted from TEE evidence alone and predicts the hidden
    TEE region, and registration only supplies the pose and phase that relate
    them afterwards. That separation is the whole point of the comparison.
    """

    observations = problem.observations
    model = problem.model
    cmr_fit, tee_fit, cmr_config, _ = _late_fusion_fits(
        problem, config, alignment, modality_overrides)

    # Register on the anatomical interface, not merely on whichever vertices
    # both modalities happened to retain after dropout. Both reconstructions
    # predict the whole template, so restricting the registration to the ten or
    # fifteen surviving shared observations would hand this arm an
    # under-determined ICP problem and let the comparison be won by dropout
    # rather than by method. The annulus is the structure the chamber and the
    # leaflets actually share, so it is the fairest and the strongest choice.
    registration_vertices = np.union1d(
        np.intersect1d(observations.cmr_vertices, observations.tee_vertices),
        np.sort(model.groups["annulus"]),
    )
    # Spatial registration may use the whole annulus, because both
    # reconstructions predict every template vertex and those coordinates
    # therefore exist. Temporal alignment may not: it reads the *observed* point
    # sequences, and a vertex dropout removed has no column in them. Conflating
    # the two let searchsorted return an insertion point for a missing
    # identifier and silently read a neighbouring vertex's trajectory, which
    # happened on every late-fusion sample of the v3 benchmark -- 1 to 5 missing
    # annular identifiers on the CMR side and 5 to 14 on the TEE side.
    temporal_vertices = np.intersect1d(
        observations.cmr_vertices, observations.tee_vertices
    )
    if len(registration_vertices) < 4:
        return ArmResult(
            baseline_id="late_fusion",
            succeeded=False,
            failure_reason="insufficient_shared_anatomy_for_post_hoc_registration",
            attachment_residual_mm=float("nan"),
            heldout_cmr_mm=None,
            heldout_tee_mm=None,
            rotation_deg=None,
            translation_mm=None,
            phase_error_cycles=None,
            cost=float("inf"),
            iterations=0,
        )

    if len(temporal_vertices) < 4:
        return ArmResult(
            baseline_id="late_fusion",
            succeeded=False,
            failure_reason="insufficient_shared_observed_vertices_for_temporal_alignment",
            attachment_residual_mm=float("nan"),
            heldout_cmr_mm=None,
            heldout_tee_mm=None,
            rotation_deg=None,
            translation_mm=None,
            phase_error_cycles=None,
            cost=float("inf"),
            iterations=0,
        )

    # Post-hoc circular temporal alignment from observed features only.
    def curve_features(points: np.ndarray) -> np.ndarray:
        centered = points - points.mean(axis=1, keepdims=True)
        radii = np.linalg.norm(centered, axis=2)
        return np.stack(
            [
                radii.mean(axis=1),
                radii.std(axis=1),
                centered[:, :, 2].mean(axis=1),
                np.linalg.norm(points.mean(axis=1), axis=1),
            ],
            axis=1,
        )

    def observed_columns(available: np.ndarray, wanted: np.ndarray) -> np.ndarray:
        """Columns of ``available`` holding ``wanted``, refusing to guess.

        searchsorted returns an insertion point for an absent identifier rather
        than failing, so the membership check is the part that matters.
        """
        columns = np.searchsorted(available, wanted)
        if columns.size and (columns.max() >= available.size
                             or not np.array_equal(available[columns], wanted)):
            raise ValueError(
                "temporal alignment asked for vertices absent from the observed "
                "sequence; registration and temporal vertex sets must not be shared"
            )
        return columns

    cmr_columns = observed_columns(observations.cmr_vertices, temporal_vertices)
    tee_columns = observed_columns(observations.tee_vertices, temporal_vertices)
    temporal = align_periodic_features(
        curve_features(observations.cmr_points[:, cmr_columns, :]),
        curve_features(observations.tee_points[:, tee_columns, :]),
    )

    # Post-hoc robust rigid alignment of the two independent reconstructions.
    cmr_geometry, _ = predict(
        cmr_fit.parameters,
        observations,
        model,
        cmr_config,
        cmr_times=np.array([0.0]),
        cmr_vertices=registration_vertices,
        tee_times=np.empty(0),
    )
    tee_geometry, _ = predict(
        replace(tee_fit.parameters, pose_axis_angle=np.zeros(3), pose_translation=np.zeros(3)),
        observations,
        model,
        cmr_config,
        cmr_times=np.array([float(temporal.offset_cycles)]),
        cmr_vertices=registration_vertices,
        tee_times=np.empty(0),
    )
    spatial = register_point_sets(
        cmr_geometry.reshape(-1, 3),
        tee_fit.parameters.transform().apply(tee_geometry.reshape(-1, 3)),
    )

    # The interface endpoint for this arm is *not* either fit's internal
    # residual. Both fits use the coupled model, so each satisfies attachment on
    # its own and reporting max(cmr, tee) yields a flat zero -- which would claim
    # post-hoc registration produces a valid interface, the exact opposite of
    # what it does. The real defect is between the two reconstructions: CMR
    # places the annulus somewhere, TEE places it somewhere else, and after
    # registration the gap is whatever it is. That gap is measured here, over the
    # cycle, on the same vertices and the same phase grid the coupled arms are
    # scored on.
    # Registration was fitted on the shared vertices, which include the annulus,
    # so this gap is measured where late fusion is at its best. That is
    # deliberate: the baseline should fail on its merits, not on a handicap.
    annulus = np.sort(model.groups["annulus"])
    phases = np.linspace(0.0, 1.0, 32, endpoint=False)
    chamber_side, _ = predict(
        cmr_fit.parameters,
        observations,
        model,
        cmr_config,
        cmr_times=phases,
        cmr_vertices=annulus,
        tee_times=np.empty(0),
    )
    valve_side, _ = predict(
        replace(tee_fit.parameters, pose_axis_angle=np.zeros(3), pose_translation=np.zeros(3)),
        observations,
        model,
        cmr_config,
        cmr_times=(phases + float(temporal.offset_cycles)) % 1.0,
        cmr_vertices=annulus,
        tee_times=np.empty(0),
    )
    # ``spatial`` was fitted with the CMR reconstruction as source and the posed
    # TEE reconstruction as target, and is used below as the pose estimate, so it
    # maps the CMR frame into the TEE frame. Both sides are therefore compared in
    # the TEE frame; a rigid map preserves distance, so the gap is the same
    # either way, but applying it to the wrong side is not a small error.
    posed = tee_fit.parameters.transform().apply(valve_side)
    registered = spatial.transform.apply(chamber_side.reshape(-1, 3)).reshape(
        chamber_side.shape
    )
    interface_gap = float(np.abs(registered - posed).max())

    cmr_errors = cmr_fit.heldout_errors()
    tee_errors = tee_fit.heldout_errors()
    estimated = replace(
        cmr_fit.parameters,
        pose_axis_angle=_matrix_to_axis_angle(spatial.transform.rotation),
        pose_translation=spatial.transform.translation,
        phase_offset=float(temporal.offset_cycles),
    )
    pose = pose_errors(estimated, problem.truth)
    return ArmResult(
        baseline_id="late_fusion",
        succeeded=True,
        failure_reason=None,
        attachment_residual_mm=interface_gap,
        heldout_cmr_mm=cmr_errors.get("heldout_cmr_mm"),
        heldout_tee_mm=tee_errors.get("heldout_tee_mm"),
        rotation_deg=pose["rotation_deg"],
        translation_mm=pose["translation_mm"],
        phase_error_cycles=float(
            circular_distance(
                np.array([float(temporal.offset_cycles)]),
                np.array([problem.truth.phase_offset]),
            )[0]
        ),
        cost=float(cmr_fit.cost + tee_fit.cost),
        iterations=int(cmr_fit.iterations + tee_fit.iterations),
        heldout_cmr_p95_mm=cmr_errors.get("heldout_cmr_p95_mm"),
        heldout_tee_p95_mm=tee_errors.get("heldout_tee_p95_mm"),
        # Late fusion's temporal estimate is the full DTW map, not just its
        # offset, so that is what its composed phase map is scored on.
        phase_map_cycles=float(np.median(circular_distance(
            np.asarray(temporal.target_to_source_phase, dtype=np.float64),
            np.asarray(problem.truth.warp().evaluate(observations.tee_times),
                       dtype=np.float64)))),
    )


def run_late_fusion_trajectory(
    problem: JointProblem,
    config: SolverConfig,
    *,
    alignment: StartingAlignment | None = None,
    modality_overrides: dict | None = None,
    store_parameters: bool = False,
) -> ArmResult:
    """Late fusion's two reconstructions, registered as whole trajectories.

    The fits are late fusion's own, so held-out errors are identical and the
    arms differ in registration alone. The registration is exact repair's,
    extended by the warp (``trajectory_registration.py``): a rigid map, a phase
    offset and a monotone warp between the two fitted trajectories, over the
    annulus and every vertex both modalities observed, at 32 phases. No truth
    is used, and the ring discrepancy is measured as for every other arm.
    """

    return late_fusion_trajectory_fit(
        problem, config, alignment=alignment, modality_overrides=modality_overrides,
        store_parameters=store_parameters)[0]


def late_fusion_trajectory_fit(
    problem: JointProblem,
    config: SolverConfig,
    *,
    alignment: StartingAlignment | None = None,
    modality_overrides: dict | None = None,
    store_parameters: bool = False,
) -> tuple[ArmResult, tuple]:
    """``run_late_fusion_trajectory``'s result and what it was computed from.

    The second element is ``(cmr_fit, tee_fit, cmr_config, tee_config,
    registration)``; ``registration`` is None when too few vertices are shared.
    E119 adds a residual field to these fits (``late_fusion_field.py``).
    """

    from .trajectory_registration import register_trajectories, registered_interface_gap

    observations = problem.observations
    model = problem.model
    cmr_fit, tee_fit, cmr_config, tee_config = _late_fusion_fits(
        problem, config, alignment, modality_overrides)
    vertices = np.union1d(
        np.intersect1d(observations.cmr_vertices, observations.tee_vertices),
        np.sort(model.groups["annulus"]),
    )
    if len(vertices) < 4:
        return ArmResult(
            baseline_id="late_fusion_trajectory",
            succeeded=False,
            failure_reason="insufficient_shared_anatomy_for_post_hoc_registration",
            attachment_residual_mm=float("nan"),
            heldout_cmr_mm=None,
            heldout_tee_mm=None,
            rotation_deg=None,
            translation_mm=None,
            phase_error_cycles=None,
            cost=float("inf"),
            iterations=0,
        ), (cmr_fit, tee_fit, cmr_config, tee_config, None)

    registration = register_trajectories(
        cmr_fit.parameters, tee_fit.parameters, model, cmr_config, vertices)
    pose = registration.place(tee_fit.parameters.transform())
    tee_warp = tee_fit.parameters.warp()
    times = np.asarray(observations.tee_times, dtype=np.float64)
    offset = float(registration.warp.evaluate(tee_warp.evaluate(np.array([0.0])))[0])
    estimated = replace(
        tee_fit.parameters,
        pose_axis_angle=_matrix_to_axis_angle(pose.rotation),
        pose_translation=np.asarray(pose.translation, dtype=np.float64),
        phase_offset=offset,
    )
    errors = pose_errors(estimated, problem.truth)
    cmr_errors = cmr_fit.heldout_errors()
    tee_errors = tee_fit.heldout_errors()
    extras = None
    if store_parameters:
        extras = {
            "cmr": parameters_record(cmr_fit.parameters),
            "tee": parameters_record(tee_fit.parameters),
            "registration": {
                "rotation": np.asarray(registration.transform.rotation),
                "translation": np.asarray(registration.transform.translation),
                "offset": float(registration.warp.offset),
                "logits": np.asarray(registration.warp.logits),
            },
        }
    return ArmResult(
        baseline_id="late_fusion_trajectory",
        succeeded=bool(np.isfinite(cmr_fit.cost) and np.isfinite(tee_fit.cost)
                       and np.isfinite(registration.cost)),
        failure_reason=None if (np.isfinite(cmr_fit.cost) and np.isfinite(tee_fit.cost)
                                and np.isfinite(registration.cost)) else "non_finite_cost",
        attachment_residual_mm=registered_interface_gap(
            cmr_fit.parameters, tee_fit.parameters, registration, model, cmr_config),
        heldout_cmr_mm=cmr_errors.get("heldout_cmr_mm"),
        heldout_tee_mm=tee_errors.get("heldout_tee_mm"),
        rotation_deg=errors["rotation_deg"],
        translation_mm=errors["translation_mm"],
        phase_error_cycles=float(circular_distance(
            np.array([offset]), np.array([problem.truth.phase_offset]))[0]),
        cost=float(cmr_fit.cost + tee_fit.cost),
        iterations=int(cmr_fit.iterations + tee_fit.iterations + registration.iterations),
        heldout_cmr_p95_mm=cmr_errors.get("heldout_cmr_p95_mm"),
        heldout_tee_p95_mm=tee_errors.get("heldout_tee_p95_mm"),
        phase_map_cycles=float(np.median(circular_distance(
            registration.warp.evaluate(tee_warp.evaluate(times)),
            np.asarray(problem.truth.warp().evaluate(times), dtype=np.float64)))),
        repair_registration_rms_mm=registration.rms_mm,
        extras=extras,
    ), (cmr_fit, tee_fit, cmr_config, tee_config, registration)


def _run_repair(baseline_id, problem, config, alignment, options) -> ArmResult:
    """Independent fits, a registered consensus ring, then exact or penalty repair."""

    from .repair import exact_repair, penalty_repair

    overrides = options.get("modality_overrides")
    if baseline_id == "independent_penalty_repair":
        if "ring_weight" not in options:
            raise ValueError("penalty repair needs an explicit ring_weight")
        result = penalty_repair(problem.observations, problem.model, config,
                                alignment=alignment, ring_weight=float(options["ring_weight"]),
                                modality_overrides=overrides)
        scored = _score(baseline_id, result["fit"], problem, with_identifiability=False)
        return replace(scored, iterations=result["iterations"],
                       repair_registration_rms_mm=result["registration_rms_mm"],
                       repair_projection_rms_mm=result["projection_rms_mm"])
    result = exact_repair(problem.observations, problem.model, config,
                          alignment=alignment, modality_overrides=overrides)
    cmr, tee = result["cmr"], result["tee"]
    cmr_errors, tee_errors = cmr.heldout_errors(), tee.heldout_errors()
    pose = pose_errors(tee.parameters, problem.truth)
    return ArmResult(
        baseline_id=baseline_id,
        succeeded=bool(np.isfinite(cmr.cost) and np.isfinite(tee.cost)),
        failure_reason=None if np.isfinite(cmr.cost) and np.isfinite(tee.cost) else "non_finite_cost",
        attachment_residual_mm=result["interface_gap_mm"],
        heldout_cmr_mm=cmr_errors.get("heldout_cmr_mm"),
        heldout_tee_mm=tee_errors.get("heldout_tee_mm"),
        rotation_deg=pose["rotation_deg"],
        translation_mm=pose["translation_mm"],
        phase_error_cycles=float(circular_distance(
            np.array([tee.parameters.phase_offset]), np.array([problem.truth.phase_offset]))[0]),
        cost=float(cmr.cost + tee.cost),
        iterations=result["iterations"],
        heldout_cmr_p95_mm=cmr_errors.get("heldout_cmr_p95_mm"),
        heldout_tee_p95_mm=tee_errors.get("heldout_tee_p95_mm"),
        phase_map_cycles=phase_map_error(tee.parameters, problem),
        repair_registration_rms_mm=result["registration_rms_mm"],
        repair_projection_rms_mm=result["projection_rms_mm"],
        repair_free_identity_directions=result["free_identity_directions"],
        repair_free_shape_directions=result["free_shape_directions"],
    )


def _matrix_to_axis_angle(rotation: np.ndarray) -> np.ndarray:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.zeros(3)
    return axis / norm * angle


def run_arm(
    baseline_id: str,
    problem: JointProblem,
    config: SolverConfig,
    *,
    soft_attachment_weight: float = 1e4,
    alignment: StartingAlignment | None = None,
    options: dict | None = None,
) -> ArmResult:
    """Run one arm. ``alignment`` gives every fit the same observation-derived
    pose and phase start; None keeps the historical zero start, so every
    earlier result reproduces."""

    if baseline_id == "cmr_only":
        return _unimodal(baseline_id, problem, config, use_cmr=True, use_tee=False,
                         alignment=alignment)
    if baseline_id == "tee_only":
        return _unimodal(baseline_id, problem, config, use_cmr=False, use_tee=True,
                         alignment=alignment)
    options = options or {}
    if baseline_id == "late_fusion":
        return run_late_fusion(problem, config, alignment=alignment,
                               modality_overrides=options.get("modality_overrides"))
    if baseline_id == "late_fusion_trajectory":
        return run_late_fusion_trajectory(
            problem, config, alignment=alignment,
            modality_overrides=options.get("modality_overrides"),
            store_parameters=bool(options.get("store_parameters", False)))
    if baseline_id in ("independent_exact_repair", "independent_penalty_repair"):
        return _run_repair(baseline_id, problem, config, alignment, options)
    if baseline_id in ("joint_soft_attachment", "joint_soft_attachment_matched"):
        # The matched variant keeps the coupled model's coordinates, prior,
        # motion and session subspaces and gauges, so it differs from the exact
        # arm by the attachment treatment alone and converges to it as the
        # penalty stiffens. The historical arm is kept for E51's reproducibility.
        detached = problem.model.detached(
            matched=baseline_id == "joint_soft_attachment_matched")
        arm_config = replace(config, soft_attachment_weight=soft_attachment_weight)
        fit = fit_joint_scene(problem.observations, detached, arm_config,
                              start=_start(alignment, detached, arm_config))
        return _score(baseline_id, fit, problem)
    if baseline_id == "joint_exact_no_gauges":
        arm_config = replace(config, enforce_gauges=False, session_prior_weight=0.0)
        fit = fit_joint_scene(problem.observations, problem.model, arm_config,
                              start=_start(alignment, problem.model, arm_config))
        return _score(baseline_id, fit, problem)
    if baseline_id == "rooted4d_full":
        fit = fit_joint_scene(problem.observations, problem.model, config,
                              start=_start(alignment, problem.model, config))
        scored = _score(baseline_id, fit, problem)
        if options.get("store_parameters"):
            scored = replace(scored, extras={"parameters": parameters_record(fit.parameters)})
        return scored
    if baseline_id == "spatiotemporal_registration":
        # E114: adapted from Zhang et al. (2011); see published_fusion.py.
        from .published_fusion import run_spatiotemporal_registration

        return run_spatiotemporal_registration(
            problem, config, alignment=alignment,
            modality_overrides=options.get("modality_overrides"),
            widths_mm=tuple(float(w) for w in options.get("widths_mm", (10.0,))))
    if baseline_id == "model_based_fusion":
        # E114: adapted from Grbic et al. (2011); see published_fusion.py.
        from .published_fusion import run_model_based_fusion

        return run_model_based_fusion(
            problem, config, alignment=alignment,
            grid=[tuple(float(v) for v in pair) for pair in options.get("grid", [(1.0, 100.0)])],
            centers=int(options.get("centers", 12)),
            iterations=int(options.get("iterations", 25)))
    if baseline_id in ("late_fusion_bcpd", "late_fusion_ndp"):
        # E115: late fusion, then a recent non-rigid registration attaches the
        # valve; see recent_fusion.py.
        from .recent_fusion import bcpd_attachment, ndp_attachment, run_registration_attachment

        if baseline_id == "late_fusion_bcpd":
            grid = [tuple(float(v) for v in pair) for pair in options.get("grid", [(2.0, 2.0)])]
            method = bcpd_attachment(grid, gamma=float(options.get("gamma", 0.1)))
            primary = f"{grid[0][0]:g},{grid[0][1]:g}"
        else:
            levels = int(options.get("levels", 9))
            settings = {"iterations": int(options["iterations"])} if "iterations" in options else {}
            method = ndp_attachment(levels, seed=int(options.get("seed", 0)), **settings)
            primary = f"{int(options.get('primary_levels', levels))}"
        return run_registration_attachment(
            baseline_id, problem, config, alignment=alignment,
            modality_overrides=options.get("modality_overrides"), method=method, primary=primary)
    if baseline_id == "gpmm_fusion":
        # E115: model-based fusion with a Gaussian process morphable model.
        from .recent_fusion import run_gpmm_fusion

        return run_gpmm_fusion(
            problem, config, alignment=alignment,
            grid=[tuple(float(v) for v in triple) for triple in options.get("grid", [(0.0, 0.0, 1.0)])],
            iterations=int(options.get("iterations", 25)))
    if baseline_id == "late_fusion_residual_field":
        # E119: late fusion's two fits plus E116's residual field; see late_fusion_field.py.
        from .late_fusion_field import run_late_fusion_residual_field

        return run_late_fusion_residual_field(
            problem, config, alignment=alignment,
            modality_overrides=options.get("modality_overrides"),
            configs=[(str(c[0]), float(c[1]), float(c[2]), float(c[3]))
                     for c in options.get("configs", [("session", 0.0, 0.0, 10.0)])])
    if baseline_id == "rooted4d_residual_field":
        # E116: Rooted4D's fit plus a Gaussian-kernel residual field; see residual_field.py.
        from .residual_field import run_rooted4d_residual_field

        return run_rooted4d_residual_field(
            problem, config, alignment=alignment,
            configs=[(str(c[0]), float(c[1]), float(c[2]), float(c[3]))
                     for c in options.get("configs", [("shared", 0.0, 0.0, 10.0)])])
    raise ValueError(f"unknown baseline id: {baseline_id}")


def run_all_arms(
    problem: JointProblem,
    config: SolverConfig,
    *,
    arms: tuple[str, ...] = BASELINE_IDS,
    soft_attachment_weight: float = 1e4,
    alignment: StartingAlignment | None = None,
    options: dict | None = None,
) -> dict[str, ArmResult]:
    return {
        arm: run_arm(
            arm, problem, config, soft_attachment_weight=soft_attachment_weight,
            alignment=alignment, options=options,
        )
        for arm in arms
    }
