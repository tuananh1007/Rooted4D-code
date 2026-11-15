r"""Generate a joint-inference problem with known pose, phase, and session truth.

The truth is drawn inside the model's own subspaces on purpose. This measures
whether the staged solver *recovers* parameters that are in principle
recoverable; it is not a test of model misspecification, which needs the
held-out simulator families that remain an open readiness gate.

Held-out CMR frames and TEE vertices are removed before the solver sees
anything, so every baseline arm is scored on identical hidden evidence, as the
baseline lock requires.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .joint import session_design, session_rows, Observations, SolverConfig, predict
from rooted4d.synthetic.families import (
    TEST_FAMILY,
    TRAIN_FAMILY,
    VAL_FAMILY,
    apply_pathology,
    arrhythmic_phase_samples,
    correlated_vertex_noise,
    evaluate_spline_motion,
    spline_motion_coefficients,
)
from rooted4d.motion import periodic_basis

from .scene import SceneModel, SceneParameters


@dataclass(frozen=True)
class JointProblem:
    observations: Observations
    truth: SceneParameters
    model: SceneModel
    config: SolverConfig
    cmr_noise_mm: float
    tee_noise_mm: float


def make_joint_problem(
    model: SceneModel,
    config: SolverConfig = SolverConfig(),
    *,
    seed: int = 0,
    n_cmr_frames: int = 16,
    n_tee_frames: int = 20,
    heldout_cmr_frames: int = 4,
    heldout_tee_fraction: float = 0.25,
    cmr_noise_mm: float = 0.35,
    tee_noise_mm: float = 0.45,
    pose_translation_mm: float = 9.0,
    pose_rotation_rad: float = 0.18,
    identity_scale: float = 1.6,
    motion_scale: float = 1.1,
    session_scale: float = 0.35,
) -> JointProblem:
    rng = np.random.default_rng(seed)

    # The simulator is periodic regardless of what basis the solver will use.
    # Motion in a real heart is cyclic; whether the estimator represents it with
    # a Fourier basis or with free per-frame nodes is an inference choice, not a
    # property of the data. Generating with the solver's basis would turn the
    # per-frame ablation into a different experiment, one where periodicity is
    # absent from the truth as well, and it would no longer measure what
    # Proposition 4 claims.
    generator = replace(config, motion_model="periodic")

    axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    session_raw = {
        name: rng.normal(
            0.0, session_scale, size=(session_rows(config), model.bases.n_session)
        )
        for name in model.sessions
    }
    # Impose the cross-session gauge on the truth so it is the canonical triple.
    mean = sum(session_raw[name] for name in model.sessions) / len(model.sessions)
    truth = SceneParameters(
        pose_axis_angle=axis * pose_rotation_rad,
        pose_translation=rng.normal(0.0, pose_translation_mm, size=3),
        phase_offset=float(rng.uniform(0.0, 1.0)),
        phase_logits=rng.normal(0.0, 0.6, size=model.n_phase_nodes),
        identity=rng.normal(0.0, identity_scale, size=model.n_modes),
        motion=rng.normal(
            0.0, motion_scale, size=(2 * generator.fourier_order, model.bases.n_motion)
        ),
        session={name: session_raw[name] - mean for name in model.sessions},
    )

    all_cmr_times = np.linspace(0.0, 1.0, n_cmr_frames, endpoint=False)
    all_tee_times = np.linspace(0.0, 1.0, n_tee_frames, endpoint=False)
    cmr_vertices = np.sort(model.groups["cmr_visible"])
    tee_vertices = np.sort(model.groups["tee_visible"])

    hidden_cmr = np.sort(
        rng.choice(n_cmr_frames, size=heldout_cmr_frames, replace=False)
    )
    visible_cmr = np.setdiff1d(np.arange(n_cmr_frames), hidden_cmr)
    n_hidden_tee = max(1, int(round(heldout_tee_fraction * len(tee_vertices))))
    hidden_tee = np.sort(rng.choice(tee_vertices, size=n_hidden_tee, replace=False))
    visible_tee = np.setdiff1d(tee_vertices, hidden_tee)

    blank = Observations(
        cmr_times=all_cmr_times,
        cmr_vertices=cmr_vertices,
        cmr_points=np.zeros((n_cmr_frames, len(cmr_vertices), 3)),
        tee_times=all_tee_times,
        tee_vertices=tee_vertices,
        tee_points=np.zeros((n_tee_frames, len(tee_vertices), 3)),
        heldout_cmr_times=np.empty(0),
        heldout_cmr_vertices=np.empty(0, dtype=int),
        heldout_cmr_points=np.zeros((0, 0, 3)),
        heldout_tee_times=np.empty(0),
        heldout_tee_vertices=np.empty(0, dtype=int),
        heldout_tee_points=np.zeros((0, 0, 3)),
    )
    clean_cmr, clean_tee = predict(truth, blank, model, generator)

    def noisy(values: np.ndarray, sigma: float, stream: np.random.Generator) -> np.ndarray:
        return values + stream.normal(0.0, sigma, size=values.shape)

    cmr_rng = np.random.default_rng(seed + 7919)
    tee_rng = np.random.default_rng(seed + 104729)
    cmr_all = noisy(clean_cmr, cmr_noise_mm, cmr_rng)
    tee_all = noisy(clean_tee, tee_noise_mm, tee_rng)

    tee_visible_columns = np.searchsorted(tee_vertices, visible_tee)
    tee_hidden_columns = np.searchsorted(tee_vertices, hidden_tee)

    observations = Observations(
        cmr_times=all_cmr_times[visible_cmr],
        cmr_vertices=cmr_vertices,
        cmr_points=cmr_all[visible_cmr],
        tee_times=all_tee_times,
        tee_vertices=visible_tee,
        tee_points=tee_all[:, tee_visible_columns, :],
        heldout_cmr_times=all_cmr_times[hidden_cmr],
        heldout_cmr_vertices=cmr_vertices,
        heldout_cmr_points=cmr_all[hidden_cmr],
        heldout_tee_times=all_tee_times,
        heldout_tee_vertices=hidden_tee,
        heldout_tee_points=tee_all[:, tee_hidden_columns, :],
    )
    return JointProblem(
        observations=observations,
        truth=truth,
        model=model,
        config=config,
        cmr_noise_mm=cmr_noise_mm,
        tee_noise_mm=tee_noise_mm,
    )


def pose_errors(estimated: SceneParameters, truth: SceneParameters) -> dict[str, float]:
    left = estimated.transform().rotation
    right = truth.transform().rotation
    cosine = np.clip((np.trace(left @ right.T) - 1.0) / 2.0, -1.0, 1.0)
    return {
        "rotation_deg": float(np.degrees(np.arccos(cosine))),
        "translation_mm": float(
            np.linalg.norm(estimated.pose_translation - truth.pose_translation)
        ),
    }


def make_family_problem(
    model: SceneModel,
    config: SolverConfig = SolverConfig(),
    *,
    family: str = TRAIN_FAMILY,
    seed: int = 0,
    n_cmr_frames: int = 16,
    n_tee_frames: int = 20,
    heldout_cmr_frames: int = 4,
    heldout_tee_fraction: float = 0.25,
    cmr_noise_mm: float = 0.35,
    tee_noise_mm: float = 0.45,
) -> JointProblem:
    """Build a problem from a held-out generator family.

    The train family reproduces :func:`make_joint_problem` exactly, so a
    degradation measured against it is attributable to the family and not to a
    different code path. The held-out families break assumptions the solver
    makes: the validation family replaces the Fourier motion basis with a
    periodic cubic B-spline and independent noise with correlated noise, and the
    test family adds pathology, regional wall-motion abnormality, and irregular
    beat spacing.

    Ground-truth ``SceneParameters`` are still returned for pose and phase, which
    are family independent. Identity and motion coefficients are *not*
    recoverable for the held-out families, because the truth does not live in the
    model's span; comparing them would be meaningless and callers should score
    held-out evidence instead.
    """

    if family not in (TRAIN_FAMILY, VAL_FAMILY, TEST_FAMILY):
        raise ValueError(f"unknown generator family: {family}")
    if family == TRAIN_FAMILY:
        return make_joint_problem(
            model,
            config,
            seed=seed,
            n_cmr_frames=n_cmr_frames,
            n_tee_frames=n_tee_frames,
            heldout_cmr_frames=heldout_cmr_frames,
            heldout_tee_fraction=heldout_tee_fraction,
            cmr_noise_mm=cmr_noise_mm,
            tee_noise_mm=tee_noise_mm,
        )

    base = make_joint_problem(
        model,
        config,
        seed=seed,
        n_cmr_frames=n_cmr_frames,
        n_tee_frames=n_tee_frames,
        heldout_cmr_frames=heldout_cmr_frames,
        heldout_tee_fraction=heldout_tee_fraction,
        cmr_noise_mm=cmr_noise_mm,
        tee_noise_mm=tee_noise_mm,
    )
    truth = base.truth
    rng = np.random.default_rng(90_000 + seed)

    identity_geometry = model.physical_coordinates(truth.identity)
    warp = truth.warp()

    # One motion field for the patient, shared by both modalities. Drawing it
    # per modality would give CMR and TEE independent anatomy, and no
    # cross-modal method could reconcile that; the failure would be the
    # generator's, not the method's.
    motion_coefficients = spline_motion_coefficients(
        identity_geometry, 7, rng, amplitude=1.0
    )
    # A bounded session discrepancy in the model's own session subspace, matching
    # what the train family does. The families must differ in motion basis,
    # pathology and sampling, not in whether the session term is bounded at all.
    session_offsets = {
        name: rng.normal(
            0.0, 0.35, size=(session_rows(config), model.bases.n_session)
        )
        for name in model.sessions
    }
    session_mean = sum(session_offsets.values()) / len(session_offsets)
    session_offsets = {k: v - session_mean for k, v in session_offsets.items()}

    def geometry_for(times: np.ndarray, session: str) -> np.ndarray:
        # Acquisition times differ between sessions; under the test family the
        # beat spacing is irregular as well. This is realistic and is not the
        # same thing as giving the two modalities different anatomy.
        if family == TEST_FAMILY:
            sample_times = arrhythmic_phase_samples(len(times), rng)
        else:
            sample_times = times
        phases = warp.evaluate(sample_times) if session == "tee" else sample_times
        geometry = identity_geometry + evaluate_spline_motion(phases, motion_coefficients)
        residual = (
            session_design(phases, config)
            @ session_offsets["tee" if session == "tee" else "cmr"]
        ) @ model.bases.session.T
        geometry = geometry + model.physical_coordinates(residual) - model.template_physical
        if family == TEST_FAMILY:
            geometry, _ = apply_pathology(geometry, phases, model.groups, rng)
        return geometry

    cmr_geometry = geometry_for(np.concatenate(
        [base.observations.cmr_times, base.observations.heldout_cmr_times]
    ), "cmr")
    tee_geometry = geometry_for(base.observations.tee_times, "tee")

    correlation = 12.0 if family == VAL_FAMILY else 20.0
    cmr_noise = correlated_vertex_noise(
        model.template_physical, cmr_geometry.shape, cmr_noise_mm, correlation, rng
    )
    tee_noise = correlated_vertex_noise(
        model.template_physical, tee_geometry.shape, tee_noise_mm, correlation, rng
    )
    if family == TEST_FAMILY:
        # Heavier tails and occlusion, as the locked test family specifies.
        cmr_noise = cmr_noise * rng.standard_t(3.0, size=(len(cmr_noise), 1, 1))
        tee_noise = tee_noise * rng.standard_t(3.0, size=(len(tee_noise), 1, 1))

    posed_tee = truth.transform().apply(tee_geometry + tee_noise)
    noisy_cmr = cmr_geometry + cmr_noise

    n_fit = len(base.observations.cmr_times)
    observations = Observations(
        cmr_times=base.observations.cmr_times,
        cmr_vertices=base.observations.cmr_vertices,
        cmr_points=noisy_cmr[:n_fit][:, base.observations.cmr_vertices, :],
        tee_times=base.observations.tee_times,
        tee_vertices=base.observations.tee_vertices,
        tee_points=posed_tee[:, base.observations.tee_vertices, :],
        heldout_cmr_times=base.observations.heldout_cmr_times,
        heldout_cmr_vertices=base.observations.heldout_cmr_vertices,
        heldout_cmr_points=noisy_cmr[n_fit:][:, base.observations.heldout_cmr_vertices, :],
        heldout_tee_times=base.observations.heldout_tee_times,
        heldout_tee_vertices=base.observations.heldout_tee_vertices,
        heldout_tee_points=posed_tee[:, base.observations.heldout_tee_vertices, :],
    )
    return JointProblem(
        observations=observations,
        truth=truth,
        model=model,
        config=config,
        cmr_noise_mm=cmr_noise_mm,
        tee_noise_mm=tee_noise_mm,
    )
