r"""Rooted4D with a Gaussian-kernel residual field (E116).

Pre-registered in ``Docs/E116_RESIDUAL_FIELD_PREREGISTRATION_2026-10-23.md``.
Rooted4D's fit is kept as it is: pose, timing, identity, shared motion and
session terms. A residual field on the physical vertices is then added to its
geometry, with the Gaussian kernel of the GPMM comparator (E115) in space and
Rooted4D's own periodic basis in time,

    e(x, tau) = sum_j psi_j(tau) E_j(x),
    psi = [1, cos 2 pi tau, sin 2 pi tau, ..., cos 2 pi K tau, sin 2 pi K tau],
    E_j ~ GP(0, s_j^2 exp(-|x - x'|^2 / 2 w^2) I),

with ``K`` the solver's Fourier order, ``s_0`` the static scale and ``s_j`` the
dynamic scale for every harmonic coefficient. Given the fit, the evidence is
linear in the field, so its posterior mean under isotropic noise is solved in
closed form. Pose and timing are not re-estimated, so the posterior mean
depends on the scales only through their ratio to the noise; scales are
therefore stated in units of a 1 mm noise.

Two properties of Rooted4D survive by construction. The field lives on the
physical vertices, so the chamber and the valve still share every annulus
vertex and attachment stays exact. Its temporal basis is periodic, so the cycle
still closes.

``shared``
    One field, updated by both modalities' residuals: the TEE residuals are
    mapped into the CMR frame by the fitted pose's rotation, under which the
    isotropic kernel is unchanged.
``session``
    One field per modality, each from its own residuals only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from rooted4d.motion import periodic_basis

from .baselines import ArmResult, _score, _start
from .initialization import StartingAlignment
from .joint import SceneFit, fit_joint_scene, latent_trajectory, predict
from .problem import JointProblem
from .published_fusion import _failure, _mean_and_p95
from .recent_fusion import GAP_PHASES

SHARINGS = ("shared", "session")
#: Kernel eigenvalues below this fraction of the largest are dropped. A
#: numerical guard: those directions carry less than 1e-10 of the leading
#: direction's prior variance.
RELATIVE_FLOOR = 1e-10


def temporal_design(phases: np.ndarray, order: int) -> np.ndarray:
    """``[1, cos 2 pi tau, sin 2 pi tau, ...]`` up to ``order``, shape (frames, 2 order + 1)."""

    tau = np.asarray(phases, dtype=np.float64)
    return np.concatenate([np.ones(tau.shape + (1,)), periodic_basis(tau, order)], axis=-1)


def gaussian_factor(template: np.ndarray, width: float) -> np.ndarray:
    """``F`` with ``F F^T = exp(-d^2 / 2 w^2)`` on ``template``, shape (vertices, kept modes)."""

    squared = ((template[:, None] - template[None]) ** 2).sum(-1)
    values, vectors = np.linalg.eigh(np.exp(-squared / (2.0 * float(width) ** 2)))
    keep = values > RELATIVE_FLOOR * values.max()
    return vectors[:, keep] * np.sqrt(values[keep])


@dataclass(frozen=True)
class FieldEvidence:
    """One modality's residuals in the CMR frame: every observed row at every frame."""

    phases: np.ndarray
    rows: np.ndarray
    residuals: np.ndarray


def field_evidence(fit: SceneFit) -> dict[str, FieldEvidence]:
    """Residuals of the visible evidence after the fit, per modality, in the CMR frame."""

    parameters, observations, model = fit.parameters, fit.observations, fit.model
    cmr_prediction, tee_prediction = predict(parameters, observations, model, fit.config)
    out = {}
    if cmr_prediction is not None and fit.config.use_cmr:
        out["cmr"] = FieldEvidence(
            phases=np.asarray(observations.cmr_times, dtype=np.float64),
            rows=model.observation_rows(np.asarray(observations.cmr_vertices)),
            residuals=np.asarray(observations.cmr_points, dtype=np.float64) - cmr_prediction)
    if tee_prediction is not None and fit.config.use_tee:
        # y - (R x + t) = R (R^T (y - t) - x): a row vector v maps to R^T v as v @ R.
        rotation = parameters.transform().rotation
        out["tee"] = FieldEvidence(
            phases=np.asarray(parameters.warp().evaluate(np.asarray(observations.tee_times)), dtype=np.float64),
            rows=model.observation_rows(np.asarray(observations.tee_vertices)),
            residuals=(np.asarray(observations.tee_points, dtype=np.float64) - tee_prediction) @ rotation)
    return out


def normal_equations(pieces: list[FieldEvidence], factor: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """``H^T H`` and ``H^T r`` of the whitened field coefficients, ordered (temporal, spatial).

    An observation of row ``a`` at phase ``tau`` reads ``psi(tau) kron F[a]``,
    so ``H^T H`` sums ``psi psi^T kron F[a]^T F[a]`` over observations.
    """

    vertices, modes = factor.shape
    terms = 2 * order + 1
    outer = np.zeros((vertices, terms, terms))
    projected = np.zeros((vertices, terms, 3))
    for piece in pieces:
        design = temporal_design(piece.phases, order)
        # Duplicated rows (one physical vertex observed twice) accumulate.
        np.add.at(outer, piece.rows, np.broadcast_to(design.T @ design, (len(piece.rows), terms, terms)))
        np.add.at(projected, piece.rows, np.einsum("fj,fkc->kjc", design, piece.residuals))
    gram = np.empty((terms, modes, terms, modes))
    for j in range(terms):
        for k in range(j, terms):
            block = factor.T @ (outer[:, j, k, None] * factor)
            gram[j, :, k, :] = block
            gram[k, :, j, :] = block.T
    right = np.einsum("ajc,am->jmc", projected, factor)
    return gram.reshape(terms * modes, terms * modes), right.reshape(terms * modes, 3)


def solve_field(gram: np.ndarray, right: np.ndarray, factor: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Posterior mean of the field coefficients ``E``, shape (vertices, terms, 3), at unit noise."""

    vertices, modes = factor.shape
    scales = np.asarray(scales, dtype=np.float64)
    if not np.any(scales > 0):
        return np.zeros((vertices, len(scales), 3))
    d = np.repeat(scales, modes)
    whitened = np.linalg.solve(d[:, None] * gram * d[None, :] + np.eye(len(d)), d[:, None] * right)
    coefficients = (d[:, None] * whitened).reshape(len(scales), modes, 3)
    return np.einsum("am,jmc->ajc", factor, coefficients)


def evaluate_field(coefficients: np.ndarray, phases: np.ndarray, order: int) -> np.ndarray:
    """The field at ``phases``, shape (frames, vertices, 3)."""

    return np.einsum("fj,ajc->fac", temporal_design(phases, order), coefficients)


def field_scales(static: float, dynamic: float, order: int) -> np.ndarray:
    return np.array([float(static)] + [float(dynamic)] * (2 * order))


def config_name(sharing: str, static: float, dynamic: float, width: float) -> str:
    return f"{sharing}/{float(static):g},{float(dynamic):g},{float(width):g}"


@dataclass(frozen=True)
class FieldSettings:
    """One setting of the field: sharing, scales in units of a 1 mm noise, width in mm."""

    sharing: str
    static: float
    dynamic: float
    width: float

    def __post_init__(self) -> None:
        if self.sharing not in SHARINGS:
            raise ValueError(f"sharing must be one of {SHARINGS}")
        if self.static < 0 or self.dynamic < 0 or self.width <= 0:
            raise ValueError("scales cannot be negative and the width must be positive")


@dataclass(frozen=True)
class ResidualField:
    """A fitted field: coefficients per session, shape (vertices, 2 order + 1, 3)."""

    settings: FieldSettings
    order: int
    coefficients: dict[str, np.ndarray]

    def displacement(self, phases: np.ndarray, session: str) -> np.ndarray:
        """The field in the CMR frame at ``phases``, shape (frames, vertices, 3)."""

        return evaluate_field(self.coefficients[session], phases, self.order)

    def geometry(self, fit: SceneFit, phases: np.ndarray, session: str) -> np.ndarray:
        """The fit's physical vertices plus the field, in the CMR frame, (frames, vertices, 3).

        For the TEE session, ``phases`` are cardiac phases; map acquisition times
        through ``fit.parameters.warp()`` first, and place the result in the TEE
        frame with ``fit.parameters.transform()``.
        """

        tau = np.asarray(phases, dtype=np.float64)
        latent = latent_trajectory(fit.parameters, tau, session, fit.model, fit.config)
        return fit.model.physical_coordinates(latent) + self.displacement(tau, session)


def fit_residual_field(fit: SceneFit, settings: FieldSettings) -> ResidualField:
    """The field's posterior mean given a Rooted4D fit; the arm's computation for one setting."""

    order = fit.config.fourier_order
    evidence = field_evidence(fit)
    factor = gaussian_factor(fit.model.template_physical, settings.width)
    scales = field_scales(settings.static, settings.dynamic, order)
    if settings.sharing == "shared":
        shared = solve_field(*normal_equations(list(evidence.values()), factor, order), factor, scales)
        coefficients = {"cmr": shared, "tee": shared}
    else:
        coefficients = {name: solve_field(*normal_equations([evidence[name]], factor, order), factor, scales)
                        for name in ("cmr", "tee")}
    return ResidualField(settings=settings, order=order, coefficients=coefficients)


def fit_rooted4d_with_field(
    observations,
    model,
    config,
    settings: FieldSettings,
    *,
    alignment: StartingAlignment,
) -> tuple[SceneFit, ResidualField]:
    """Rooted4D's fit from the input-derived start, then the field: the main configuration's method."""

    fit = fit_joint_scene(observations, model, config, start=_start(alignment, model, config))
    return fit, fit_residual_field(fit, settings)


def run_rooted4d_residual_field(
    problem: JointProblem,
    config,
    *,
    alignment: StartingAlignment | None,
    configs: list[tuple[str, float, float, float]],
) -> ArmResult:
    """``configs`` lists (sharing, static scale, dynamic scale, width); the first is the arm's."""

    if alignment is None:
        return _failure("rooted4d_residual_field", "needs_the_input_derived_start")
    model, observations = problem.model, problem.observations
    fit = fit_joint_scene(observations, model, config, start=_start(alignment, model, config))
    base = _score("rooted4d_residual_field", fit, problem)
    parameters, order = fit.parameters, config.fourier_order
    evidence = field_evidence(fit)

    # Rooted4D's own geometry where every endpoint is read; each field is added to it.
    cmr_times = np.asarray(observations.heldout_cmr_times, dtype=np.float64)
    cmr_rows = model.observation_rows(np.asarray(observations.heldout_cmr_vertices))
    cmr_base = model.physical_coordinates(latent_trajectory(parameters, cmr_times, "cmr", model, config))
    tee_phases = np.asarray(parameters.warp().evaluate(np.asarray(observations.heldout_tee_times)), dtype=np.float64)
    tee_rows = model.observation_rows(np.asarray(observations.heldout_tee_vertices))
    tee_base = model.physical_coordinates(latent_trajectory(parameters, tee_phases, "tee", model, config))
    gap_base = model.physical_coordinates(latent_trajectory(parameters, GAP_PHASES, "cmr", model, config))
    transform = parameters.transform()

    def scored(cmr_field: np.ndarray, tee_field: np.ndarray) -> dict:
        out = {}
        if len(cmr_times):
            out["heldout_cmr_mm"], out["heldout_cmr_p95_mm"] = _mean_and_p95(
                (cmr_base + evaluate_field(cmr_field, cmr_times, order))[:, cmr_rows],
                observations.heldout_cmr_points)
        if len(tee_phases):
            out["heldout_tee_mm"], out["heldout_tee_p95_mm"] = _mean_and_p95(
                transform.apply((tee_base + evaluate_field(tee_field, tee_phases, order))[:, tee_rows]),
                observations.heldout_tee_points)
        positions = gap_base + evaluate_field(cmr_field, GAP_PHASES, order)
        out["attachment_residual_mm"] = float(max(
            model.attachment_map.max_residual(model.attachment_map.assemble(row)) for row in positions))
        field = evaluate_field(cmr_field, GAP_PHASES, order)
        out["field_rms_mm"] = float(np.sqrt(np.mean((field ** 2).sum(-1))))
        return out

    by_config, systems = {}, {}
    for sharing, static, dynamic, width in configs:
        name = config_name(sharing, static, dynamic, width)
        if width not in systems:
            factor = gaussian_factor(model.template_physical, width)
            systems[width] = (factor, {
                "shared": normal_equations(list(evidence.values()), factor, order),
                "cmr": normal_equations([evidence["cmr"]], factor, order),
                "tee": normal_equations([evidence["tee"]], factor, order)})
        factor, equations = systems[width]
        scales = field_scales(static, dynamic, order)
        try:
            if sharing == "shared":
                cmr_field = tee_field = solve_field(*equations["shared"], factor, scales)
            elif sharing == "session":
                cmr_field = solve_field(*equations["cmr"], factor, scales)
                tee_field = solve_field(*equations["tee"], factor, scales)
            else:
                raise ValueError(f"unknown sharing: {sharing}")
            by_config[name] = scored(cmr_field, tee_field)
        except np.linalg.LinAlgError:
            by_config[name] = {"heldout_cmr_mm": float("nan"), "heldout_tee_mm": float("nan")}

    primary = by_config[config_name(*configs[0])]
    finite = bool(base.succeeded) and all(np.isfinite(primary.get(k, np.nan))
                                          for k in ("heldout_cmr_mm", "heldout_tee_mm"))
    return replace(
        base,
        succeeded=finite,
        failure_reason=base.failure_reason if base.failure_reason else (None if finite else "non_finite_score"),
        attachment_residual_mm=primary.get("attachment_residual_mm", float("nan")),
        heldout_cmr_mm=primary.get("heldout_cmr_mm"),
        heldout_tee_mm=primary.get("heldout_tee_mm"),
        heldout_cmr_p95_mm=primary.get("heldout_cmr_p95_mm"),
        heldout_tee_p95_mm=primary.get("heldout_tee_p95_mm"),
        extras={"by_config": by_config, "configurations": [config_name(*c) for c in configs],
                "rooted4d": {"heldout_cmr_mm": base.heldout_cmr_mm, "heldout_tee_mm": base.heldout_tee_mm,
                             "attachment_residual_mm": base.attachment_residual_mm},
                "modes": {f"{w:g}": int(f.shape[1]) for w, (f, _) in systems.items()}},
    )
