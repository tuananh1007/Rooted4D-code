r"""Fusion designs built from recent registration and shape-model methods (E115).

Pre-registered in
``Docs/E115_RECENT_FUSION_COMPARATORS_PREREGISTRATION_2026-10-22.md``. Every
arm sees the observations, held-out masks and input-derived start every other
arm sees, and is scored by the same endpoints.

``late_fusion_bcpd``, late fusion with a Bayesian coherent point drift attachment
    Late fusion's two reconstructions, timing and rigid registration
    (``run_late_fusion_trajectory``), then, at each phase, BCPD (Hirose, TPAMI
    2021) registers the TEE annulus onto the CMR annulus without
    correspondence, with the non-rigid model ``T(y) = y + v`` (``-Tn``), both
    point sets normalized by the target (``-ux``), the initial width scaled by
    0.1 (``-g0.1``; the authors' advice for roughly aligned inputs), no
    outliers and equal fixed mixing weights (the defaults). The rest of the TEE
    anatomy moves with the GP posterior mean of ``v``, the same expression BCPD
    uses at the annulus. The CMR chamber is not deformed.

``late_fusion_ndp``, late fusion with a neural deformation pyramid attachment
    The same, with NDP (Li and Harada, NeurIPS 2022) fitted to the known
    annulus correspondences, its landmark mode (``ndp.py``).

``gpmm_fusion``, model-based fusion with a Gaussian process morphable model
    One GPMM (Luthi et al., TPAMI 2018) in the CMR frame, its kernel the
    model's shape covariance plus a Gaussian kernel, fitted to both modalities
    by alternating the GP posterior mean with a rigid update of the TEE frame.
    Static, as the model-based design of E114: each visible CMR frame is
    fused with the TEE frame nearest in phase under the shared start's timing,
    and held-out frames are interpolated over phase.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np

from rooted4d.geometry.rigid import RigidTransform
from rooted4d.motion import MonotoneCircularPhaseWarp, circular_distance

from .baselines import ArmResult, _late_fusion_fits, _matrix_to_axis_angle
from .initialization import StartingAlignment
from .joint import SolverConfig, initial_parameters
from .phase_pose import kabsch
from .problem import JointProblem, pose_errors
from .published_fusion import _failure, _mean_and_p95
from .trajectory_registration import (
    inverse_phase,
    register_trajectories,
    registered_interface_gap,
    session_free_trajectory,
)

#: Phases at which the interface gap is measured, as for every other arm.
GAP_PHASES = np.linspace(0.0, 1.0, 32, endpoint=False)


# --------------------------------------------------------------------------
# curves


def ring_order(ring: np.ndarray) -> np.ndarray:
    """Indices that order ring points by angle about their centroid in their best-fit plane."""

    ring = np.asarray(ring, dtype=np.float64)
    centered = ring - ring.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    return np.argsort(np.arctan2(centered @ axes[1], centered @ axes[0]))


def _to_polyline(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Distance from each point to the closed polyline through ``polyline`` in order."""

    start = polyline
    edge = np.roll(polyline, -1, axis=0) - start
    offset = points[:, None, :] - start[None]
    t = np.clip((offset * edge[None]).sum(-1) / np.maximum((edge ** 2).sum(-1), 1e-300)[None], 0.0, 1.0)
    return np.linalg.norm(offset - t[..., None] * edge[None], axis=-1).min(axis=1)


def curve_gap(first: np.ndarray, second: np.ndarray) -> float:
    """Symmetric Hausdorff distance between two closed polylines given in order."""

    return float(max(_to_polyline(first, second).max(), _to_polyline(second, first).max()))


# --------------------------------------------------------------------------
# BCPD, non-rigid model, full rank, as register/bcpd.c computes it


def bcpd_nonrigid(source: np.ndarray, target: np.ndarray, *, beta: float, lam: float, gamma: float = 0.1,
                  omega: float = 0.0, tolerance: float = 1e-4, max_iterations: int = 500,
                  min_iterations: int = 30) -> dict:
    """BCPD with ``T(y) = y + v``, ``-ux`` normalization and fixed equal mixing weights.

    Follows ``bcpd()`` in the authors' ``register/bcpd.c`` with the options
    ``-Tn -ux -g<gamma> -b<beta> -l<lam> -w<omega>`` and no acceleration: the
    posterior-covariance term is off (no ``-a``), so ``b_m = 1/M``; an omega of
    zero becomes 1e-250 as in ``main.c``. One departure: a width below the
    tolerance stops the loop at any iteration, not only after the minimum
    number (30). Two rings that already agree reach zero width within a few
    iterations, and the authors' next iteration would divide by it; this is
    the failure their README answers with ``-N1``. Returns the deformed source and the
    displacement field, both in the input unit. ``field(points)`` is the GP
    posterior mean of ``v`` at any points, which at the source is ``v`` itself.
    Raises ``numpy.linalg.LinAlgError`` where the authors' code aborts.
    """

    X0, Y0 = np.asarray(target, dtype=np.float64), np.asarray(source, dtype=np.float64)
    N, D = X0.shape
    M = len(Y0)
    mu = X0.mean(axis=0)
    scale = float(np.sqrt(((X0 - mu) ** 2).sum() / (N * D)))
    X, Y = (X0 - mu) / scale, (Y0 - mu) / scale
    omega = 1e-250 if omega == 0 else omega

    def kernel(a, b, width):
        return np.exp(-((a[:, None, :] - b[None, :, :]) ** 2).sum(-1) / (2.0 * width ** 2))

    G = kernel(Y, Y, beta)
    r = np.sqrt(max(N * (Y ** 2).sum() + M * (X ** 2).sum() - 2.0 * (Y.sum(0) * X.sum(0)).sum(), 0.0)
                / (M * N * D)) * gamma
    volume = float(np.prod(X.max(axis=0) - X.min(axis=0)))
    a = np.full(M, 1.0 / M)
    y = Y.copy()
    coefficients = np.zeros_like(Y)
    loops = 0
    for loops in range(max_iterations):
        det = np.exp(-0.5 * D * (np.log(2.0 * np.pi) + 2.0 * np.log(r)))
        c = (1.0 / volume) * (omega / (1.0 - omega)) / det
        K = kernel(y, X, r)                                   # M x N
        q = 1.0 / (a @ K + c)
        f = 1.0 - q * c
        w = np.maximum(a * (K @ q), 1e-20)
        PX = a[:, None] * (K @ (q[:, None] * X))
        Np = w.sum()
        xhat = PX / w[:, None]
        cc = lam * r ** 2
        factor = np.linalg.cholesky(G + np.diag(cc / w))
        coefficients = np.linalg.solve(factor.T, np.linalg.solve(factor, xhat - Y))
        y = Y + G @ coefficients
        old = r
        r2 = ((X ** 2).sum(1) @ f + (y ** 2).sum(1) @ w - 2.0 * (PX * y).sum()) / (Np * D)
        r = float(np.sqrt(abs(r2)))
        if r < tolerance or (loops > min_iterations and abs(old - r) < tolerance):
            break
    if not np.all(np.isfinite(y)):
        raise np.linalg.LinAlgError("non-finite BCPD state")

    def field(points):
        flat = (np.asarray(points, dtype=np.float64).reshape(-1, D) - mu) / scale
        return (kernel(flat, Y, beta) @ coefficients * scale).reshape(np.shape(points))

    return {"deformed": y * scale + mu, "field": field, "iterations": loops + 1, "sigma": r * scale,
            "stopped_at_zero_width": bool(r < tolerance and loops <= min_iterations)}


# --------------------------------------------------------------------------
# attachment after late fusion (BCPD, NDP)


AttachMethod = Callable[[np.ndarray, np.ndarray, np.ndarray], dict]


def bcpd_attachment(grid: list[tuple[float, float]], gamma: float = 0.1) -> AttachMethod:
    """Every (beta, lambda) of ``grid``; a configuration whose fit aborts scores NaN."""

    def apply(source, target, queries):
        out = {}
        for beta, lam in grid:
            try:
                fit = bcpd_nonrigid(source, target, beta=beta, lam=lam, gamma=gamma)
                out[f"{beta:g},{lam:g}"] = (fit["deformed"], queries + fit["field"](queries))
            except np.linalg.LinAlgError:
                out[f"{beta:g},{lam:g}"] = None
        return out

    return apply


def ndp_attachment(levels: int, seed: int = 0, **settings) -> AttachMethod:
    """Every pyramid of 1 to ``levels`` levels, read from one fit; ``settings`` override NDP's."""

    from .ndp import landmark_pyramid

    def apply(source, target, queries):
        per_level = landmark_pyramid(source, target, [source, queries], levels=levels, seed=seed, **settings)
        return {f"{m + 1}": (ring, moved) for m, (ring, moved) in enumerate(per_level)}

    return apply


def run_registration_attachment(
    baseline_id: str,
    problem: JointProblem,
    config: SolverConfig,
    *,
    alignment: StartingAlignment | None,
    modality_overrides: dict | None,
    method: AttachMethod,
    primary: str,
) -> ArmResult:
    """Late fusion, then ``method`` attaches the TEE anatomy to the CMR annulus at each phase.

    ``primary`` names the configuration reported in the arm's own fields;
    every configuration is scored into ``extras["by_config"]``.
    """

    observations, model, truth = problem.observations, problem.model, problem.truth
    cmr_fit, tee_fit, cmr_config, tee_config = _late_fusion_fits(problem, config, alignment, modality_overrides)
    if not (np.isfinite(cmr_fit.cost) and np.isfinite(tee_fit.cost)):
        return _failure(baseline_id, "non_finite_cost")
    vertices = np.union1d(np.intersect1d(observations.cmr_vertices, observations.tee_vertices),
                          np.sort(model.groups["annulus"]))
    registration = register_trajectories(cmr_fit.parameters, tee_fit.parameters, model, cmr_config, vertices)
    if not np.isfinite(registration.cost):
        return _failure(baseline_id, "non_finite_cost")
    G, chi = registration.transform, registration.warp
    cmr, tee = cmr_fit.parameters, tee_fit.parameters
    tee_pose, tee_warp = tee.transform(), tee.warp()

    template = model.template_physical
    ring = np.sort(model.groups["annulus"])
    ring = ring[ring_order(template[model.observation_rows(ring)])]
    leaflet = np.setdiff1d(np.asarray(observations.tee_vertices), ring)
    ring_rows, leaflet_rows = model.observation_rows(ring), model.observation_rows(leaflet)
    nearest = leaflet[np.argmin(np.linalg.norm(
        template[ring_rows][:, None] - template[leaflet_rows][None], axis=-1), axis=1)]
    hinge_distance = float(np.median(np.linalg.norm(
        template[ring_rows] - template[model.observation_rows(nearest)], axis=-1)))
    heldout_vertices = np.asarray(observations.heldout_tee_vertices)
    queries = np.concatenate([ring, nearest, heldout_vertices])
    n_ring, n_hinge = len(ring), len(nearest)

    def tee_at(phases, which):
        return G.apply(session_free_trajectory(tee, model, tee_config, phases, which))

    def cmr_at(phases, which):
        return session_free_trajectory(cmr, model, cmr_config, phases, which)

    # (TEE phase, CMR phase) pairs: the 32 gap phases, then each held-out TEE frame.
    heldout_times = np.asarray(observations.heldout_tee_times, dtype=np.float64)
    heldout_tee = np.asarray(tee_warp.evaluate(heldout_times), dtype=np.float64)
    tee_phases = np.concatenate([inverse_phase(chi, GAP_PHASES), heldout_tee])
    cmr_phases = np.concatenate([GAP_PHASES, np.asarray(chi.evaluate(heldout_tee), dtype=np.float64)])
    moving = tee_at(tee_phases, queries)
    fixed = cmr_at(cmr_phases, ring)

    fits = [method(moving[i, :n_ring], fixed[i], moving[i]) for i in range(len(tee_phases))]
    names = list(fits[0])
    inverse = G.inverse()
    observed = np.asarray(observations.heldout_tee_points, dtype=np.float64)
    gap_count = len(GAP_PHASES)

    def held_out_error(moved):
        valve = moved[gap_count:, n_ring + n_hinge:]
        return _mean_and_p95(tee_pose.apply(inverse.apply(valve)), observed)

    by_config = {}
    for name in names:
        if any(fit[name] is None for fit in fits):
            by_config[name] = {"heldout_tee_mm": float("nan"), "heldout_tee_p95_mm": float("nan"),
                               "attachment_residual_mm": float("nan"), "curve_gap_mm": float("nan"),
                               "hinge_shear_mm": float("nan"), "aborted_fits": sum(f[name] is None for f in fits)}
            continue
        moved = np.stack([fit[name][1] for fit in fits])
        displacement = moved - moving
        mean, p95 = held_out_error(moved)
        by_config[name] = {
            "heldout_tee_mm": mean, "heldout_tee_p95_mm": p95,
            "attachment_residual_mm": float(np.abs(moved[:gap_count, :n_ring] - fixed[:gap_count]).max()),
            "curve_gap_mm": max(curve_gap(moved[i, :n_ring], fixed[i]) for i in range(gap_count)),
            "hinge_shear_mm": float(np.linalg.norm(
                displacement[:gap_count, :n_ring] - displacement[:gap_count, n_ring:n_ring + n_hinge],
                axis=-1).max()),
        }
    undeformed, _ = held_out_error(moving)
    chosen = by_config[primary]

    pose = registration.place(tee_pose)
    times = np.asarray(observations.tee_times, dtype=np.float64)
    offset = float(chi.evaluate(tee_warp.evaluate(np.array([0.0])))[0])
    estimated = replace(tee, pose_axis_angle=_matrix_to_axis_angle(pose.rotation),
                        pose_translation=np.asarray(pose.translation, dtype=np.float64), phase_offset=offset)
    errors = pose_errors(estimated, truth)
    cmr_errors = cmr_fit.heldout_errors()
    succeeded = bool(np.isfinite(chosen["heldout_tee_mm"]))
    return ArmResult(
        baseline_id=baseline_id,
        succeeded=succeeded,
        failure_reason=None if succeeded else "aborted_or_non_finite",
        attachment_residual_mm=chosen["attachment_residual_mm"],
        heldout_cmr_mm=cmr_errors.get("heldout_cmr_mm"),
        heldout_tee_mm=chosen["heldout_tee_mm"],
        rotation_deg=errors["rotation_deg"],
        translation_mm=errors["translation_mm"],
        phase_error_cycles=float(circular_distance(np.array([offset]), np.array([truth.phase_offset]))[0]),
        cost=float(cmr_fit.cost + tee_fit.cost),
        iterations=int(cmr_fit.iterations + tee_fit.iterations + registration.iterations),
        heldout_cmr_p95_mm=cmr_errors.get("heldout_cmr_p95_mm"),
        heldout_tee_p95_mm=chosen["heldout_tee_p95_mm"],
        phase_map_cycles=float(np.median(circular_distance(
            np.asarray(chi.evaluate(tee_warp.evaluate(times)), dtype=np.float64),
            np.asarray(truth.warp().evaluate(times), dtype=np.float64)))),
        repair_registration_rms_mm=registration.rms_mm,
        extras={"by_config": by_config, "configurations": names, "primary": primary,
                "undeformed_heldout_tee_mm": undeformed,
                "late_fusion_gap_mm": registered_interface_gap(cmr, tee, registration, model, cmr_config),
                "late_fusion_curve_gap_mm": max(curve_gap(moving[i, :n_ring], fixed[i]) for i in range(gap_count)),
                "hinge_template_distance_mm": hinge_distance},
    )


# --------------------------------------------------------------------------
# GPMM fusion


def gpmm_kernel(model, gaussian_scale: float, gaussian_width: float, scale_floor: float) -> np.ndarray:
    """The model's shape covariance plus ``s^2 exp(-d^2 / 2w^2) I`` on the template, (3n, 3n)."""

    basis = np.asarray(model.shape_basis, dtype=np.float64)
    modes = basis.shape[1]
    scales = (np.ones(modes) if model.shape_scales is None
              else np.maximum(np.asarray(model.shape_scales, dtype=np.float64),
                              scale_floor * float(np.max(model.shape_scales))))
    kernel = (basis * scales ** 2) @ basis.T
    if gaussian_scale > 0:
        template = model.template_physical
        squared = ((template[:, None] - template[None]) ** 2).sum(-1)
        kernel = kernel + np.kron(gaussian_scale ** 2 * np.exp(-squared / (2.0 * gaussian_width ** 2)), np.eye(3))
    return kernel


def _coordinates(rows: np.ndarray) -> np.ndarray:
    return (3 * np.asarray(rows)[:, None] + np.arange(3)[None]).ravel()


def fit_gpmm_fusion(model, observations, alignment: StartingAlignment, *, gaussian_scale: float,
                    gaussian_width: float, noise: float, iterations: int = 25,
                    scale_floor: float = 0.005) -> dict:
    """Per-frame GP posterior means and one rigid map from the TEE frame, alternated."""

    template = model.template_physical
    n = len(template)
    cmr_rows = model.observation_rows(np.asarray(observations.cmr_vertices))
    tee_rows = model.observation_rows(np.asarray(observations.tee_vertices))
    observed = np.concatenate([_coordinates(cmr_rows), _coordinates(tee_rows)])
    kernel = gpmm_kernel(model, gaussian_scale, gaussian_width, scale_floor)
    factor = np.linalg.cholesky(kernel[np.ix_(observed, observed)] + noise ** 2 * np.eye(len(observed)))
    cross = kernel[:, observed]

    timing = MonotoneCircularPhaseWarp(np.zeros(model.n_phase_nodes), offset=alignment.phase_offset)
    tee_phases = np.asarray(timing.evaluate(np.asarray(observations.tee_times)), dtype=np.float64)
    cmr_phases = np.mod(np.asarray(observations.cmr_times, dtype=np.float64), 1.0)
    pairing = np.array([int(np.argmin(circular_distance(tee_phases, np.full_like(tee_phases, tau))))
                        for tau in cmr_phases])
    cmr_points = np.asarray(observations.cmr_points, dtype=np.float64)
    tee_points = np.asarray(observations.tee_points, dtype=np.float64)[pairing]
    frames = len(cmr_phases)
    flat_tee = tee_points.reshape(-1, 3)
    cmr_residual = (cmr_points - template[cmr_rows]).reshape(frames, -1)

    def posterior(rigid):
        placed = rigid.apply(flat_tee).reshape(tee_points.shape)
        residual = np.concatenate([cmr_residual, (placed - template[tee_rows]).reshape(frames, -1)], axis=1)
        alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, residual.T))
        return (cross @ alpha).T.reshape(frames, n, 3)

    # The shared start's TEE pose, inverted: TEE frame to CMR frame, as in E114's model-based fusion.
    rigid = RigidTransform.from_axis_angle(alignment.pose_axis_angle, alignment.pose_translation).inverse()
    for _ in range(iterations):
        deformation = posterior(rigid)
        rigid = kabsch(flat_tee, (template[tee_rows][None] + deformation[:, tee_rows]).reshape(-1, 3))
    deformation = posterior(rigid)
    order = np.argsort(cmr_phases)
    return {"phases": cmr_phases[order], "deformations": deformation[order], "rigid": rigid,
            "timing": timing, "pairing": pairing}


def circular_interpolate(known: np.ndarray, values: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """Linear interpolation over the circle of ``values`` (frames, ...) known at sorted ``known``."""

    known = np.asarray(known, dtype=np.float64)
    phases = np.mod(np.asarray(phases, dtype=np.float64), 1.0)
    if len(known) == 1:
        return np.repeat(values[:1], len(phases), axis=0)
    extended = np.concatenate([known[-1:] - 1.0, known, known[:1] + 1.0])
    stacked = np.concatenate([values[-1:], values, values[:1]])
    upper = np.clip(np.searchsorted(extended, phases, side="right"), 1, len(extended) - 1)
    lower = upper - 1
    span = extended[upper] - extended[lower]
    weight = np.where(span > 0, (phases - extended[lower]) / np.where(span > 0, span, 1.0), 0.0)
    shape = (-1,) + (1,) * (values.ndim - 1)
    return stacked[lower] * (1.0 - weight.reshape(shape)) + stacked[upper] * weight.reshape(shape)


def score_gpmm_fusion(fit: dict, model, observations, truth, config: SolverConfig) -> dict:
    """Every endpoint of one GPMM fusion fit."""

    template = model.template_physical

    def geometry(phases, vertices):
        positions = template[None] + circular_interpolate(fit["phases"], fit["deformations"], phases)
        return positions[:, model.observation_rows(np.asarray(vertices)), :]

    out = {}
    if len(observations.heldout_cmr_times):
        out["heldout_cmr_mm"], out["heldout_cmr_p95_mm"] = _mean_and_p95(
            geometry(observations.heldout_cmr_times, observations.heldout_cmr_vertices),
            observations.heldout_cmr_points)
    if len(observations.heldout_tee_times):
        phases = fit["timing"].evaluate(np.asarray(observations.heldout_tee_times))
        out["heldout_tee_mm"], out["heldout_tee_p95_mm"] = _mean_and_p95(
            fit["rigid"].inverse().apply(geometry(phases, observations.heldout_tee_vertices)),
            observations.heldout_tee_points)
    positions = template[None] + circular_interpolate(fit["phases"], fit["deformations"], GAP_PHASES)
    out["attachment_residual_mm"] = float(max(
        model.attachment_map.max_residual(model.attachment_map.assemble(row)) for row in positions))
    pose = fit["rigid"].inverse()
    estimated = replace(initial_parameters(model, config),
                        pose_axis_angle=_matrix_to_axis_angle(pose.rotation),
                        pose_translation=np.asarray(pose.translation),
                        phase_offset=float(fit["timing"].offset))
    out.update(pose_errors(estimated, truth))
    times = np.asarray(observations.tee_times, dtype=np.float64)
    out["phase_map_cycles"] = float(np.median(circular_distance(
        np.asarray(fit["timing"].evaluate(times), dtype=np.float64),
        np.asarray(truth.warp().evaluate(times), dtype=np.float64))))
    out["phase_error_cycles"] = float(circular_distance(
        np.array([fit["timing"].offset]), np.array([truth.phase_offset]))[0])
    return out


def run_gpmm_fusion(
    problem: JointProblem,
    config: SolverConfig,
    *,
    alignment: StartingAlignment | None,
    grid: list[tuple[float, float, float]],
    iterations: int = 25,
) -> ArmResult:
    """``grid`` lists (Gaussian scale, Gaussian width, noise) in mm; the first is the arm's."""

    if alignment is None:
        return _failure("gpmm_fusion", "needs_the_input_derived_start")
    by_config = {}
    for scale, width, noise in grid:
        name = f"{float(scale):g},{float(width):g},{float(noise):g}"
        try:
            fit = fit_gpmm_fusion(problem.model, problem.observations, alignment,
                                  gaussian_scale=float(scale), gaussian_width=float(width), noise=float(noise),
                                  iterations=iterations, scale_floor=config.identity_prior_scale_floor)
            by_config[name] = score_gpmm_fusion(fit, problem.model, problem.observations, problem.truth, config)
        except np.linalg.LinAlgError:
            by_config[name] = {"heldout_cmr_mm": float("nan"), "heldout_tee_mm": float("nan")}
    first = grid[0]
    primary = by_config[f"{float(first[0]):g},{float(first[1]):g},{float(first[2]):g}"]
    finite = all(np.isfinite(primary.get(k, np.nan)) for k in ("heldout_cmr_mm", "heldout_tee_mm"))
    nan = float("nan")
    return ArmResult(
        baseline_id="gpmm_fusion",
        succeeded=bool(finite),
        failure_reason=None if finite else "non_finite_score",
        attachment_residual_mm=primary.get("attachment_residual_mm", nan),
        heldout_cmr_mm=primary.get("heldout_cmr_mm"),
        heldout_tee_mm=primary.get("heldout_tee_mm"),
        rotation_deg=primary.get("rotation_deg"),
        translation_mm=primary.get("translation_mm"),
        phase_error_cycles=primary.get("phase_error_cycles"),
        cost=0.0,
        iterations=int(iterations),
        heldout_cmr_p95_mm=primary.get("heldout_cmr_p95_mm"),
        heldout_tee_p95_mm=primary.get("heldout_tee_p95_mm"),
        phase_map_cycles=primary.get("phase_map_cycles"),
        extras={"by_config": by_config, "grid": [list(map(float, g)) for g in grid]},
    )
