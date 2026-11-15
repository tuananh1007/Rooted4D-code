r"""Two published fusion designs, re-implemented on point evidence (E114).

Pre-registered in
``Docs/E114_PUBLISHED_FUSION_BASELINES_PREREGISTRATION_2026-10-22.md``. Both
arms see the observations, held-out masks and input-derived start every other
arm sees, and are scored by the same endpoints.

``spatiotemporal_registration``, adapted from Zhang et al. (Phys Med Biol 2011)
    The deformation between the modalities is decoupled into a temporal part
    (frame correspondence from cardiac-phase landmarks, piecewise linear) and
    a spatial part (a rigid map, then a non-rigid deformation of each
    temporally aligned pair, MR as the reference). The reconstructions are late
    fusion's own fits; the landmarks are the phases of maximum and minimum
    area enclosed by each fit's annulus; the non-rigid step is a Gaussian
    radial-basis field interpolating the shared vertices' displacements.

``model_based_fusion``, adapted from Grbic et al. (MICCAI 2011)
    One point-distribution model of the anatomy in the reference (CMR) frame
    and a transform ``phi = D o A`` from the TEE frame, ``A`` rigid and ``D`` a
    Gaussian radial-basis warp with few centers, estimated by alternating the
    model update (ridge regression on the PCA coefficients) with the rigid and
    the warp updates. The method is static, so each visible CMR frame is fused
    with the TEE frame nearest in phase under the shared start's timing; held-
    out frames are predicted by interpolating the per-frame coefficients.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from rooted4d.geometry.rigid import RigidTransform
from rooted4d.motion import MonotoneCircularPhaseWarp, circular_distance

from .baselines import ArmResult, _late_fusion_fits, _matrix_to_axis_angle
from .initialization import StartingAlignment
from .joint import SolverConfig, initial_parameters, latent_trajectory
from .phase_pose import kabsch
from .problem import JointProblem, pose_errors

#: Phases at which the interface gap is measured, as for every other arm.
GAP_PHASES = np.linspace(0.0, 1.0, 32, endpoint=False)
#: Phases at which B1's temporal landmarks are searched and its rigid map fitted.
LANDMARK_GRID = np.linspace(0.0, 1.0, 256, endpoint=False)
REGISTRATION_PHASES = np.linspace(0.0, 1.0, 32, endpoint=False)
#: Relative ridge on the radial-basis interpolation systems, for conditioning.
INTERPOLATION_RIDGE = 1e-6


# --------------------------------------------------------------------------
# shared helpers


def _trajectory(parameters, model, config: SolverConfig, phases, vertices) -> np.ndarray:
    """Session-free coordinates of ``vertices`` at ``phases``: (phases, vertices, 3)."""

    tau = np.mod(np.asarray(phases, dtype=np.float64), 1.0)
    latent = latent_trajectory(parameters, tau, "cmr", model, replace(config, use_session=False))
    return model.physical_coordinates(latent)[:, model.observation_rows(np.asarray(vertices)), :]


def ring_area(ring: np.ndarray) -> np.ndarray:
    """Area enclosed by ordered rings, (..., n, 3) -> (...), by the vector-area formula."""

    centered = ring - ring.mean(axis=-2, keepdims=True)
    following = np.roll(centered, -1, axis=-2)
    return 0.5 * np.linalg.norm(np.cross(centered, following).sum(axis=-2), axis=-1)


def piecewise_circular_map(x, source: tuple[float, float], target: tuple[float, float]) -> np.ndarray:
    """The monotone circular map taking ``source[i]`` to ``target[i]``, linear in between.

    The arc from ``source[0]`` forward to ``source[1]`` goes onto the arc from
    ``target[0]`` forward to ``target[1]``, and the remaining arc onto the
    remaining arc. Coincident landmarks degrade to a pure offset.
    """

    x = np.mod(np.asarray(x, dtype=np.float64), 1.0)
    first_source = float(np.mod(source[1] - source[0], 1.0))
    first_target = float(np.mod(target[1] - target[0], 1.0))
    if not (1e-6 < first_source < 1.0 - 1e-6) or not (1e-6 < first_target < 1.0 - 1e-6):
        return np.mod(x - source[0] + target[0], 1.0)
    u = np.mod(x - source[0], 1.0)
    inside = u < first_source
    mapped = np.where(
        inside,
        target[0] + u / first_source * first_target,
        target[1] + (u - first_source) / (1.0 - first_source) * (1.0 - first_target),
    )
    return np.mod(mapped, 1.0)


def _spaced(points: np.ndarray, spacing: float) -> np.ndarray:
    """Indices of a subset at least ``spacing`` apart, greedily by farthest point from index 0."""

    chosen = [0]
    distance = np.linalg.norm(points - points[0], axis=1)
    while distance.max() >= spacing:
        chosen.append(int(np.argmax(distance)))
        distance = np.minimum(distance, np.linalg.norm(points - points[chosen[-1]], axis=1))
    return np.array(chosen)


def _kernel(points: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    squared = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    return np.exp(-squared / (2.0 * width ** 2))


class RadialField:
    """Gaussian radial-basis displacement field through given displacements.

    Exact Gaussian interpolation through closely spaced nodes is ill-conditioned
    once the width exceeds the node spacing: the weights grow without bound and
    the field swings far from the nodes (E114, second addendum). A field wider
    than the node spacing is therefore built in two levels: a least-squares fit
    at the requested width on centers at least one width apart, which carries
    the smooth part, and an exact interpolation of the remainder at the node
    spacing, which closes it at the nodes and decays within a few spacings.
    """

    def __init__(self, sources: np.ndarray, displacements: np.ndarray, width: float) -> None:
        self.sources = np.asarray(sources, dtype=np.float64)
        self.width = float(width)
        remainder = np.asarray(displacements, dtype=np.float64)
        gaps = np.linalg.norm(self.sources[:, None] - self.sources[None], axis=-1)
        np.fill_diagonal(gaps, np.inf)
        self.spacing = float(np.median(gaps.min(axis=1)))
        self.levels = []
        fine = self.width
        if self.width > self.spacing:
            centers = self.sources[_spaced(self.sources, self.width)]
            design = _kernel(self.sources, centers, self.width)
            normal = design.T @ design
            ridge = INTERPOLATION_RIDGE * max(float(np.trace(normal)) / len(normal), 1.0)
            weights = np.linalg.solve(normal + ridge * np.eye(len(normal)), design.T @ remainder)
            self.levels.append((centers, self.width, weights))
            remainder = remainder - design @ weights
            fine = self.spacing
        kernel = _kernel(self.sources, self.sources, fine)
        ridge = INTERPOLATION_RIDGE * max(float(np.trace(kernel)) / len(kernel), 1.0)
        self.levels.append((self.sources, fine, np.linalg.solve(kernel + ridge * np.eye(len(kernel)), remainder)))

    def __call__(self, points: np.ndarray) -> np.ndarray:
        flat = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        out = sum(_kernel(flat, centers, width) @ weights for centers, width, weights in self.levels)
        return out.reshape(np.shape(points))


def _place(pose: RigidTransform, registration: RigidTransform) -> RigidTransform:
    """The moving acquisition's pose in the reference frame, ``pose o G^-1``."""

    inverse = registration.inverse()
    return RigidTransform(pose.rotation @ inverse.rotation,
                          pose.rotation @ inverse.translation + pose.translation)


def _mean_and_p95(prediction: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    distances = np.linalg.norm(prediction - observed, axis=-1)
    return float(np.mean(distances)), float(np.percentile(distances, 95))


def _failure(baseline_id: str, reason: str) -> ArmResult:
    return ArmResult(baseline_id=baseline_id, succeeded=False, failure_reason=reason,
                     attachment_residual_mm=float("nan"), heldout_cmr_mm=None, heldout_tee_mm=None,
                     rotation_deg=None, translation_mm=None, phase_error_cycles=None,
                     cost=float("inf"), iterations=0)


# --------------------------------------------------------------------------
# B1: decoupled spatio-temporal registration (Zhang et al. 2011)


def run_spatiotemporal_registration(
    problem: JointProblem,
    config: SolverConfig,
    *,
    alignment: StartingAlignment | None,
    modality_overrides: dict | None,
    widths_mm: tuple[float, ...],
) -> ArmResult:
    """B1. The first width is the arm's; every width is scored into ``extras``."""

    observations, model, truth = problem.observations, problem.model, problem.truth
    cmr_fit, tee_fit, cmr_config, tee_config = _late_fusion_fits(
        problem, config, alignment, modality_overrides)
    cmr, tee = cmr_fit.parameters, tee_fit.parameters
    if not (np.isfinite(cmr_fit.cost) and np.isfinite(tee_fit.cost)):
        return _failure("spatiotemporal_registration", "non_finite_cost")

    ring = np.sort(model.groups["annulus"])
    shared = np.union1d(np.intersect1d(observations.cmr_vertices, observations.tee_vertices), ring)

    # Temporal: two landmarks per modality on the one structure both observe.
    cmr_area = ring_area(_trajectory(cmr, model, cmr_config, LANDMARK_GRID, ring))
    tee_area = ring_area(_trajectory(tee, model, tee_config, LANDMARK_GRID, ring))
    source = (float(LANDMARK_GRID[np.argmax(tee_area)]), float(LANDMARK_GRID[np.argmin(tee_area)]))
    target = (float(LANDMARK_GRID[np.argmax(cmr_area)]), float(LANDMARK_GRID[np.argmin(cmr_area)]))

    def to_cmr(phases):
        return piecewise_circular_map(phases, source, target)

    def to_tee(phases):
        return piecewise_circular_map(phases, target, source)

    # Spatial, rigid: shared vertices over the temporally aligned pairs.
    moving = _trajectory(tee, model, tee_config, REGISTRATION_PHASES, shared)
    fixed = _trajectory(cmr, model, cmr_config, to_cmr(REGISTRATION_PHASES), shared)
    registration = kabsch(moving.reshape(-1, 3), fixed.reshape(-1, 3))
    rigid_rms = float(np.sqrt(((registration.apply(moving) - fixed) ** 2).sum(-1).mean()))

    def fused_valve(cmr_phases, vertices, width):
        """The deformed TEE anatomy in the CMR frame at ``cmr_phases`` (None: rigid only)."""

        tee_phases = to_tee(cmr_phases)
        geometry = registration.apply(_trajectory(tee, model, tee_config, tee_phases, vertices))
        if width is None:
            return geometry
        sources = registration.apply(_trajectory(tee, model, tee_config, tee_phases, shared))
        targets = _trajectory(cmr, model, cmr_config, cmr_phases, shared)
        out = np.empty_like(geometry)
        for index in range(len(geometry)):
            field = RadialField(sources[index], targets[index] - sources[index], width)
            out[index] = geometry[index] + field(geometry[index])
        return out

    tee_warp = tee.warp()
    tee_pose = tee.transform()
    heldout_times = np.asarray(observations.heldout_tee_times, dtype=np.float64)
    heldout_cmr_phases = to_cmr(tee_warp.evaluate(heldout_times))
    inverse = registration.inverse()

    def heldout_tee(width):
        valve = fused_valve(heldout_cmr_phases, observations.heldout_tee_vertices, width)
        return _mean_and_p95(tee_pose.apply(inverse.apply(valve)), observations.heldout_tee_points)

    def gap(width):
        chamber = _trajectory(cmr, model, cmr_config, GAP_PHASES, ring)
        return float(np.abs(fused_valve(GAP_PHASES, ring, width) - chamber).max())

    # Hinge shear (pre-registration addendum): the deformation closes the gap
    # by moving the TEE ring; how differently it moves the leaflet vertex
    # nearest each ring vertex (in the template) shows how much of the gap it
    # moved into the leaflet instead of removing it.
    leaflet = np.setdiff1d(np.asarray(observations.tee_vertices), ring)
    template = model.template_physical
    ring_rows, leaflet_rows = model.observation_rows(ring), model.observation_rows(leaflet)
    nearest = leaflet[np.argmin(np.linalg.norm(
        template[ring_rows][:, None] - template[leaflet_rows][None], axis=-1), axis=1)]
    hinge_distance = float(np.median(np.linalg.norm(
        template[ring_rows] - template[model.observation_rows(nearest)], axis=-1)))

    def hinge(width):
        tee_phases = to_tee(GAP_PHASES)
        sources = registration.apply(_trajectory(tee, model, tee_config, tee_phases, shared))
        targets = _trajectory(cmr, model, cmr_config, GAP_PHASES, shared)
        at_ring = registration.apply(_trajectory(tee, model, tee_config, tee_phases, ring))
        at_leaflet = registration.apply(_trajectory(tee, model, tee_config, tee_phases, nearest))
        worst = 0.0
        for index in range(len(GAP_PHASES)):
            field = RadialField(sources[index], targets[index] - sources[index], width)
            shear = np.linalg.norm(field(at_ring[index]) - field(at_leaflet[index]), axis=-1)
            worst = max(worst, float(shear.max()))
        return worst

    cmr_errors = cmr_fit.heldout_errors()
    undeformed, _ = heldout_tee(None)
    by_width = {}
    for width in widths_mm:
        mean, p95 = heldout_tee(float(width))
        by_width[f"{float(width):g}"] = {"heldout_tee_mm": mean, "heldout_tee_p95_mm": p95,
                                         "attachment_residual_mm": gap(float(width)),
                                         "hinge_shear_mm": hinge(float(width))}
    primary = by_width[f"{float(widths_mm[0]):g}"]

    times = np.asarray(observations.tee_times, dtype=np.float64)
    estimated_map = to_cmr(tee_warp.evaluate(times))
    offset = float(to_cmr(tee_warp.evaluate(np.array([0.0])))[0])
    pose = _place(tee_pose, registration)
    estimated = replace(tee, pose_axis_angle=_matrix_to_axis_angle(pose.rotation),
                        pose_translation=np.asarray(pose.translation), phase_offset=offset)
    errors = pose_errors(estimated, truth)
    return ArmResult(
        baseline_id="spatiotemporal_registration",
        succeeded=bool(np.isfinite(primary["heldout_tee_mm"])),
        failure_reason=None if np.isfinite(primary["heldout_tee_mm"]) else "non_finite_score",
        attachment_residual_mm=primary["attachment_residual_mm"],
        heldout_cmr_mm=cmr_errors.get("heldout_cmr_mm"),
        heldout_tee_mm=primary["heldout_tee_mm"],
        rotation_deg=errors["rotation_deg"],
        translation_mm=errors["translation_mm"],
        phase_error_cycles=float(circular_distance(
            np.array([offset]), np.array([truth.phase_offset]))[0]),
        cost=float(cmr_fit.cost + tee_fit.cost),
        iterations=int(cmr_fit.iterations + tee_fit.iterations),
        heldout_cmr_p95_mm=cmr_errors.get("heldout_cmr_p95_mm"),
        heldout_tee_p95_mm=primary["heldout_tee_p95_mm"],
        phase_map_cycles=float(np.median(circular_distance(
            estimated_map, np.asarray(truth.warp().evaluate(times), dtype=np.float64)))),
        repair_registration_rms_mm=rigid_rms,
        extras={"by_width": by_width, "undeformed_heldout_tee_mm": undeformed,
                "landmarks": {"tee_max_min": list(source), "cmr_max_min": list(target)},
                "widths_mm": [float(w) for w in widths_mm],
                "hinge_template_distance_mm": hinge_distance},
    )


# --------------------------------------------------------------------------
# B2: model-based fusion (Grbic et al. 2011)


def farthest_points(points: np.ndarray, count: int) -> np.ndarray:
    """Indices of ``count`` points by farthest-point sampling from the one nearest the centroid."""

    points = np.asarray(points, dtype=np.float64)
    chosen = [int(np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1)))]
    distance = np.linalg.norm(points - points[chosen[0]], axis=1)
    while len(chosen) < min(count, len(points)):
        chosen.append(int(np.argmax(distance)))
        distance = np.minimum(distance, np.linalg.norm(points - points[chosen[-1]], axis=1))
    return np.array(chosen)


def _gaussian(points: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    squared = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    return np.exp(-squared / (2.0 * width ** 2))


def fit_model_based_fusion(model, observations, alignment: StartingAlignment, *,
                           shape_weight: float, warp_ridge: float, centers: int = 12,
                           iterations: int = 25, scale_floor: float = 0.005) -> dict:
    """The alternating estimate of per-frame coefficients, ``A`` and ``D``."""

    n_physical = model.n_physical
    basis = model.shape_basis.reshape(n_physical, 3, -1)
    modes = basis.shape[-1]
    template = model.template_physical
    cmr_rows = model.observation_rows(np.asarray(observations.cmr_vertices))
    tee_rows = model.observation_rows(np.asarray(observations.tee_vertices))
    basis_cmr = basis[cmr_rows].reshape(-1, modes)
    basis_tee = basis[tee_rows].reshape(-1, modes)

    scales = (np.ones(modes) if model.shape_scales is None
              else np.maximum(np.asarray(model.shape_scales, dtype=np.float64),
                              scale_floor * float(np.max(model.shape_scales))))
    normal = basis_cmr.T @ basis_cmr + basis_tee.T @ basis_tee + shape_weight * np.diag(1.0 / scales ** 2)
    factor = np.linalg.cholesky(normal)

    def solve(rhs):
        return np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))

    timing = MonotoneCircularPhaseWarp(np.zeros(model.n_phase_nodes), offset=alignment.phase_offset)
    tee_phases = np.asarray(timing.evaluate(np.asarray(observations.tee_times)), dtype=np.float64)
    cmr_phases = np.mod(np.asarray(observations.cmr_times, dtype=np.float64), 1.0)
    pairing = np.array([int(np.argmin(circular_distance(tee_phases, np.full_like(tee_phases, tau))))
                        for tau in cmr_phases])
    cmr_points = np.asarray(observations.cmr_points, dtype=np.float64)
    tee_points = np.asarray(observations.tee_points, dtype=np.float64)[pairing]
    frames = len(cmr_phases)

    anchors = template[tee_rows][farthest_points(template[tee_rows], centers)]
    gaps = np.linalg.norm(anchors[:, None] - anchors[None], axis=-1) + np.diag(np.full(len(anchors), np.inf))
    width = float(np.median(gaps.min(axis=1)))

    rigid = RigidTransform.from_axis_angle(alignment.pose_axis_angle, alignment.pose_translation).inverse()
    weights = np.zeros((len(anchors), 3))
    cmr_target = (cmr_points - template[cmr_rows]).reshape(frames, -1)
    flat_tee = tee_points.reshape(-1, 3)
    coefficients = np.zeros((frames, modes))
    for _ in range(iterations):
        placed = rigid.apply(flat_tee)
        design = _gaussian(placed, anchors, width)
        mapped = (placed + design @ weights).reshape(tee_points.shape)
        rhs = basis_cmr.T @ cmr_target.T + basis_tee.T @ (mapped - template[tee_rows]).reshape(frames, -1).T
        coefficients = solve(rhs).T
        model_tee = (template[tee_rows][None] + (coefficients @ basis_tee.T).reshape(frames, -1, 3)).reshape(-1, 3)
        rigid = kabsch(flat_tee, model_tee - design @ weights)
        placed = rigid.apply(flat_tee)
        design = _gaussian(placed, anchors, width)
        weights = np.linalg.solve(design.T @ design + warp_ridge * np.eye(len(anchors)),
                                  design.T @ (model_tee - placed))
    placed = rigid.apply(flat_tee)
    mapped = (placed + _gaussian(placed, anchors, width) @ weights).reshape(tee_points.shape)
    rhs = basis_cmr.T @ cmr_target.T + basis_tee.T @ (mapped - template[tee_rows]).reshape(frames, -1).T
    coefficients = solve(rhs).T
    order = np.argsort(cmr_phases)
    return {"phases": cmr_phases[order], "coefficients": coefficients[order], "rigid": rigid,
            "anchors": anchors, "width": width, "weights": weights, "timing": timing,
            "pairing": pairing}


def _coefficients_at(fit: dict, phases: np.ndarray) -> np.ndarray:
    phases = np.mod(np.asarray(phases, dtype=np.float64), 1.0)
    return np.stack([np.interp(phases, fit["phases"], fit["coefficients"][:, k], period=1.0)
                     for k in range(fit["coefficients"].shape[1])], axis=-1)


def _to_tee_frame(fit: dict, points: np.ndarray, iterations: int = 30) -> np.ndarray:
    """``phi^-1``: undo the warp by fixed-point iteration, then the rigid map."""

    flat = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    undone = flat.copy()
    for _ in range(iterations):
        undone = flat - _gaussian(undone, fit["anchors"], fit["width"]) @ fit["weights"]
    return fit["rigid"].inverse().apply(undone).reshape(np.shape(points))


def score_model_based_fusion(fit: dict, model, observations, truth, config: SolverConfig) -> dict:
    """Every endpoint of one B2 fit."""

    def geometry(phases, vertices):
        coefficients = _coefficients_at(fit, phases)
        return model.physical_coordinates(coefficients)[:, model.observation_rows(np.asarray(vertices)), :]

    out = {}
    if len(observations.heldout_cmr_times):
        out["heldout_cmr_mm"], out["heldout_cmr_p95_mm"] = _mean_and_p95(
            geometry(observations.heldout_cmr_times, observations.heldout_cmr_vertices),
            observations.heldout_cmr_points)
    if len(observations.heldout_tee_times):
        phases = fit["timing"].evaluate(np.asarray(observations.heldout_tee_times))
        out["heldout_tee_mm"], out["heldout_tee_p95_mm"] = _mean_and_p95(
            _to_tee_frame(fit, geometry(phases, observations.heldout_tee_vertices)),
            observations.heldout_tee_points)
    latent = _coefficients_at(fit, GAP_PHASES)
    out["attachment_residual_mm"] = float(max(model.attachment_residual(row) for row in latent))
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


def run_model_based_fusion(
    problem: JointProblem,
    config: SolverConfig,
    *,
    alignment: StartingAlignment | None,
    grid: list[tuple[float, float]],
    centers: int = 12,
    iterations: int = 25,
) -> ArmResult:
    """B2. ``grid`` lists (shape weight, warp ridge); the first is the arm's."""

    if alignment is None:
        return _failure("model_based_fusion", "needs_the_input_derived_start")
    by_config = {}
    for shape_weight, warp_ridge in grid:
        fit = fit_model_based_fusion(problem.model, problem.observations, alignment,
                                     shape_weight=float(shape_weight), warp_ridge=float(warp_ridge),
                                     centers=centers, iterations=iterations,
                                     scale_floor=config.identity_prior_scale_floor)
        by_config[f"{float(shape_weight):g},{float(warp_ridge):g}"] = score_model_based_fusion(
            fit, problem.model, problem.observations, problem.truth, config)
    primary = by_config[f"{float(grid[0][0]):g},{float(grid[0][1]):g}"]
    finite = all(np.isfinite(primary.get(k, np.nan)) for k in ("heldout_cmr_mm", "heldout_tee_mm"))
    return ArmResult(
        baseline_id="model_based_fusion",
        succeeded=bool(finite),
        failure_reason=None if finite else "non_finite_score",
        attachment_residual_mm=primary["attachment_residual_mm"],
        heldout_cmr_mm=primary.get("heldout_cmr_mm"),
        heldout_tee_mm=primary.get("heldout_tee_mm"),
        rotation_deg=primary["rotation_deg"],
        translation_mm=primary["translation_mm"],
        phase_error_cycles=primary["phase_error_cycles"],
        cost=0.0,
        iterations=int(iterations),
        heldout_cmr_p95_mm=primary.get("heldout_cmr_p95_mm"),
        heldout_tee_p95_mm=primary.get("heldout_tee_p95_mm"),
        phase_map_cycles=primary["phase_map_cycles"],
        extras={"by_config": by_config, "grid": [list(map(float, g)) for g in grid],
                "centers": int(centers)},
    )
