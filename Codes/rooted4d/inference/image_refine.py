r"""Image-domain refinement against the CMR slice and TEE sector likelihoods.

Until now the joint solver consumed point evidence. This wires in the actual
observation operators of section 4.5: a finite-thickness CMR plane renderer and a
diverging TEE sector renderer, both fitted as segmentation-style likelihoods
rather than raw intensities, which is the order section 4.5 asks for.

**Why this is a refinement and not the whole solve.** Each likelihood render
costs on the order of a tenth of a second, and a central-difference Jacobian over
the full parameter vector needs eighty-six of them per iteration. A cold joint
image-domain solve in pure NumPy is therefore hours, not minutes. The staged
design of work package 5 already prescribes the alternative: initialize cheaply,
then refine. This module refines pose, circular phase, and the leading shape
modes -- the variables the image evidence actually constrains -- and reports its
runtime honestly. An analytic or adjoint gradient is what would make a full
image-domain joint solve practical, and that remains open.

Held-out evaluation follows the preregistered protocol: whole CMR *planes* are
withheld, not merely pixels, so the first co-primary endpoint is measured on
evidence the fit never saw.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from rooted4d.observation import (
    CMRPlaneStack,
    TEESectorGrid,
    render_cmr_surface_likelihood,
    render_tee_surface_likelihood,
)

from .joint import SolverConfig, _levenberg_marquardt, latent_trajectory
from .scene import SceneModel, SceneParameters


@dataclass(frozen=True)
class ImageObservations:
    """Rendered likelihood volumes with an explicit held-out plane/phase split."""

    stack: CMRPlaneStack
    cmr_faces: np.ndarray
    cmr_times: np.ndarray
    cmr_volume: np.ndarray
    fit_slices: np.ndarray
    heldout_slices: np.ndarray
    grid: TEESectorGrid
    tee_faces: np.ndarray
    tee_times: np.ndarray
    tee_volume: np.ndarray
    heldout_tee_times: np.ndarray
    heldout_tee_volume: np.ndarray
    #: Noiseless rendering of the same truth, used for evaluation only and never
    #: shown to the fit. Held-out error against the noisy volume saturates at the
    #: observation noise level and cannot discriminate between geometries; this
    #: separates geometric error from the noise floor.
    clean_cmr_volume: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.cmr_volume.ndim != 4:
            raise ValueError("cmr_volume must have shape (frames, slices, rows, columns)")
        if self.tee_volume.ndim != 4:
            raise ValueError("tee_volume must have shape (frames, depth, elevation, azimuth)")
        if len(np.intersect1d(self.fit_slices, self.heldout_slices)):
            raise ValueError("a CMR plane cannot be both fitted and held out")
        if not len(self.fit_slices):
            raise ValueError("at least one CMR plane must be fitted")


@dataclass(frozen=True)
class ImageRefinementConfig:
    """Which variables the image evidence is allowed to move."""

    n_shape_modes: int = 4
    cmr_weight: float = 1.0
    tee_weight: float = 1.0
    max_iterations: int = 12
    finite_difference_step: float = 1e-4
    cmr_sigma_mm: float = 1.5
    tee_sigma_mm: float = 1.5
    through_plane_samples: int = 2

    def __post_init__(self) -> None:
        if self.n_shape_modes < 0:
            raise ValueError("n_shape_modes cannot be negative")
        if self.cmr_weight < 0.0 or self.tee_weight < 0.0:
            raise ValueError("modality weights cannot be negative")
        if self.cmr_weight == 0.0 and self.tee_weight == 0.0:
            raise ValueError("at least one modality must carry weight")


def _pack_refinement(
    parameters: SceneParameters, refinement: ImageRefinementConfig
) -> np.ndarray:
    return np.concatenate(
        [
            parameters.pose_axis_angle,
            parameters.pose_translation,
            np.array([parameters.phase_offset]),
            parameters.identity[: refinement.n_shape_modes],
        ]
    )


def _unpack_refinement(
    vector: np.ndarray, base: SceneParameters, refinement: ImageRefinementConfig
) -> SceneParameters:
    identity = base.identity.copy()
    if refinement.n_shape_modes:
        identity[: refinement.n_shape_modes] = vector[7 : 7 + refinement.n_shape_modes]
    return replace(
        base,
        pose_axis_angle=vector[0:3],
        pose_translation=vector[3:6],
        phase_offset=float(vector[6]),
        identity=identity,
    )


def render_predicted(
    parameters: SceneParameters,
    observations: ImageObservations,
    model: SceneModel,
    config: SolverConfig,
    refinement: ImageRefinementConfig,
    *,
    cmr_times: np.ndarray | None = None,
    tee_times: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Render the model through both observation operators."""

    cmr_prediction = None
    tee_prediction = None

    times = observations.cmr_times if cmr_times is None else cmr_times
    if len(times) and refinement.cmr_weight > 0.0:
        latent = latent_trajectory(parameters, times, "cmr", model, config)
        geometry = model.physical_coordinates(latent)
        cmr_prediction = render_cmr_surface_likelihood(
            geometry,
            observations.cmr_faces,
            observations.stack,
            sigma_mm=refinement.cmr_sigma_mm,
            through_plane_samples=refinement.through_plane_samples,
        )

    times = observations.tee_times if tee_times is None else tee_times
    if len(times) and refinement.tee_weight > 0.0:
        warp = parameters.warp()
        latent = latent_trajectory(parameters, warp.evaluate(times), "tee", model, config)
        geometry = model.physical_coordinates(latent)
        posed = parameters.transform().apply(geometry)
        tee_prediction = render_tee_surface_likelihood(
            posed,
            observations.tee_faces,
            observations.grid,
            sigma_mm=refinement.tee_sigma_mm,
        )
    return cmr_prediction, tee_prediction


def _image_residuals(
    vector: np.ndarray,
    base: SceneParameters,
    observations: ImageObservations,
    model: SceneModel,
    config: SolverConfig,
    refinement: ImageRefinementConfig,
) -> np.ndarray:
    parameters = _unpack_refinement(vector, base, refinement)
    cmr_prediction, tee_prediction = render_predicted(
        parameters, observations, model, config, refinement
    )
    pieces: list[np.ndarray] = []
    if cmr_prediction is not None:
        # Only the fitted planes contribute; held-out planes stay unseen.
        difference = (
            cmr_prediction[:, observations.fit_slices]
            - observations.cmr_volume[:, observations.fit_slices]
        )
        pieces.append(np.sqrt(refinement.cmr_weight) * difference.reshape(-1))
    if tee_prediction is not None:
        difference = tee_prediction - observations.tee_volume
        pieces.append(np.sqrt(refinement.tee_weight) * difference.reshape(-1))
    if not pieces:
        raise RuntimeError("no image residual was produced")
    return np.concatenate(pieces)


@dataclass(frozen=True)
class ImageRefinementResult:
    parameters: SceneParameters
    start_parameters: SceneParameters
    cost_before: float
    cost_after: float
    iterations: int
    n_residuals: int
    n_parameters: int
    seconds: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "cost_before": self.cost_before,
            "cost_after": self.cost_after,
            "cost_reduction": self.cost_before - self.cost_after,
            "iterations": self.iterations,
            "n_residuals": self.n_residuals,
            "n_parameters": self.n_parameters,
            "seconds": self.seconds,
        }


def refine_against_images(
    start: SceneParameters,
    observations: ImageObservations,
    model: SceneModel,
    config: SolverConfig,
    refinement: ImageRefinementConfig = ImageRefinementConfig(),
) -> ImageRefinementResult:
    """Refine pose, circular phase, and leading shape modes on image evidence."""

    import time

    vector = _pack_refinement(start, refinement)

    def residual_function(value: np.ndarray) -> np.ndarray:
        return _image_residuals(value, start, observations, model, config, refinement)

    before = residual_function(vector)
    solver_config = replace(
        config,
        max_iterations=refinement.max_iterations,
        finite_difference_step=refinement.finite_difference_step,
    )
    started = time.time()
    solution, cost, iterations = _levenberg_marquardt(
        residual_function, vector, solver_config
    )
    elapsed = time.time() - started
    return ImageRefinementResult(
        parameters=_unpack_refinement(solution, start, refinement),
        start_parameters=start,
        cost_before=float(before @ before),
        cost_after=float(cost),
        iterations=iterations,
        n_residuals=int(before.size),
        n_parameters=int(vector.size),
        seconds=float(elapsed),
    )


def heldout_plane_error(
    parameters: SceneParameters,
    observations: ImageObservations,
    model: SceneModel,
    config: SolverConfig,
    refinement: ImageRefinementConfig,
) -> dict[str, float]:
    """First co-primary endpoint, in the image domain: hidden CMR planes."""

    if not len(observations.heldout_slices):
        return {}
    cmr_prediction, _ = render_predicted(
        parameters,
        observations,
        model,
        config,
        replace(refinement, tee_weight=0.0),
    )
    difference = (
        cmr_prediction[:, observations.heldout_slices]
        - observations.cmr_volume[:, observations.heldout_slices]
    )
    result = {
        "heldout_plane_rmse": float(np.sqrt(np.mean(difference**2))),
        "heldout_plane_mae": float(np.mean(np.abs(difference))),
        "n_heldout_planes": int(len(observations.heldout_slices)),
    }
    if observations.clean_cmr_volume is not None:
        clean = (
            cmr_prediction[:, observations.heldout_slices]
            - observations.clean_cmr_volume[:, observations.heldout_slices]
        )
        result["heldout_plane_rmse_vs_clean"] = float(np.sqrt(np.mean(clean**2)))
        result["heldout_plane_mae_vs_clean"] = float(np.mean(np.abs(clean)))
    return result


def make_image_problem(
    truth: SceneParameters,
    model: SceneModel,
    config: SolverConfig,
    refinement: ImageRefinementConfig = ImageRefinementConfig(),
    *,
    seed: int = 0,
    n_cmr_frames: int = 6,
    n_tee_frames: int = 6,
    n_slices: int = 4,
    image_shape: tuple[int, int] = (12, 12),
    tee_grid_shape: tuple[int, int, int] = (8, 6, 6),
    image_noise_std: float = 0.02,
    speckle_std: float = 0.10,
    renderer_family: str = "analytic_distance_likelihood_v1",
) -> ImageObservations:
    """Render observed likelihood volumes from known truth.

    Resolutions are small by necessity: a central-difference Jacobian needs two
    renders per parameter per iteration. They are large enough for the fit to be
    driven by image evidence and are reported alongside the runtime rather than
    presented as clinical resolution.
    """

    from rooted4d.observation import ShadowCone, default_cmr_stack, default_tee_grid
    from rooted4d.synthetic.families import (
        TEST_RENDERER,
        TRAIN_RENDERER,
        VAL_RENDERER,
        render_ray_raster_tee,
        render_supersampled_cmr,
    )

    if renderer_family not in (TRAIN_RENDERER, VAL_RENDERER, TEST_RENDERER):
        raise ValueError(f"unknown renderer family: {renderer_family}")
    if n_slices < 2:
        raise ValueError("at least two CMR planes are needed to hold one out")
    rng = np.random.default_rng(seed)
    complex_faces = _toy_faces(model)

    cmr_times = np.linspace(0.0, 1.0, n_cmr_frames, endpoint=False)
    all_tee_times = np.linspace(0.0, 1.0, n_tee_frames, endpoint=False)
    hidden_tee = np.arange(n_tee_frames)[1::3]
    visible_tee = np.setdiff1d(np.arange(n_tee_frames), hidden_tee)
    tee_times = all_tee_times[visible_tee]

    latent = latent_trajectory(truth, cmr_times, "cmr", model, config)
    cmr_geometry = model.physical_coordinates(latent)
    stack = default_cmr_stack(
        model.template_physical[model.groups["cmr_visible"]],
        n_slices=n_slices,
        image_shape=image_shape,
    )
    clean_cmr_volume = render_cmr_surface_likelihood(
        cmr_geometry,
        complex_faces["cmr_surface"],
        stack,
        sigma_mm=refinement.cmr_sigma_mm,
        through_plane_samples=refinement.through_plane_samples,
    )
    if renderer_family == VAL_RENDERER:
        # A different quadrature for the same anatomy: the fit must not depend on
        # the training renderer's sampling scheme.
        cmr_volume = render_supersampled_cmr(
            cmr_geometry,
            complex_faces["cmr_surface"],
            stack,
            sigma_mm=refinement.cmr_sigma_mm,
            noise_std=image_noise_std,
            seed=seed + 11,
        )
    else:
        cmr_volume = render_cmr_surface_likelihood(
            cmr_geometry,
            complex_faces["cmr_surface"],
            stack,
            sigma_mm=refinement.cmr_sigma_mm,
            through_plane_samples=refinement.through_plane_samples,
            noise_std=image_noise_std,
            seed=seed + 11,
        )

    warp = truth.warp()
    latent = latent_trajectory(truth, warp.evaluate(all_tee_times), "tee", model, config)
    posed = truth.transform().apply(model.physical_coordinates(latent))
    grid = default_tee_grid(posed[0][model.groups["tee_visible"]], grid_shape=tee_grid_shape)
    shadow = ShadowCone(
        azimuth_rad=float(rng.uniform(-0.15, 0.15)),
        elevation_rad=float(rng.uniform(-0.15, 0.15)),
        angular_radius_rad=float(rng.uniform(0.08, 0.16)),
        start_depth_mm=float(rng.uniform(grid.depths_mm[0], grid.depths_mm[-1])),
        strength=float(rng.uniform(0.3, 0.7)),
    )
    if renderer_family == TEST_RENDERER:
        # Front-to-back occlusion instead of independent voxels: a genuinely
        # different forward model, not a reparameterization of the same one.
        tee_volume = render_ray_raster_tee(
            posed,
            complex_faces["tee_surface"],
            grid,
            sigma_mm=refinement.tee_sigma_mm,
            shadows=(shadow,),
            noise_std=image_noise_std,
            seed=seed + 23,
        )
    else:
        tee_volume = render_tee_surface_likelihood(
            posed,
            complex_faces["tee_surface"],
            grid,
            sigma_mm=refinement.tee_sigma_mm,
            shadows=(shadow,),
            speckle_std=speckle_std,
            noise_std=image_noise_std,
            seed=seed + 23,
        )
    visible_tee_volume = tee_volume[visible_tee]

    # Withhold alternating planes so the hidden set spans the stack rather than
    # sitting at one end, where extrapolation would be trivially bad.
    all_slices = np.arange(n_slices)
    heldout = all_slices[1::2]
    fitted = np.setdiff1d(all_slices, heldout)
    return ImageObservations(
        stack=stack,
        cmr_faces=complex_faces["cmr_surface"],
        cmr_times=cmr_times,
        cmr_volume=cmr_volume,
        fit_slices=fitted,
        heldout_slices=heldout,
        grid=grid,
        tee_faces=complex_faces["tee_surface"],
        tee_times=tee_times,
        tee_volume=visible_tee_volume,
        heldout_tee_times=all_tee_times[hidden_tee],
        heldout_tee_volume=tee_volume[hidden_tee],
        clean_cmr_volume=clean_cmr_volume,
    )


def _toy_faces(model: SceneModel) -> dict[str, np.ndarray]:
    """Recover the renderable face sets for the procedural template."""

    from rooted4d.geometry import make_toy_complex

    n_annulus = len(model.groups["annulus"])
    complex_ = make_toy_complex(n_annulus=n_annulus)
    if len(complex_.global_vertices) != model.n_physical:
        raise ValueError(
            "image rendering currently supports the procedural template only; a "
            "frozen production template must supply its own face sets"
        )
    return complex_.faces


def heldout_tee_error(
    parameters: SceneParameters,
    observations: ImageObservations,
    model: SceneModel,
    config: SolverConfig,
    refinement: ImageRefinementConfig,
) -> dict[str, float]:
    """Error on hidden TEE phases, where pose and circular phase are observable.

    Held-out CMR planes cannot score pose or phase: neither enters the CMR
    branch of the forward model, so a pose-only perturbation leaves the predicted
    planes bitwise unchanged. This is the same structural blindness the missing
    view factorial reports for the CMR-only arm.
    """

    if not len(observations.heldout_tee_times):
        return {}
    _, prediction = render_predicted(
        parameters,
        observations,
        model,
        config,
        replace(refinement, cmr_weight=0.0),
        tee_times=observations.heldout_tee_times,
    )
    difference = prediction - observations.heldout_tee_volume
    return {
        "heldout_tee_rmse": float(np.sqrt(np.mean(difference**2))),
        "heldout_tee_mae": float(np.mean(np.abs(difference))),
        "n_heldout_tee_phases": int(len(observations.heldout_tee_times)),
    }
