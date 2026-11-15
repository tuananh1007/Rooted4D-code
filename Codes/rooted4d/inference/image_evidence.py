r"""Image-domain likelihood evidence for the joint solver.

The joint solver has consumed point evidence only, which quietly assumes the
hardest part of the problem away: a point observation names *which* template
vertex produced it. Real segmentation output does not. This module lets the
solver fit the rendered CMR slice and TEE sector likelihoods directly, so no
vertex correspondence is assumed anywhere in the objective.

**Why the evidence is a region of interest rather than a whole volume.** Fitting
the full rendered volumes is not affordable with numerical Jacobians. A single
render costs on the order of thirty million point-triangle distance evaluations
-- twenty CMR frames times three through-plane samples times 2880 voxels times
144 faces, plus the TEE sector -- and a central-difference Jacobian over sixty
parameters needs a hundred and twenty-six of them per iteration. That is hours
per sample, which is why the existing refinement path restricts itself to four
shape modes.

``surface_likelihood`` accepts arbitrary query points, so the evidence here is a
fixed set of world points chosen once from the voxels that actually carry
signal, with their observed likelihood values. Voxels far from any surface
contribute a value of essentially zero and a derivative of essentially zero, so
excluding them removes cost rather than information. The set is chosen from the
*observed* volume and then frozen, so it never moves with the iterate and the
residual keeps a fixed length.

**Weighting is not a free parameter here, and getting it wrong is silent.** Point
residuals are millimetres and the solver priors were tuned against them. Image
residuals are dimensionless likelihood differences bounded in [0, 1], so at unit
weight the image evidence is roughly two hundred times too weak to move the fit:
measured on one sample, going from the cold start to the truth improved the image
evidence by 179 while costing 36873 in prior, so the truth was not a minimum of
the objective at all and the optimizer correctly stayed near the mean shape. The
first run of the image arm did exactly that and produced 20 degrees of rotation
error, which would have read as a property of image evidence and was a property
of the weights.

:func:`noise_weights` returns the statistically correct choice, the inverse
observation-noise variance, which turns each residual into a standardized one
commensurate with a prior that is a Gaussian log-density. With it the truth costs
591687 against 1066476 at the cold start, so the minimum is where it should be.
The noise levels are known here because the generator set them; a real deployment
has to estimate them, and quoting a weight without saying where it came from is
how this class of error survives.

What this does not do: it is still a Gauss-Newton fit with numerical Jacobians,
so it is a reference, not the production path. Gradient-based rendering is the
PyTorch optimizer gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rooted4d.observation.cmr import CMRPlaneStack
from rooted4d.observation.distance import surface_likelihood
from rooted4d.observation.tee import TEESectorGrid

from .joint import SolverConfig, _apply_gauges, latent_trajectory
from .scene import SceneModel, SceneParameters

__all__ = [
    "ImageEvidence",
    "noise_weights",
    "evidence_from_sample",
    "image_residuals",
    "reconstruct_cmr_stack",
    "reconstruct_tee_grid",
]


def noise_weights(
    cmr_noise_std: float, tee_noise_std: float
) -> tuple[float, float]:
    """Inverse observation-noise variance for each modality.

    The likelihood residual divided by its own noise standard deviation is a
    standardized residual, and a sum of squared standardized residuals is what
    the solver priors are already expressed against. Weighting by anything else
    silently rescales the evidence relative to the prior.
    """

    if cmr_noise_std <= 0.0 or tee_noise_std <= 0.0:
        raise ValueError("noise standard deviations must be positive")
    return 1.0 / cmr_noise_std**2, 1.0 / tee_noise_std**2


@dataclass(frozen=True)
class ImageEvidence:
    """Observed likelihood at a frozen set of world query points."""

    cmr_times: np.ndarray
    cmr_points: np.ndarray
    cmr_values: np.ndarray
    cmr_faces: np.ndarray
    #: World offsets along the slice normal. The generator renders each CMR
    #: voxel as the mean over three samples spanning the slice thickness, so a
    #: prediction taken on the centre plane alone is a different observation
    #: operator from the one that produced the data. Over a 6 mm slice that is
    #: not a rounding difference, and the fit would be biased by it.
    cmr_through_plane: np.ndarray
    tee_times: np.ndarray
    tee_points: np.ndarray
    tee_values: np.ndarray
    tee_faces: np.ndarray
    sigma_mm: float
    cmr_normal: np.ndarray

    def __post_init__(self) -> None:
        if self.cmr_values.shape != (len(self.cmr_times), len(self.cmr_points)):
            raise ValueError("cmr_values must have shape (frames, points)")
        if self.tee_values.shape != (len(self.tee_times), len(self.tee_points)):
            raise ValueError("tee_values must have shape (frames, points)")
        if self.sigma_mm <= 0.0:
            raise ValueError("sigma_mm must be positive")

    @property
    def n_residuals(self) -> int:
        return int(self.cmr_values.size + self.tee_values.size)


def reconstruct_cmr_stack(
    affine: np.ndarray,
    slice_offsets_mm: np.ndarray,
    image_shape: tuple[int, int],
    slice_thickness_mm: float,
) -> CMRPlaneStack:
    """Rebuild the plane stack a sample was rendered through.

    The affine stores row and column directions scaled by pixel spacing, the
    slice normal, and the origin of the first slice. The centre is recovered by
    undoing the origin offset rather than recomputed from geometry, which would
    only coincide with the original by luck.
    """

    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("affine must be 4x4")
    offsets = np.asarray(slice_offsets_mm, dtype=np.float64)
    rows, columns = int(image_shape[0]), int(image_shape[1])

    row_scaled = matrix[:3, 0]
    column_scaled = matrix[:3, 1]
    normal = matrix[:3, 2]
    row_spacing = float(np.linalg.norm(row_scaled))
    column_spacing = float(np.linalg.norm(column_scaled))
    if row_spacing <= 0.0 or column_spacing <= 0.0:
        raise ValueError("affine has a degenerate in-plane axis")
    row_direction = row_scaled / row_spacing
    column_direction = column_scaled / column_spacing
    origin = matrix[:3, 3]
    centre = (
        origin
        + 0.5 * (rows - 1) * row_spacing * row_direction
        + 0.5 * (columns - 1) * column_spacing * column_direction
        - offsets[0] * normal
    )
    return CMRPlaneStack(
        center=centre,
        row_direction=row_direction,
        column_direction=column_direction,
        slice_offsets_mm=offsets,
        image_shape=(rows, columns),
        pixel_spacing_mm=(row_spacing, column_spacing),
        slice_thickness_mm=float(slice_thickness_mm),
    )


def reconstruct_tee_grid(
    apex: np.ndarray,
    axis: np.ndarray,
    depths_mm: np.ndarray,
    azimuths_rad: np.ndarray,
    elevations_rad: np.ndarray,
) -> TEESectorGrid:
    """Rebuild the sector grid a sample was rendered through.

    ``lateral_direction`` is not stored because ``default_tee_grid`` fixes it at
    the x axis; that constant is relied on here and will need storing if the
    generator ever chooses it from the anatomy.
    """

    return TEESectorGrid(
        apex=np.asarray(apex, dtype=np.float64),
        axis=np.asarray(axis, dtype=np.float64),
        lateral_direction=np.array([1.0, 0.0, 0.0]),
        depths_mm=np.asarray(depths_mm, dtype=np.float64),
        azimuths_rad=np.asarray(azimuths_rad, dtype=np.float64),
        elevations_rad=np.asarray(elevations_rad, dtype=np.float64),
    )


def _select(volume: np.ndarray, points: np.ndarray, threshold: float, budget: int, seed: int):
    """Freeze a query set from the voxels that carry signal in any frame.

    Selection uses the per-voxel maximum over frames, so a voxel the surface
    only visits at one phase is still kept; selecting per frame would give each
    frame a different query set and a residual whose length changed with the
    iterate.
    """

    flat = volume.reshape(len(volume), -1)
    strength = flat.max(axis=0)
    keep = np.flatnonzero(strength >= threshold)
    if keep.size == 0:
        raise ValueError("no voxel exceeds the signal threshold")
    if keep.size > budget:
        # Deterministic thinning, keeping the strongest half and sampling the
        # rest, so the set spans the band rather than only its brightest ridge.
        order = keep[np.argsort(strength[keep])[::-1]]
        strong = order[: budget // 2]
        remainder = order[budget // 2 :]
        chosen = np.random.default_rng(seed).choice(
            remainder, size=budget - len(strong), replace=False
        )
        keep = np.sort(np.concatenate([strong, chosen]))
    return points[keep], flat[:, keep]


def evidence_from_sample(
    path: str | Path,
    model: SceneModel,
    *,
    sigma_mm: float = 1.5,
    slice_thickness_mm: float = 6.0,
    through_plane_samples: int = 3,
    threshold: float = 0.15,
    budget: int = 400,
    n_cmr_frames: int = 6,
    n_tee_frames: int = 6,
    seed: int = 0,
) -> ImageEvidence:
    """Build frozen image evidence from a stored benchmark sample."""

    from rooted4d.geometry import toy_complex_for_vertices

    with np.load(Path(path), allow_pickle=True) as blob:
        # The faces must come from the complex the sample was built on. This used
        # to be the 24-node complex unconditionally, which on a denser benchmark
        # would have rendered the wrong surfaces instead of failing.
        complex_ = toy_complex_for_vertices(len(blob["template_vertices"]))
        stack = reconstruct_cmr_stack(
            blob["cmr_plane_affine_lps"],
            blob["cmr_slice_offsets_mm"],
            tuple(np.asarray(blob["cmr_surface_likelihood"]).shape[2:]),
            slice_thickness_mm,
        )
        grid = reconstruct_tee_grid(
            blob["tee_grid_apex"], blob["tee_grid_axis"], blob["tee_grid_depths_mm"],
            blob["tee_grid_azimuths_rad"], blob["tee_grid_elevations_rad"],
        )
        cmr_volume = np.asarray(blob["cmr_surface_likelihood"], dtype=np.float64)
        tee_volume = np.asarray(blob["tee_surface_likelihood"], dtype=np.float64)
        cmr_times = np.asarray(blob["cmr_times"], dtype=np.float64)
        tee_times = np.asarray(blob["tee_times"], dtype=np.float64)

    cmr_world = stack.world_grid(0.0).reshape(-1, 3)
    tee_world = grid.world_grid().reshape(-1, 3)

    cmr_frames = np.linspace(0, len(cmr_times) - 1, n_cmr_frames).round().astype(int)
    tee_frames = np.linspace(0, len(tee_times) - 1, n_tee_frames).round().astype(int)
    cmr_points, cmr_values = _select(
        cmr_volume[cmr_frames], cmr_world, threshold, budget, seed
    )
    tee_points, tee_values = _select(
        tee_volume[tee_frames], tee_world, threshold, budget, seed + 1
    )
    return ImageEvidence(
        cmr_times=cmr_times[cmr_frames],
        cmr_points=cmr_points,
        cmr_values=cmr_values,
        cmr_faces=np.asarray(complex_.faces["cmr_surface"], dtype=np.int64),
        cmr_through_plane=np.linspace(-0.5, 0.5, through_plane_samples)
        * slice_thickness_mm,
        tee_times=tee_times[tee_frames],
        tee_points=tee_points,
        tee_values=tee_values,
        tee_faces=np.asarray(complex_.faces["tee_surface"], dtype=np.int64),
        sigma_mm=sigma_mm,
        cmr_normal=stack.normal,
    )


def image_residuals(
    parameters: SceneParameters,
    evidence: ImageEvidence,
    model: SceneModel,
    config: SolverConfig,
) -> np.ndarray:
    """Predicted-minus-observed likelihood at the frozen query points."""

    # The geometry compared with the images is the gauged one, as in
    # ``predict`` and in the session prior. Evaluating the raw session blocks
    # here let their cross-session mean move the image geometry at no prior
    # cost, a direction the fit's final gauge then removed from the reported
    # geometry (E120).
    parameters = _apply_gauges(parameters, model, config)
    pieces: list[np.ndarray] = []
    if config.cmr_image_weight > 0.0 and len(evidence.cmr_times):
        latent = latent_trajectory(parameters, evidence.cmr_times, "cmr", model, config)
        geometry = model.physical_coordinates(latent)
        # Mean over the slice thickness, matching the renderer that produced
        # the observed volume.
        predicted = np.stack(
            [
                np.mean(
                    [
                        surface_likelihood(
                            evidence.cmr_points + shift * evidence.cmr_normal,
                            frame,
                            evidence.cmr_faces,
                            sigma_mm=evidence.sigma_mm,
                        )
                        for shift in evidence.cmr_through_plane
                    ],
                    axis=0,
                )
                for frame in geometry
            ]
        )
        pieces.append(
            np.sqrt(config.cmr_image_weight)
            * (predicted - evidence.cmr_values).reshape(-1)
        )
    if config.tee_image_weight > 0.0 and len(evidence.tee_times):
        warp = parameters.warp()
        latent = latent_trajectory(
            parameters, warp.evaluate(evidence.tee_times), "tee", model, config
        )
        geometry = parameters.transform().apply(model.physical_coordinates(latent))
        predicted = np.stack(
            [
                surface_likelihood(
                    evidence.tee_points, frame, evidence.tee_faces,
                    sigma_mm=evidence.sigma_mm,
                )
                for frame in geometry
            ]
        )
        pieces.append(
            np.sqrt(config.tee_image_weight)
            * (predicted - evidence.tee_values).reshape(-1)
        )
    if not pieces:
        return np.zeros(0)
    return np.concatenate(pieces)
