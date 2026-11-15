r"""Held-out generator and renderer families for the anti-inverse-crime protocol.

`configs/simulator_protocol.yaml` names three families and, until now, only the
train family existed. Everything measured so far therefore fitted data generated
by the same model class the solver assumes, which is an inverse crime: it
measures optimization, not method robustness.

These are the two held-out generators:

``spline_loading_motion_v1`` (validation)
    Periodic cubic B-spline motion instead of a truncated Fourier series, plus
    spatially correlated rather than independent noise. A Fourier basis of the
    order the solver uses cannot represent a B-spline exactly, so any fit carries
    a genuine model-misspecification error.

``heldout_pathology_mechanics_v1`` (locked test)
    Adds the interventions the specification lists: prolapse-like free-edge
    displacement, leaflet tethering, regional wall-motion abnormality, and
    arrhythmia-like phase irregularity, together with heavier-tailed noise and
    occlusion.

Nothing here is used to fit anything. The families exist so degradation under
misspecification can be measured rather than assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRAIN_FAMILY = "harmonic_anatomy_motion_v1"
VAL_FAMILY = "spline_loading_motion_v1"
TEST_FAMILY = "heldout_pathology_mechanics_v1"
FAMILIES = (TRAIN_FAMILY, VAL_FAMILY, TEST_FAMILY)


def periodic_bspline_basis(phases: np.ndarray, n_control: int) -> np.ndarray:
    """Uniform periodic cubic B-spline basis evaluated on the circle.

    Deliberately not a Fourier basis. A degree-three spline with compact support
    has different smoothness and different local behaviour, which is exactly the
    misspecification the validation family is meant to introduce.
    """

    if n_control < 4:
        raise ValueError("a periodic cubic B-spline needs at least four control points")
    tau = np.mod(np.asarray(phases, dtype=np.float64), 1.0)
    scaled = tau * n_control
    index = np.floor(scaled).astype(np.int64)
    local = scaled - index

    # Uniform cubic B-spline weights for the four supporting control points.
    weights = np.stack(
        [
            (1.0 - local) ** 3 / 6.0,
            (3.0 * local**3 - 6.0 * local**2 + 4.0) / 6.0,
            (-3.0 * local**3 + 3.0 * local**2 + 3.0 * local + 1.0) / 6.0,
            local**3 / 6.0,
        ],
        axis=-1,
    )
    basis = np.zeros((*tau.shape, n_control), dtype=np.float64)
    for offset in range(4):
        columns = np.mod(index + offset - 1, n_control)
        np.put_along_axis(
            basis, columns[..., None], weights[..., offset][..., None], axis=-1
        )
    return basis


def correlated_vertex_noise(
    vertices: np.ndarray,
    shape: tuple[int, ...],
    scale_mm: float,
    correlation_mm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Spatially correlated noise, in place of the independent draws used in training.

    Independent per-vertex noise is the easiest possible case for a smooth shape
    basis, which averages it away. Correlated noise cannot be averaged out and is
    the more honest stress.
    """

    points = np.asarray(vertices, dtype=np.float64)
    if correlation_mm <= 0.0:
        raise ValueError("correlation_mm must be positive")
    squared = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)
    kernel = np.exp(-0.5 * squared / correlation_mm**2)
    kernel = kernel + 1e-8 * np.eye(len(points))
    factor = np.linalg.cholesky(kernel)
    white = rng.normal(size=(*shape[:-2], len(points), 3))
    return scale_mm * np.einsum("ij,...jk->...ik", factor, white)


@dataclass(frozen=True)
class FamilyPerturbation:
    """What a held-out family did to one case, recorded for the audit trail."""

    family_id: str
    motion_basis: str
    noise_model: str
    interventions: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "motion_basis": self.motion_basis,
            "noise_model": self.noise_model,
            "interventions": list(self.interventions),
        }


def spline_motion_coefficients(
    base: np.ndarray,
    n_control: int,
    rng: np.random.Generator,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Draw one patient's B-spline motion coefficients.

    Separated from evaluation on purpose. Drawing inside the evaluator makes it
    far too easy to call it once per modality and hand each one an independent
    motion field, which would mean the two modalities observe different anatomy
    rather than the same anatomy through different operators.
    """

    return rng.normal(0.0, amplitude, size=(n_control, *np.shape(base)))


def evaluate_spline_motion(
    phases: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    """Cycle-centered B-spline motion, so periodicity still holds exactly."""

    n_control = coefficients.shape[0]
    basis = periodic_bspline_basis(phases, n_control)
    field = np.einsum("...k,kvc->...vc", basis, coefficients)
    # Remove the cycle mean so the family differs in basis, not in gauge.
    dense = periodic_bspline_basis(np.linspace(0.0, 1.0, 512, endpoint=False), n_control)
    mean_field = np.einsum("...k,kvc->...vc", dense, coefficients).mean(axis=0)
    return field - mean_field


def spline_motion_field(
    phases: np.ndarray,
    base: np.ndarray,
    n_control: int,
    rng: np.random.Generator,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Draw and evaluate in one step. Use only for a single-modality case."""

    return evaluate_spline_motion(
        phases, spline_motion_coefficients(base, n_control, rng, amplitude)
    )


def apply_pathology(
    geometry: np.ndarray,
    phases: np.ndarray,
    groups: dict[str, np.ndarray],
    rng: np.random.Generator,
    *,
    prolapse_mm: float = 3.5,
    tethering_mm: float = 2.0,
    wall_motion_attenuation: float = 0.55,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Apply the section 5.4 pathology interventions to a clean trajectory."""

    result = np.array(geometry, dtype=np.float64, copy=True)
    interventions: list[str] = []

    leaflet = groups.get("leaflet_free")
    if leaflet is not None and len(leaflet):
        # Prolapse: free edge overshoots toward the atrium near end systole.
        systole = np.clip(np.cos(2.0 * np.pi * phases), 0.0, None)[..., None]
        half = len(leaflet) // 2
        prolapsing = leaflet[:half] if half else leaflet
        result[..., prolapsing, 2] += prolapse_mm * systole
        interventions.append("p2_prolapse_free_edge_displacement")

        # Tethering: the opposing leaflet is pulled apically instead.
        tethered = leaflet[half:] if half else leaflet
        if len(tethered):
            result[..., tethered, 2] -= tethering_mm * systole
            interventions.append("leaflet_tethering")

    chamber = groups.get("cmr_visible")
    if chamber is not None and len(chamber):
        # Regional wall-motion abnormality: one angular sector contracts less.
        centre = result[..., chamber, :].mean(axis=tuple(range(result.ndim - 2)))
        angles = np.arctan2(centre[:, 1], centre[:, 0])
        region = chamber[(angles > -0.6) & (angles < 0.9)]
        if len(region):
            reference = result[..., :1, region, :] if result.ndim == 3 else result[region]
            if result.ndim == 3:
                result[:, region, :] = reference + wall_motion_attenuation * (
                    result[:, region, :] - reference
                )
                interventions.append("regional_wall_motion_abnormality")

    return result, tuple(interventions)


def arrhythmic_phase_samples(
    n_frames: int, rng: np.random.Generator, irregularity: float = 0.35
) -> np.ndarray:
    """Irregular beat-to-beat sampling, still spanning one cycle in order.

    The solver assumes evenly spaced acquisition times. This keeps the ordering
    and the cycle but breaks the spacing, which is what an irregular rhythm does.
    """

    if not 0.0 <= irregularity < 1.0:
        raise ValueError("irregularity must lie in [0, 1)")
    gaps = 1.0 + irregularity * rng.normal(size=n_frames)
    gaps = np.clip(gaps, 0.15, None)
    cumulative = np.cumsum(gaps)
    return np.concatenate(([0.0], cumulative[:-1] / cumulative[-1]))


# --------------------------------------------------------- renderer families --
TRAIN_RENDERER = "analytic_distance_likelihood_v1"
VAL_RENDERER = "supersampled_contour_sector_v2"
TEST_RENDERER = "voxel_raster_ray_shadow_v3"
RENDERER_FAMILIES = (TRAIN_RENDERER, VAL_RENDERER, TEST_RENDERER)


def render_supersampled_cmr(
    vertices: np.ndarray,
    faces: np.ndarray,
    stack,
    *,
    sigma_mm: float,
    factor: int = 3,
    through_plane_samples: int = 5,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Validation CMR renderer: render finer, then average down.

    The training renderer evaluates one sample per pixel centre. This evaluates
    ``factor**2`` sub-pixel samples and box-averages them, and uses more
    through-plane samples. The result is a different discretization error, not a
    different anatomy, which is precisely the misspecification a held-out
    renderer family should introduce: a solver tuned to one quadrature should not
    silently depend on it.
    """

    from dataclasses import replace as _replace

    from rooted4d.observation import render_cmr_surface_likelihood

    if factor < 2:
        raise ValueError("supersampling factor must be at least two")
    fine = _replace(
        stack,
        image_shape=(stack.image_shape[0] * factor, stack.image_shape[1] * factor),
        pixel_spacing_mm=(
            stack.pixel_spacing_mm[0] / factor,
            stack.pixel_spacing_mm[1] / factor,
        ),
    )
    rendered = render_cmr_surface_likelihood(
        vertices,
        faces,
        fine,
        sigma_mm=sigma_mm,
        through_plane_samples=through_plane_samples,
        noise_std=noise_std,
        seed=seed,
    )
    frames, slices, rows, columns = rendered.shape
    blocked = rendered.reshape(
        frames, slices, rows // factor, factor, columns // factor, factor
    )
    return blocked.mean(axis=(3, 5)).astype(np.float32)


def render_ray_raster_tee(
    vertices: np.ndarray,
    faces: np.ndarray,
    grid,
    *,
    sigma_mm: float,
    occlusion: float = 0.7,
    attenuation_per_mm: float = 0.003,
    shadows: tuple = (),
    noise_std: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Test TEE renderer: accumulate along rays with front-to-back occlusion.

    The training renderer treats every voxel independently, so a surface behind
    another surface is rendered just as brightly. This one walks each ray outward
    from the apex and attenuates by what it has already passed through, which is
    closer to how an ultrasound sector actually behaves and is a genuinely
    different forward model rather than a reparameterization of the same one.
    """

    from rooted4d.observation import surface_likelihood
    from rooted4d.observation.tee import _shadow_mask

    if not 0.0 <= occlusion < 1.0:
        raise ValueError("occlusion must lie in [0, 1)")
    values = np.asarray(vertices, dtype=np.float64)
    if values.ndim == 2:
        values = values[None, ...]
    world = grid.world_grid()
    # Share the training renderer's depth attenuation so the two families differ
    # specifically by occlusion rather than by overall gain. A brightness
    # difference would be a trivial misspecification; a visibility difference is
    # the interesting one.
    depth_attenuation = np.exp(-attenuation_per_mm * grid.depths_mm)[:, None, None]
    # Shadow cones too. Without them this family would simply be handed cleaner
    # data than the training family, and any apparent robustness would be an
    # artefact of removing an occluder rather than of changing the model.
    visibility = depth_attenuation * _shadow_mask(grid, tuple(shadows))
    rendered = []
    for frame in values:
        response = surface_likelihood(world, frame, faces, sigma_mm=sigma_mm)
        response = response * visibility
        # Front-to-back accumulation along the depth axis.
        transmitted = np.ones_like(response[0])
        output = np.empty_like(response)
        for depth in range(response.shape[0]):
            output[depth] = response[depth] * transmitted
            transmitted = transmitted * (1.0 - occlusion * response[depth])
        rendered.append(output)
    result = np.asarray(rendered, dtype=np.float64)
    if noise_std:
        rng = np.random.default_rng(seed)
        result = result + rng.normal(0.0, noise_std, result.shape)
    return np.clip(result, 0.0, 1.0).astype(np.float32)
