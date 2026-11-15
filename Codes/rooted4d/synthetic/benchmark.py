"""Generate paired CMR/TEE-like observations with exact nuisance-variable truth."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from rooted4d.geometry import HeterogeneousTemplate, RigidTransform, make_toy_complex
from rooted4d.motion import FourierMotion, MonotoneCircularPhaseWarp
from rooted4d.observation import (
    ShadowCone,
    default_cmr_stack,
    default_tee_grid,
    render_cmr_surface_likelihood,
    render_tee_surface_likelihood,
)


@dataclass(frozen=True)
class SyntheticCase:
    template_vertices: np.ndarray
    motion_coefficients: np.ndarray
    cmr_times: np.ndarray
    cmr_indices: np.ndarray
    cmr_points: np.ndarray
    tee_times: np.ndarray
    tee_true_phases: np.ndarray
    tee_indices: np.ndarray
    tee_points: np.ndarray
    pose_rotation: np.ndarray
    pose_translation: np.ndarray
    phase_logits: np.ndarray
    phase_offset: float
    session_static: np.ndarray
    session_dynamic: np.ndarray
    attachment_residual_mm: float
    periodic_seam_mm: float
    phase_minimum_increment: float
    local_to_global: dict[str, np.ndarray]
    observation_domain: str = "surface_points"
    cmr_surface_likelihood: np.ndarray | None = None
    cmr_plane_affine_lps: np.ndarray | None = None
    cmr_slice_offsets_mm: np.ndarray | None = None
    tee_surface_likelihood: np.ndarray | None = None
    tee_grid_apex: np.ndarray | None = None
    tee_grid_axis: np.ndarray | None = None
    tee_grid_depths_mm: np.ndarray | None = None
    tee_grid_azimuths_rad: np.ndarray | None = None
    tee_grid_elevations_rad: np.ndarray | None = None
    #: A richer procedural instance's interiors (W2.6); (1, 1) is the reduced one.
    chamber_levels: int = 1
    leaflet_rows: int = 1

    def arrays(self) -> dict[str, np.ndarray]:
        arrays = {
            "template_vertices": self.template_vertices,
            "motion_coefficients": self.motion_coefficients,
            "cmr_times": self.cmr_times,
            "cmr_indices": self.cmr_indices,
            "cmr_points": self.cmr_points,
            "tee_times": self.tee_times,
            "tee_true_phases": self.tee_true_phases,
            "tee_indices": self.tee_indices,
            "tee_points": self.tee_points,
            "pose_rotation": self.pose_rotation,
            "pose_translation": self.pose_translation,
            "phase_logits": self.phase_logits,
            "session_static": self.session_static,
            "session_dynamic": self.session_dynamic,
        }
        optional = {
            "cmr_surface_likelihood": self.cmr_surface_likelihood,
            "cmr_plane_affine_lps": self.cmr_plane_affine_lps,
            "cmr_slice_offsets_mm": self.cmr_slice_offsets_mm,
            "tee_surface_likelihood": self.tee_surface_likelihood,
            "tee_grid_apex": self.tee_grid_apex,
            "tee_grid_axis": self.tee_grid_axis,
            "tee_grid_depths_mm": self.tee_grid_depths_mm,
            "tee_grid_azimuths_rad": self.tee_grid_azimuths_rad,
            "tee_grid_elevations_rad": self.tee_grid_elevations_rad,
        }
        arrays.update({key: value for key, value in optional.items() if value is not None})
        if self.chamber_levels != 1 or self.leaflet_rows != 1:
            # Declared in the sample, because a vertex count alone is ambiguous
            # between topologies (``sample_topology``).
            arrays["topology__chamber_levels"] = np.asarray(self.chamber_levels, dtype=np.int64)
            arrays["topology__leaflet_rows"] = np.asarray(self.leaflet_rows, dtype=np.int64)
        return arrays

    def metadata(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "units": "millimetres",
            "phase_units": "cycles",
            "phase_offset": self.phase_offset,
            "attachment_residual_mm": self.attachment_residual_mm,
            "periodic_seam_mm": self.periodic_seam_mm,
            "phase_minimum_increment": self.phase_minimum_increment,
            "observation_domain": self.observation_domain,
            "local_to_global": {
                name: indices.tolist() for name, indices in self.local_to_global.items()
            },
            "scope": (
                "Controlled semi-synthetic observation benchmark; not evidence of "
                "patient-specific clinical CMR-TEE fusion."
            ),
        }


def _default_motion(
    base: np.ndarray,
    leaflet_free: np.ndarray,
    order: int,
    rng: np.random.Generator,
    leaflet_interior: tuple[tuple[np.ndarray, float], ...] = (),
) -> FourierMotion:
    coefficients = np.zeros((2 * order, *base.shape), dtype=np.float64)

    # First harmonic: ventricular/annular contraction and a small twist.
    coefficients[0, :, :2] = -0.075 * base[:, :2]
    coefficients[0, :, 2] = -0.012 * base[:, 2]
    coefficients[1, :, 0] = -0.018 * base[:, 1]
    coefficients[1, :, 1] = 0.018 * base[:, 0]

    # Second harmonic: leaflet closure-like displacement, represented globally.
    if order >= 2:
        coefficients[2, leaflet_free, 2] = -4.0
        coefficients[3, leaflet_free, :2] = -0.035 * base[leaflet_free, :2]
        # On a richer instance (W2.6) the rows between the root and the free edge
        # close in proportion to their depth.
        for row, fraction in leaflet_interior:
            coefficients[2, row, 2] = -4.0 * fraction
            coefficients[3, row, :2] = -0.035 * fraction * base[row, :2]

    # Small higher-frequency subject-specific motion, decaying with order.
    for harmonic in range(3, order + 1):
        cosine = 2 * (harmonic - 1)
        sine = cosine + 1
        scale = 0.35 / harmonic
        coefficients[cosine] = rng.normal(0.0, scale, size=base.shape)
        coefficients[sine] = rng.normal(0.0, scale, size=base.shape)
    return FourierMotion(base=base, coefficients=coefficients)


def _session_terms(
    base: np.ndarray,
    tee_visible: np.ndarray,
    static_scale: float = 1.0,
    dynamic_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    static = np.zeros_like(base)
    dynamic = np.zeros_like(base)
    static[tee_visible, :2] = static_scale * 0.012 * base[tee_visible, :2]
    dynamic[tee_visible, 2] = dynamic_scale * 0.55
    return static, dynamic


def session_residual(
    phases: np.ndarray,
    static: np.ndarray,
    dynamic: np.ndarray,
) -> np.ndarray:
    modulation = np.cos(2.0 * np.pi * phases)[..., None, None]
    return static + modulation * dynamic


def _production_runtime(topology: HeterogeneousTemplate) -> tuple[Any, dict[str, np.ndarray], dict[str, np.ndarray]]:
    components = topology.components

    def global_vertices(*names: str) -> np.ndarray:
        values = np.concatenate([components[name].local_to_global for name in names])
        return np.unique(values).astype(np.int64)

    def global_faces(*names: str) -> np.ndarray:
        return np.concatenate(
            [
                components[name].local_to_global[components[name].faces]
                for name in names
            ],
            axis=0,
        ).astype(np.int64)

    leaflet_vertices = global_vertices("anterior_leaflet", "posterior_leaflet")
    leaflet_free = np.setdiff1d(
        leaflet_vertices, topology.annulus_global_indices, assume_unique=False
    )
    groups = {
        "annulus": topology.annulus_global_indices,
        "leaflet_free": leaflet_free,
        "cmr_visible": global_vertices(
            "lv_endocardium", "lv_epicardium", "left_atrium"
        ),
        "tee_visible": global_vertices(
            "left_atrium", "anterior_leaflet", "posterior_leaflet"
        ),
    }
    faces = {
        "cmr_surface": global_faces(
            "lv_endocardium", "lv_epicardium", "left_atrium"
        ),
        "tee_surface": global_faces(
            "left_atrium", "anterior_leaflet", "posterior_leaflet"
        ),
    }
    return topology.attachment_map, groups, faces


def generate_case(
    seed: int = 7,
    n_cmr_frames: int = 20,
    n_tee_frames: int = 24,
    fourier_order: int = 3,
    cmr_noise_mm: float = 0.35,
    tee_noise_mm: float = 0.45,
    cmr_dropout_fraction: float = 0.0,
    tee_dropout_fraction: float = 0.20,
    tee_axis_angle_rad: tuple[float, float, float] = (0.08, -0.12, 0.18),
    tee_translation_mm: tuple[float, float, float] = (7.0, -4.0, 11.0),
    phase_logits: tuple[float, ...] = (-0.4, 0.2, 0.8, -0.1, 0.5, -0.6, 0.1, 0.4),
    phase_offset: float = 0.13,
    n_annulus: int = 24,
    template_vertices: np.ndarray | None = None,
    component_seeds: dict[str, int] | None = None,
    session_static_field: np.ndarray | None = None,
    session_dynamic_field: np.ndarray | None = None,
    session_static_scale: float = 1.0,
    session_dynamic_scale: float = 1.0,
    observation_domain: str = "surface_points",
    cmr_n_slices: int = 5,
    cmr_image_shape: tuple[int, int] = (24, 24),
    cmr_pixel_spacing_mm: tuple[float, float] = (2.5, 2.5),
    cmr_slice_thickness_mm: float = 6.0,
    cmr_surface_sigma_mm: float = 1.5,
    tee_grid_shape: tuple[int, int, int] = (20, 12, 12),
    tee_surface_sigma_mm: float = 1.5,
    image_noise_std: float = 0.015,
    tee_speckle_std: float = 0.12,
    topology: HeterogeneousTemplate | None = None,
    chamber_levels: int = 1,
    leaflet_rows: int = 1,
    tee_shadow: bool = True,
) -> SyntheticCase:
    if not 0.0 <= cmr_dropout_fraction < 1.0:
        raise ValueError("cmr_dropout_fraction must lie in [0, 1)")
    if not 0.0 <= tee_dropout_fraction < 1.0:
        raise ValueError("tee_dropout_fraction must lie in [0, 1)")
    if observation_domain not in {"surface_points", "image_likelihood"}:
        raise ValueError("observation_domain must be surface_points or image_likelihood")
    if component_seeds is None:
        sequences = np.random.SeedSequence(seed).spawn(8)
        component_seeds = {
            "motion": int(sequences[0].generate_state(1, dtype=np.uint64)[0]),
            "cmr_dropout": int(sequences[1].generate_state(1, dtype=np.uint64)[0]),
            "tee_dropout": int(sequences[2].generate_state(1, dtype=np.uint64)[0]),
            "cmr_noise": int(sequences[3].generate_state(1, dtype=np.uint64)[0]),
            "tee_noise": int(sequences[4].generate_state(1, dtype=np.uint64)[0]),
        }
        if observation_domain == "image_likelihood":
            component_seeds.update(
                {
                    "cmr_image": int(sequences[5].generate_state(1, dtype=np.uint64)[0]),
                    "tee_image": int(sequences[6].generate_state(1, dtype=np.uint64)[0]),
                    "tee_shadow": int(sequences[7].generate_state(1, dtype=np.uint64)[0]),
                }
            )
    required_seeds = {"motion", "cmr_dropout", "tee_dropout", "cmr_noise", "tee_noise"}
    if observation_domain == "image_likelihood":
        required_seeds |= {"cmr_image", "tee_image", "tee_shadow"}
    if set(component_seeds) != required_seeds:
        raise ValueError(f"component_seeds must contain exactly {sorted(required_seeds)}")
    rng_motion = np.random.default_rng(component_seeds["motion"])
    rng_cmr_dropout = np.random.default_rng(component_seeds["cmr_dropout"])
    rng_tee_dropout = np.random.default_rng(component_seeds["tee_dropout"])
    rng_cmr_noise = np.random.default_rng(component_seeds["cmr_noise"])
    rng_tee_noise = np.random.default_rng(component_seeds["tee_noise"])
    if topology is None:
        fixture = make_toy_complex(n_annulus=n_annulus, chamber_levels=chamber_levels,
                                   leaflet_rows=leaflet_rows)
        default_vertices = fixture.global_vertices
        attachment_map = fixture.attachment_map
        groups = fixture.groups
        faces = fixture.faces
    else:
        topology.validate().raise_if_failed()
        if n_annulus != len(topology.annulus_global_indices):
            raise ValueError(
                "n_annulus must match the frozen production template annular count"
            )
        default_vertices = topology.vertices
        attachment_map, groups, faces = _production_runtime(topology)
    base = default_vertices
    if template_vertices is not None:
        base = np.asarray(template_vertices, dtype=np.float64)
        if base.shape != default_vertices.shape:
            raise ValueError(
                f"template_vertices must have shape {default_vertices.shape}, got {base.shape}"
            )
        if not np.all(np.isfinite(base)):
            raise ValueError("template_vertices contains non-finite values")
    interior_rows: tuple[tuple[np.ndarray, float], ...] = ()
    if topology is None and leaflet_rows > 1:
        rows = np.setdiff1d(groups["tee_visible"], groups["annulus"]).reshape(leaflet_rows, n_annulus)
        interior_rows = tuple((rows[r], (r + 1) / leaflet_rows) for r in range(leaflet_rows - 1))
    motion = _default_motion(
        base,
        groups["leaflet_free"],
        fourier_order,
        rng_motion,
        leaflet_interior=interior_rows,
    )
    warp = MonotoneCircularPhaseWarp(np.asarray(phase_logits), offset=phase_offset)
    pose = RigidTransform.from_axis_angle(
        np.asarray(tee_axis_angle_rad),
        np.asarray(tee_translation_mm),
    )
    if session_static_field is not None or session_dynamic_field is not None:
        # Measured session fields, drawn by the caller from the MITEA subspace.
        # The procedural alternative below displaces vertices by about 0.2 mm,
        # where real between-session variation is 1.19 mm correspondence-free and
        # 3.08 mm per vertex, so a benchmark built on it cannot test the session
        # term at all: section 6 measured its contribution as 0.003 mm against a
        # 2.2 mm representation floor.
        if session_static_field is None or session_dynamic_field is None:
            raise ValueError("supply both session fields or neither")
        static = np.asarray(session_static_field, dtype=np.float64) * session_static_scale
        dynamic = np.asarray(session_dynamic_field, dtype=np.float64) * session_dynamic_scale
        if static.shape != base.shape or dynamic.shape != base.shape:
            raise ValueError("session fields must match the template shape")
    else:
        static, dynamic = _session_terms(
            base,
            groups["tee_visible"],
            static_scale=session_static_scale,
            dynamic_scale=session_dynamic_scale,
        )

    cmr_times = np.linspace(0.0, 1.0, n_cmr_frames, endpoint=False)
    tee_times = np.linspace(0.0, 1.0, n_tee_frames, endpoint=False)
    tee_true_phases = warp.evaluate(tee_times)

    common_cmr = motion.evaluate(cmr_times)
    common_tee = motion.evaluate(tee_true_phases)
    # Symmetric residuals assign the cross-session mean to shared motion.
    cmr_geometry = common_cmr - 0.5 * session_residual(cmr_times, static, dynamic)
    tee_geometry = common_tee + 0.5 * session_residual(tee_true_phases, static, dynamic)

    all_cmr_indices = groups["cmr_visible"].copy()
    keep_cmr = max(1, int(round((1.0 - cmr_dropout_fraction) * len(all_cmr_indices))))
    cmr_indices = np.sort(rng_cmr_dropout.choice(all_cmr_indices, size=keep_cmr, replace=False))
    all_tee_indices = groups["tee_visible"].copy()
    keep = max(1, int(round((1.0 - tee_dropout_fraction) * len(all_tee_indices))))
    tee_indices = np.sort(rng_tee_dropout.choice(all_tee_indices, size=keep, replace=False))

    cmr_points = cmr_geometry[:, cmr_indices, :] + rng_cmr_noise.normal(
        0.0,
        cmr_noise_mm,
        size=(n_cmr_frames, len(cmr_indices), 3),
    )
    tee_points = pose.apply(tee_geometry[:, tee_indices, :]) + rng_tee_noise.normal(
        0.0,
        tee_noise_mm,
        size=(n_tee_frames, len(tee_indices), 3),
    )

    endpoint_geometry = motion.evaluate(np.array([0.0, 1.0]))
    parts = attachment_map.assemble(endpoint_geometry)
    attachment_residual = attachment_map.max_residual(parts)

    cmr_image = None
    cmr_affine = None
    cmr_offsets = None
    tee_image = None
    tee_grid_apex = None
    tee_grid_axis = None
    tee_grid_depths = None
    tee_grid_azimuths = None
    tee_grid_elevations = None
    if observation_domain == "image_likelihood":
        cmr_stack = default_cmr_stack(
            base[groups["cmr_visible"]],
            n_slices=cmr_n_slices,
            image_shape=tuple(int(value) for value in cmr_image_shape),
            pixel_spacing_mm=tuple(float(value) for value in cmr_pixel_spacing_mm),
            slice_thickness_mm=cmr_slice_thickness_mm,
        )
        cmr_image = render_cmr_surface_likelihood(
            cmr_geometry,
            faces["cmr_surface"],
            cmr_stack,
            sigma_mm=cmr_surface_sigma_mm,
            through_plane_samples=3,
            noise_std=image_noise_std,
            seed=component_seeds["cmr_image"],
        )
        posed_tee_geometry = pose.apply(tee_geometry)
        tee_grid = default_tee_grid(
            posed_tee_geometry[0, groups["tee_visible"]],
            grid_shape=tuple(int(value) for value in tee_grid_shape),
        )
        shadow_rng = np.random.default_rng(component_seeds["tee_shadow"])
        shadow = ShadowCone(
            azimuth_rad=float(shadow_rng.uniform(-0.20, 0.20)),
            elevation_rad=float(shadow_rng.uniform(-0.20, 0.20)),
            angular_radius_rad=float(shadow_rng.uniform(0.08, 0.18)),
            start_depth_mm=float(
                shadow_rng.uniform(tee_grid.depths_mm[0], tee_grid.depths_mm[-1])
            ),
            strength=float(shadow_rng.uniform(0.35, 0.80)),
        )
        tee_image = render_tee_surface_likelihood(
            posed_tee_geometry,
            faces["tee_surface"],
            tee_grid,
            sigma_mm=tee_surface_sigma_mm,
            # The shadow is drawn either way, from its own seed, so switching it
            # off changes the TEE image and nothing else.
            shadows=(shadow,) if tee_shadow else (),
            speckle_std=tee_speckle_std,
            noise_std=image_noise_std,
            seed=component_seeds["tee_image"],
        )
        cmr_affine = cmr_stack.affine_lps()
        cmr_offsets = cmr_stack.slice_offsets_mm
        tee_grid_apex = tee_grid.apex
        tee_grid_axis = tee_grid.axis
        tee_grid_depths = tee_grid.depths_mm
        tee_grid_azimuths = tee_grid.azimuths_rad
        tee_grid_elevations = tee_grid.elevations_rad

    return SyntheticCase(
        template_vertices=base,
        motion_coefficients=motion.coefficients,
        cmr_times=cmr_times,
        cmr_indices=cmr_indices,
        cmr_points=cmr_points,
        tee_times=tee_times,
        tee_true_phases=tee_true_phases,
        tee_indices=tee_indices,
        tee_points=tee_points,
        pose_rotation=pose.rotation,
        pose_translation=pose.translation,
        phase_logits=warp.logits,
        phase_offset=warp.offset,
        session_static=static,
        session_dynamic=dynamic,
        attachment_residual_mm=attachment_residual,
        periodic_seam_mm=motion.seam_error(),
        phase_minimum_increment=warp.minimum_increment(),
        local_to_global=dict(attachment_map.local_to_global),
        chamber_levels=chamber_levels,
        leaflet_rows=leaflet_rows,
        observation_domain=observation_domain,
        cmr_surface_likelihood=cmr_image,
        cmr_plane_affine_lps=cmr_affine,
        cmr_slice_offsets_mm=cmr_offsets,
        tee_surface_likelihood=tee_image,
        tee_grid_apex=tee_grid_apex,
        tee_grid_axis=tee_grid_axis,
        tee_grid_depths_mm=tee_grid_depths,
        tee_grid_azimuths_rad=tee_grid_azimuths,
        tee_grid_elevations_rad=tee_grid_elevations,
    )


def save_case(case: SyntheticCase, output_directory: str | Path) -> tuple[Path, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    array_path = output / "case.npz"
    metadata_path = output / "metadata.json"
    np.savez_compressed(array_path, **case.arrays())
    metadata_path.write_text(json.dumps(case.metadata(), indent=2) + "\n", encoding="utf-8")
    return array_path, metadata_path
