"""Diverging 3D TEE-sector surface-likelihood renderer with explicit shadows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distance import surface_likelihood


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must have finite shape (3,)")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return value / norm


@dataclass(frozen=True)
class ShadowCone:
    azimuth_rad: float
    elevation_rad: float
    angular_radius_rad: float
    start_depth_mm: float
    strength: float

    def __post_init__(self) -> None:
        if self.angular_radius_rad <= 0.0 or self.start_depth_mm < 0.0:
            raise ValueError("shadow radius must be positive and start depth non-negative")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("shadow strength must lie in [0, 1]")


@dataclass(frozen=True)
class TEESectorGrid:
    apex: np.ndarray
    axis: np.ndarray
    lateral_direction: np.ndarray
    depths_mm: np.ndarray
    azimuths_rad: np.ndarray
    elevations_rad: np.ndarray

    def __post_init__(self) -> None:
        apex = np.asarray(self.apex, dtype=np.float64)
        axis = _unit(self.axis, "axis")
        lateral = _unit(self.lateral_direction, "lateral_direction")
        depths = np.asarray(self.depths_mm, dtype=np.float64)
        azimuths = np.asarray(self.azimuths_rad, dtype=np.float64)
        elevations = np.asarray(self.elevations_rad, dtype=np.float64)
        if apex.shape != (3,) or not np.all(np.isfinite(apex)):
            raise ValueError("apex must have finite shape (3,)")
        lateral = lateral - float(lateral @ axis) * axis
        lateral = _unit(lateral, "orthogonalized lateral_direction")
        if depths.ndim != 1 or len(depths) < 2 or np.any(np.diff(depths) <= 0.0):
            raise ValueError("depths_mm must be a strictly increasing vector")
        if depths[0] <= 0.0:
            raise ValueError("depths_mm must be positive")
        if azimuths.ndim != 1 or elevations.ndim != 1:
            raise ValueError("azimuths_rad and elevations_rad must be vectors")
        if len(azimuths) < 2 or len(elevations) < 2:
            raise ValueError("TEE angular grids must each contain at least two values")
        object.__setattr__(self, "apex", apex)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "lateral_direction", lateral)
        object.__setattr__(self, "depths_mm", depths)
        object.__setattr__(self, "azimuths_rad", azimuths)
        object.__setattr__(self, "elevations_rad", elevations)

    @property
    def elevation_direction(self) -> np.ndarray:
        return np.cross(self.axis, self.lateral_direction)

    def ray_directions(self) -> np.ndarray:
        elevation, azimuth = np.meshgrid(
            self.elevations_rad, self.azimuths_rad, indexing="ij"
        )
        directions = (
            self.axis
            + np.tan(azimuth)[..., None] * self.lateral_direction
            + np.tan(elevation)[..., None] * self.elevation_direction
        )
        return directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    def world_grid(self) -> np.ndarray:
        directions = self.ray_directions()
        return self.apex + self.depths_mm[:, None, None, None] * directions[None, ...]


def default_tee_grid(
    vertices: np.ndarray,
    *,
    grid_shape: tuple[int, int, int] = (24, 16, 16),
    field_of_view_degrees: tuple[float, float] = (70.0, 70.0),
) -> TEESectorGrid:
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("vertices must have finite shape (vertices, 3)")
    if len(grid_shape) != 3 or any(int(value) < 2 for value in grid_shape):
        raise ValueError("grid_shape must contain depth/elevation/azimuth sizes >= 2")
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    apex = center + np.array([0.0, 0.0, max(45.0, 0.7 * diagonal)])
    axis = _unit(center - apex, "default TEE axis")
    maximum_depth = float(np.linalg.norm(center - apex) + 0.65 * diagonal)
    minimum_depth = max(2.0, float(np.linalg.norm(center - apex) - 0.65 * diagonal))
    elevation_half = np.deg2rad(field_of_view_degrees[0] / 2.0)
    azimuth_half = np.deg2rad(field_of_view_degrees[1] / 2.0)
    return TEESectorGrid(
        apex=apex,
        axis=axis,
        lateral_direction=np.array([1.0, 0.0, 0.0]),
        depths_mm=np.linspace(minimum_depth, maximum_depth, int(grid_shape[0])),
        azimuths_rad=np.linspace(-azimuth_half, azimuth_half, int(grid_shape[2])),
        elevations_rad=np.linspace(-elevation_half, elevation_half, int(grid_shape[1])),
    )


def _shadow_mask(grid: TEESectorGrid, cones: tuple[ShadowCone, ...]) -> np.ndarray:
    elevation, azimuth = np.meshgrid(
        grid.elevations_rad, grid.azimuths_rad, indexing="ij"
    )
    mask = np.ones(
        (len(grid.depths_mm), len(grid.elevations_rad), len(grid.azimuths_rad)),
        dtype=np.float64,
    )
    for cone in cones:
        angular_distance = np.sqrt(
            (azimuth - cone.azimuth_rad) ** 2 + (elevation - cone.elevation_rad) ** 2
        )
        angular_weight = np.exp(-0.5 * (angular_distance / cone.angular_radius_rad) ** 2)
        depth_weight = (grid.depths_mm >= cone.start_depth_mm).astype(np.float64)
        mask *= 1.0 - cone.strength * depth_weight[:, None, None] * angular_weight[None]
    return np.clip(mask, 0.0, 1.0)


def render_tee_surface_likelihood(
    vertices: np.ndarray,
    faces: np.ndarray,
    grid: TEESectorGrid,
    *,
    sigma_mm: float = 1.5,
    attenuation_per_mm: float = 0.003,
    shadows: tuple[ShadowCone, ...] = (),
    speckle_std: float = 0.0,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Render a dynamic surface likelihood in a diverging TEE sector.

    The geometry, sector sampling, depth attenuation, stochastic speckle, and
    explicit shadow cones are implemented. This is intentionally described as
    a likelihood renderer, not as a diagnostic-quality ultrasound simulator.
    """

    values = np.asarray(vertices, dtype=np.float64)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("vertices must have shape (vertices, 3) or (frames, vertices, 3)")
    if attenuation_per_mm < 0.0 or speckle_std < 0.0 or noise_std < 0.0:
        raise ValueError("attenuation and noise parameters cannot be negative")
    world = grid.world_grid()
    depth_attenuation = np.exp(-attenuation_per_mm * grid.depths_mm)[:, None, None]
    visibility = depth_attenuation * _shadow_mask(grid, shadows)
    rendered = [
        surface_likelihood(world, frame, faces, sigma_mm=sigma_mm) * visibility
        for frame in values
    ]
    result = np.asarray(rendered, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if speckle_std:
        result *= np.exp(rng.normal(-0.5 * speckle_std**2, speckle_std, result.shape))
    if noise_std:
        result += rng.normal(0.0, noise_std, result.shape)
    return np.clip(result, 0.0, 1.0).astype(np.float32)
