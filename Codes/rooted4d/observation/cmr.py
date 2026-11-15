"""Finite-thickness Cartesian CMR plane-stack surface-likelihood renderer."""

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
class CMRPlaneStack:
    """World-coordinate geometry of a parallel finite-thickness plane stack."""

    center: np.ndarray
    row_direction: np.ndarray
    column_direction: np.ndarray
    slice_offsets_mm: np.ndarray
    image_shape: tuple[int, int]
    pixel_spacing_mm: tuple[float, float]
    slice_thickness_mm: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        row = _unit(self.row_direction, "row_direction")
        column = _unit(self.column_direction, "column_direction")
        offsets = np.asarray(self.slice_offsets_mm, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("center must have finite shape (3,)")
        if abs(float(row @ column)) > 1e-8:
            raise ValueError("row_direction and column_direction must be orthogonal")
        if offsets.ndim != 1 or len(offsets) == 0 or not np.all(np.isfinite(offsets)):
            raise ValueError("slice_offsets_mm must be a non-empty finite vector")
        if len(self.image_shape) != 2 or any(int(value) < 2 for value in self.image_shape):
            raise ValueError("image_shape must contain two dimensions of at least 2")
        if len(self.pixel_spacing_mm) != 2 or any(
            float(value) <= 0.0 for value in self.pixel_spacing_mm
        ):
            raise ValueError("pixel_spacing_mm must contain two positive values")
        if self.slice_thickness_mm <= 0.0:
            raise ValueError("slice_thickness_mm must be positive")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "row_direction", row)
        object.__setattr__(self, "column_direction", column)
        object.__setattr__(self, "slice_offsets_mm", offsets)
        object.__setattr__(self, "image_shape", tuple(int(v) for v in self.image_shape))
        object.__setattr__(
            self, "pixel_spacing_mm", tuple(float(v) for v in self.pixel_spacing_mm)
        )

    @property
    def normal(self) -> np.ndarray:
        return np.cross(self.row_direction, self.column_direction)

    def world_grid(self, through_plane_fraction: float = 0.0) -> np.ndarray:
        if not -0.5 <= through_plane_fraction <= 0.5:
            raise ValueError("through_plane_fraction must lie in [-0.5, 0.5]")
        rows, columns = self.image_shape
        row_coordinates = (np.arange(rows) - 0.5 * (rows - 1)) * self.pixel_spacing_mm[0]
        column_coordinates = (
            np.arange(columns) - 0.5 * (columns - 1)
        ) * self.pixel_spacing_mm[1]
        grid_row, grid_column = np.meshgrid(row_coordinates, column_coordinates, indexing="ij")
        plane_centers = (
            self.center[None, :]
            + self.slice_offsets_mm[:, None] * self.normal[None, :]
            + through_plane_fraction * self.slice_thickness_mm * self.normal[None, :]
        )
        return (
            plane_centers[:, None, None, :]
            + grid_row[None, :, :, None] * self.row_direction
            + grid_column[None, :, :, None] * self.column_direction
        )

    def affine_lps(self) -> np.ndarray:
        """Return one 4x4 center-slice pixel-to-world affine in LPS-like coordinates."""

        rows, columns = self.image_shape
        origin = (
            self.center
            - 0.5 * (rows - 1) * self.pixel_spacing_mm[0] * self.row_direction
            - 0.5 * (columns - 1) * self.pixel_spacing_mm[1] * self.column_direction
            + self.slice_offsets_mm[0] * self.normal
        )
        affine = np.eye(4, dtype=np.float64)
        affine[:3, 0] = self.row_direction * self.pixel_spacing_mm[0]
        affine[:3, 1] = self.column_direction * self.pixel_spacing_mm[1]
        affine[:3, 2] = self.normal
        affine[:3, 3] = origin
        return affine


def default_cmr_stack(
    vertices: np.ndarray,
    *,
    n_slices: int = 5,
    image_shape: tuple[int, int] = (32, 32),
    pixel_spacing_mm: tuple[float, float] = (2.0, 2.0),
    slice_thickness_mm: float = 6.0,
) -> CMRPlaneStack:
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("vertices must have finite shape (vertices, 3)")
    if n_slices < 1:
        raise ValueError("n_slices must be positive")
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    extent_z = float(points[:, 2].max() - points[:, 2].min())
    offsets = np.linspace(-0.45 * extent_z, 0.45 * extent_z, n_slices)
    return CMRPlaneStack(
        center=center,
        row_direction=np.array([1.0, 0.0, 0.0]),
        column_direction=np.array([0.0, 1.0, 0.0]),
        slice_offsets_mm=offsets,
        image_shape=image_shape,
        pixel_spacing_mm=pixel_spacing_mm,
        slice_thickness_mm=slice_thickness_mm,
    )


def render_cmr_surface_likelihood(
    vertices: np.ndarray,
    faces: np.ndarray,
    stack: CMRPlaneStack,
    *,
    sigma_mm: float = 1.25,
    through_plane_samples: int = 3,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Render a dynamic triangle surface into finite-thickness CMR-like planes.

    This is an observation likelihood for segmentation/contour supervision, not
    an MR pulse-sequence simulator. It uses exact point-to-triangle distances and
    explicitly samples slice thickness.
    """

    values = np.asarray(vertices, dtype=np.float64)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("vertices must have shape (vertices, 3) or (frames, vertices, 3)")
    if through_plane_samples < 1:
        raise ValueError("through_plane_samples must be positive")
    if noise_std < 0.0:
        raise ValueError("noise_std cannot be negative")
    fractions = np.linspace(-0.5, 0.5, through_plane_samples)
    query_grids = [stack.world_grid(float(fraction)) for fraction in fractions]
    rendered = []
    for frame in values:
        samples = [
            surface_likelihood(grid, frame, faces, sigma_mm=sigma_mm)
            for grid in query_grids
        ]
        rendered.append(np.mean(samples, axis=0))
    result = np.asarray(rendered, dtype=np.float32)
    if noise_std:
        rng = np.random.default_rng(seed)
        result = np.clip(result + rng.normal(0.0, noise_std, result.shape), 0.0, 1.0)
    return result.astype(np.float32, copy=False)
