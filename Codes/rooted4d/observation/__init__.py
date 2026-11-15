"""Image-domain observation operators for CMR planes and TEE sectors."""

from .cmr import CMRPlaneStack, default_cmr_stack, render_cmr_surface_likelihood
from .distance import point_to_mesh_distance, surface_likelihood
from .slicing import (
    CMRPlane,
    MeshPlaneIntersection,
    intersect_surface_with_plane,
    sample_plane_segments,
    symmetric_contour_chamfer,
)
from .tee import ShadowCone, TEESectorGrid, default_tee_grid, render_tee_surface_likelihood

__all__ = [
    "CMRPlaneStack",
    "CMRPlane",
    "MeshPlaneIntersection",
    "ShadowCone",
    "TEESectorGrid",
    "default_cmr_stack",
    "default_tee_grid",
    "point_to_mesh_distance",
    "intersect_surface_with_plane",
    "render_cmr_surface_likelihood",
    "render_tee_surface_likelihood",
    "surface_likelihood",
    "sample_plane_segments",
    "symmetric_contour_chamfer",
]
