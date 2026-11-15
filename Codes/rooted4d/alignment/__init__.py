"""Dependency-light patient-to-template alignment primitives."""

from .features import (
    area_weighted_vertex_normals,
    barycentric_projection_matrix,
    circular_coordinate_residual,
    directed_arc_alignment_loss,
    directed_arc_coordinates,
    normal_compatibility_loss,
    patch_admissibility_mask,
    soft_correspondence_weights,
)

__all__ = [
    "area_weighted_vertex_normals",
    "barycentric_projection_matrix",
    "circular_coordinate_residual",
    "directed_arc_alignment_loss",
    "directed_arc_coordinates",
    "normal_compatibility_loss",
    "patch_admissibility_mask",
    "soft_correspondence_weights",
]
