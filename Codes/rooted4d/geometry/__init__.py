"""Fixed-topology geometry, validity, shell, rigid, and embedding utilities."""

from .deformation import (
    DeformationResult,
    barrier_value_and_gradient,
    boundary_faces,
    collision_value_and_gradient,
    elastic_value_and_gradient,
    fit_deformation,
    lattice_mesh,
    tetrahedron_edges,
)
from .embedding import BarycentricEmbedding, bounded_residual, fit_tetrahedral_embedding
from .rigid import RigidTransform, axis_angle_to_matrix
from .shell import (
    ShellReference,
    build_shell_reference,
    shell_bending_energy,
    shell_membrane_energy,
)
from .template import (
    HeterogeneousTemplate,
    SurfacePatch,
    TemplateValidationReport,
    load_frozen_template,
)
from .tetrahedra import (
    TetrahedronQualityReport,
    deformation_jacobian_determinants,
    interior_log_barrier,
    signed_tetrahedron_volumes,
    tetrahedron_jacobian_barrier,
    tetrahedron_quality_report,
    tetrahedron_scaled_jacobians,
)
from .toy import TOPOLOGY_KEYS, ToyComplex, make_toy_complex, sample_topology, toy_complex_for_vertices

__all__ = [
    "collision_value_and_gradient",
    "DeformationResult",
    "barrier_value_and_gradient",
    "boundary_faces",
    "elastic_value_and_gradient",
    "fit_deformation",
    "lattice_mesh",
    "tetrahedron_edges",
    "BarycentricEmbedding",
    "bounded_residual",
    "fit_tetrahedral_embedding",
    "RigidTransform",
    "axis_angle_to_matrix",
    "ShellReference",
    "build_shell_reference",
    "shell_bending_energy",
    "shell_membrane_energy",
    "TetrahedronQualityReport",
    "deformation_jacobian_determinants",
    "interior_log_barrier",
    "signed_tetrahedron_volumes",
    "tetrahedron_jacobian_barrier",
    "tetrahedron_quality_report",
    "tetrahedron_scaled_jacobians",
    "ToyComplex",
    "make_toy_complex",
    "toy_complex_for_vertices",
    "sample_topology",
    "TOPOLOGY_KEYS",
    "HeterogeneousTemplate",
    "SurfacePatch",
    "TemplateValidationReport",
    "load_frozen_template",
]
