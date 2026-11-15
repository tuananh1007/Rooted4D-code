"""Metrics tied to Rooted4D's defensible claims."""

from .identifiability import IdentifiabilitySummary, summarize_jacobian
from .interface import attachment_residual_over_time
from .simulation_readiness import (
    DEFAULT_WELD_TOLERANCES,
    SimulationReadinessReport,
    assess_simulation_readiness,
    interface_welds_at,
    weld_vertices,
)
from .topology import (
    SurfaceTopologyReport,
    component_overlap_volumes,
    count_self_intersections,
    edge_incidence_counts,
    euler_characteristic,
    signed_enclosed_volume,
    surface_area,
    surface_topology_report,
    triangles_intersect,
)

__all__ = [
    "IdentifiabilitySummary",
    "summarize_jacobian",
    "attachment_residual_over_time",
    "DEFAULT_WELD_TOLERANCES",
    "SimulationReadinessReport",
    "assess_simulation_readiness",
    "interface_welds_at",
    "weld_vertices",
    "SurfaceTopologyReport",
    "component_overlap_volumes",
    "count_self_intersections",
    "edge_incidence_counts",
    "euler_characteristic",
    "signed_enclosed_volume",
    "surface_area",
    "surface_topology_report",
    "triangles_intersect",
]
