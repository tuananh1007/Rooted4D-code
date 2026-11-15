r"""Differentiable point-to-mesh distance and surface likelihood.

This is the primitive the image-domain observation operators are built from, and
the one that makes their Jacobian expensive. The NumPy version in
``rooted4d.observation.distance`` remains the definition; this mirrors it in
PyTorch so the derivative can be taken exactly rather than by differencing.

The closest-point computation clamps into the triangle edge by edge rather than
clipping the barycentric coordinates independently. Independent clipping lands
outside the simplex whenever ``u + v > 1``, which is a wrong distance rather than
an imprecise one, and it is the mistake the NumPy version documents.
"""

from __future__ import annotations

import torch

__all__ = ["closest_point_on_triangles", "point_to_mesh_distance", "surface_likelihood"]


def closest_point_on_triangles(
    points: torch.Tensor, vertices: torch.Tensor, faces: torch.Tensor
) -> torch.Tensor:
    """Squared distance from each point to every triangle, shape (points, faces)."""

    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    ab = b - a
    ac = c - a
    ap = points[:, None, :] - a[None, :, :]

    d1 = torch.einsum("ptk,tk->pt", ap, ab)
    d2 = torch.einsum("ptk,tk->pt", ap, ac)
    aa = torch.einsum("tk,tk->t", ab, ab)[None, :]
    bb = torch.einsum("tk,tk->t", ab, ac)[None, :]
    cc = torch.einsum("tk,tk->t", ac, ac)[None, :]
    denominator = torch.clamp(aa * cc - bb * bb, min=1e-30)
    u = (cc * d1 - bb * d2) / denominator
    v = (aa * d2 - bb * d1) / denominator

    outside = (u < 0.0) | (v < 0.0) | (u + v > 1.0)

    def project(numerator: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
        return torch.clamp(numerator / torch.clamp(denom, min=1e-30), 0.0, 1.0)

    bc = c - b
    bp = torch.einsum("ptk,tk->pt", points[:, None, :] - b[None, :, :], bc)
    t_ab = project(d1, aa)
    t_ac = project(d2, cc)
    t_bc = project(bp, torch.einsum("tk,tk->t", bc, bc)[None, :])
    candidates = torch.stack(
        [
            torch.stack([t_ab, torch.zeros_like(t_ab)], dim=-1),
            torch.stack([torch.zeros_like(t_ac), t_ac], dim=-1),
            torch.stack([1.0 - t_bc, t_bc], dim=-1),
        ],
        dim=0,
    )
    edge_points = (
        a[None, None, :, :]
        + candidates[..., 0][..., None] * ab[None, None, :, :]
        + candidates[..., 1][..., None] * ac[None, None, :, :]
    )
    edge_squared = ((edge_points - points[None, :, None, :]) ** 2).sum(dim=-1)
    best = torch.argmin(edge_squared, dim=0)
    chosen = torch.gather(
        candidates, 0, best[None, ..., None].expand(1, *best.shape, 2)
    )[0]
    u = torch.where(outside, chosen[..., 0], u)
    v = torch.where(outside, chosen[..., 1], v)

    closest = a[None, :, :] + u[..., None] * ab[None, :, :] + v[..., None] * ac[None, :, :]
    return ((closest - points[:, None, :]) ** 2).sum(dim=-1)


def point_to_mesh_distance(
    points: torch.Tensor, vertices: torch.Tensor, faces: torch.Tensor
) -> torch.Tensor:
    """Distance from each point to the closest point on the surface."""

    squared = closest_point_on_triangles(points, vertices, faces).min(dim=1).values
    # The square root is not differentiable at zero, and a query point sitting
    # exactly on the surface is not exotic here: it is what a converged fit
    # looks like. Clamping keeps the gradient finite there instead of producing
    # a NaN that would only appear once the solver was doing well.
    return torch.sqrt(torch.clamp(squared, min=1e-24))


def surface_likelihood(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    sigma_mm: float,
) -> torch.Tensor:
    """Gaussian surface likelihood, matching the NumPy observation operator."""

    if sigma_mm <= 0.0:
        raise ValueError("sigma_mm must be positive")
    distance = point_to_mesh_distance(points, vertices, faces)
    return torch.exp(-0.5 * (distance / sigma_mm) ** 2)
