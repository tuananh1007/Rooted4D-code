"""Truth and fitted anatomy as vertex trajectories, for measurements made on fits.

Stored benchmark samples carry their composite anatomy as ``template_vertices``
and its periodic motion as ``motion_coefficients`` (the generator's
``FourierMotion``). That pair is the shared, session-free anatomy in the CMR
frame: the generator adds half the session field to one acquisition and
subtracts it from the other. A fit's counterpart is its session-free latent
trajectory mapped to physical vertices. Both are returned for every physical
vertex, in template order, so a measurement can index the reduced topology
directly (``ring_groups``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Mapping

import numpy as np

from rooted4d.inference.joint import SolverConfig, latent_trajectory
from rooted4d.inference.scene import SceneModel, SceneParameters
from rooted4d.motion import FourierMotion


def ring_groups(n_annulus: int) -> dict[str, np.ndarray]:
    """Vertex ranges of the reduced topology (``make_toy_complex``).

    The anterior leaflet has root arc ``0..n/2-1`` and free edge
    ``2n+1..2n+n/2``; the posterior leaflet has the rest of each ring.
    """

    n, half = n_annulus, n_annulus // 2
    return {
        "annulus": np.arange(0, n), "lower": np.arange(n, 2 * n), "apex": np.array([2 * n]),
        "free": np.arange(2 * n + 1, 3 * n + 1),
        "anterior_root": np.arange(0, half), "posterior_root": np.arange(half, n),
        "anterior_free": np.arange(2 * n + 1, 2 * n + 1 + half),
        "posterior_free": np.arange(2 * n + 1 + half, 3 * n + 1),
    }


def truth_trajectory(sample: Mapping[str, np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """The sample's shared anatomy at any phases, shape (phases, vertices, 3)."""

    motion = FourierMotion(base=np.asarray(sample["template_vertices"], dtype=np.float64),
                           coefficients=np.asarray(sample["motion_coefficients"], dtype=np.float64))
    return lambda phases: motion.evaluate(np.mod(np.asarray(phases, dtype=np.float64), 1.0))


def fitted_trajectory(parameters: SceneParameters, model: SceneModel, config: SolverConfig,
                      phases: np.ndarray, n_vertices: int) -> np.ndarray:
    """A fit's session-free anatomy at ``phases``, every physical vertex in template order."""

    shared = replace(config, use_session=False)
    latent = latent_trajectory(parameters, np.mod(np.asarray(phases, dtype=np.float64), 1.0),
                               "cmr", model, shared)
    coordinates = model.physical_coordinates(latent)
    return coordinates[..., model.observation_rows(np.arange(n_vertices)), :]


def chamber_volume(vertices: np.ndarray, n_annulus: int) -> float:
    """Volume of the reduced chamber: annulus, lower ring and apex, closed by a cap.

    The surface is a fan over the annulus centroid (the cap), a band of quads
    between the annulus and the lower ring, and a cone from the lower ring to the
    apex. Its volume is the divergence-theorem sum over those triangles; the
    orientation is fixed by taking the absolute value.
    """

    groups = ring_groups(n_annulus)
    ring, lower = vertices[groups["annulus"]], vertices[groups["lower"]]
    apex, cap = vertices[groups["apex"][0]], vertices[groups["annulus"]].mean(axis=0)
    triangles = []
    for i in range(n_annulus):
        j = (i + 1) % n_annulus
        triangles += [(cap, ring[j], ring[i]), (ring[i], ring[j], lower[j]),
                      (ring[i], lower[j], lower[i]), (lower[i], lower[j], apex)]
    total = sum(np.dot(a, np.cross(b, c)) for a, b, c in triangles) / 6.0
    return float(abs(total))
