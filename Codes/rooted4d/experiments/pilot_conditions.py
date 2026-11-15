"""Frozen evidence conditions for the Direction 1 inference pilot.

The pilot asks whether a budgeted inference rule protects reliable component
anatomy from weak or corrupted evidence. That question only has content if the
corruption is generated the same way every time and never derived from the
truth, so the magnitudes and seeds below are fixed here rather than passed in at
the call site, and every generator obeys three rules:

*   **Held-out evidence is never touched.** ``Observations`` carries an explicit
    held-out partition, and the pilot scores on it. Corrupting it would score the
    method on the damage instead of on the anatomy.
*   **No truth, no labels.** A generator receives observations and a seed. It
    never sees ``problem.truth``, and the solver is never told which modality was
    corrupted or by how much. Reliability has to be inferred from
    solver-available information or not claimed at all.
*   **Coherent, not i.i.d.** A per-point Gaussian perturbation is absorbed by the
    observation noise model and tests nothing. The failure this pilot is about is
    a *coherent* displacement of a connected region, which is also the case where
    disagreement between modalities cannot say which one is wrong.

The patch is chosen from the modality's own observed mean geometry, which the
solver also has, so the condition is reproducible from the evidence alone.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from rooted4d.inference.joint import Observations

#: Fixed in advance, per the pilot contract. Changing these invalidates a run.
CORRUPTION_SEED = 20260917
LOCALIZED_PATCH_FRACTION = 0.25
LOCALIZED_BIAS_MM = 3.0
MISSING_VERTEX_FRACTION = 0.35
#: MITEA scan-rescan measures 2.90 mm rms static between-session discrepancy, so
#: the session stress uses the real magnitude rather than an invented one.
SESSION_SHIFT_MM = 2.90

_MODALITIES = ("cmr", "tee")


def _fields(modality: str) -> tuple[str, str, str]:
    if modality not in _MODALITIES:
        raise ValueError("modality must be one of %r, got %r" % (_MODALITIES, modality))
    return "%s_times" % modality, "%s_vertices" % modality, "%s_points" % modality


def _observed_mean_geometry(points: np.ndarray) -> np.ndarray:
    """Mean position of each observed vertex over frames, from evidence alone."""

    return np.asarray(points, dtype=np.float64).mean(axis=0)


def localized_corruption(
    observations: Observations,
    modality: str,
    *,
    seed: int = CORRUPTION_SEED,
    fraction: float = LOCALIZED_PATCH_FRACTION,
    bias_mm: float = LOCALIZED_BIAS_MM,
) -> tuple[Observations, dict[str, Any]]:
    """Displace a connected patch of one modality's observed points.

    The patch is the ``fraction`` of observed vertices nearest a seed vertex in
    that modality's own observed mean geometry, and the whole patch is moved by
    one constant vector in every observed frame. Held-out points are untouched,
    so the clean target the pilot scores against is unaffected.
    """

    times_field, vertices_field, points_field = _fields(modality)
    points = np.asarray(getattr(observations, points_field), dtype=np.float64)
    if points.size == 0:
        return observations, {"condition": "localized", "modality": modality,
                              "applied": False, "reason": "no observed points"}

    rng = np.random.default_rng(seed)
    geometry = _observed_mean_geometry(points)
    n_vertices = geometry.shape[0]
    count = max(1, int(round(fraction * n_vertices)))
    centre = int(rng.integers(n_vertices))
    order = np.argsort(np.linalg.norm(geometry - geometry[centre], axis=1))
    patch = np.sort(order[:count])

    direction = rng.normal(size=3)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    corrupted = points.copy()
    corrupted[:, patch, :] += bias_mm * direction

    record = {
        "condition": "localized",
        "modality": modality,
        "applied": True,
        "seed": int(seed),
        "patch_vertices": int(count),
        "observed_vertices": int(n_vertices),
        "bias_mm": float(bias_mm),
        "direction": [float(v) for v in direction],
        "centre_index": centre,
    }
    return replace(observations, **{points_field: corrupted}), record


def missing_views(
    observations: Observations,
    modality: str,
    *,
    seed: int = CORRUPTION_SEED,
    fraction: float = MISSING_VERTEX_FRACTION,
) -> tuple[Observations, dict[str, Any]]:
    """Remove a fraction of one modality's observed vertices.

    Removal, not perturbation: the columns disappear from the evidence, which is
    what an absent view produces. Held-out vertices are a separate partition and
    keep their columns, so the scoring target is unchanged.
    """

    times_field, vertices_field, points_field = _fields(modality)
    vertices = np.asarray(getattr(observations, vertices_field))
    points = np.asarray(getattr(observations, points_field), dtype=np.float64)
    if vertices.size == 0:
        return observations, {"condition": "missing", "modality": modality,
                              "applied": False, "reason": "no observed vertices"}

    rng = np.random.default_rng(seed + 1)
    keep_count = max(1, int(round((1.0 - fraction) * vertices.size)))
    keep = np.sort(rng.permutation(vertices.size)[:keep_count])
    record = {
        "condition": "missing",
        "modality": modality,
        "applied": True,
        "seed": int(seed + 1),
        "kept_vertices": int(keep_count),
        "observed_vertices": int(vertices.size),
        "removed_fraction": float(1.0 - keep_count / vertices.size),
    }
    updated = {vertices_field: vertices[keep], points_field: points[:, keep, :]}
    return replace(observations, **updated), record


def session_shift(
    observations: Observations,
    modality: str,
    *,
    seed: int = CORRUPTION_SEED,
    magnitude_mm: float = SESSION_SHIFT_MM,
) -> tuple[Observations, dict[str, Any]]:
    """Offset one modality's observed points by a constant vector.

    A static between-session discrepancy, which is a different assumption from
    localized corruption: it is global and the session term is meant to be able
    to represent it. The contract keeps this condition reported separately and
    never pooled into the primary corruption result.
    """

    times_field, vertices_field, points_field = _fields(modality)
    points = np.asarray(getattr(observations, points_field), dtype=np.float64)
    if points.size == 0:
        return observations, {"condition": "session_shift", "modality": modality,
                              "applied": False, "reason": "no observed points"}

    rng = np.random.default_rng(seed + 2)
    direction = rng.normal(size=3)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    record = {
        "condition": "session_shift",
        "modality": modality,
        "applied": True,
        "seed": int(seed + 2),
        "magnitude_mm": float(magnitude_mm),
        "direction": [float(v) for v in direction],
    }
    return replace(observations, **{points_field: points + magnitude_mm * direction}), record


CONDITIONS = {
    "nominal": None,
    "localized": localized_corruption,
    "missing": missing_views,
    "session_shift": session_shift,
}


def apply_condition(
    observations: Observations,
    condition: str,
    modality: str | None,
    *,
    seed: int = CORRUPTION_SEED,
) -> tuple[Observations, dict[str, Any]]:
    """Dispatch to one generator, or return the evidence unchanged."""

    if condition == "nominal":
        return observations, {"condition": "nominal", "applied": False}
    if condition not in CONDITIONS:
        raise ValueError("unknown condition %r; expected one of %r"
                         % (condition, sorted(CONDITIONS)))
    if modality is None:
        raise ValueError("condition %r needs a modality to act on" % condition)
    return CONDITIONS[condition](observations, modality, seed=seed)


def heldout_is_untouched(before: Observations, after: Observations) -> bool:
    """Guard used by the pilot harness and by the tests.

    The scoring target must survive every condition exactly. This is checked
    rather than assumed, because a generator that quietly corrupted the held-out
    partition would improve every arm's apparent behaviour at once and look like
    a method effect.
    """

    for field in (
        "heldout_cmr_times", "heldout_cmr_vertices", "heldout_cmr_points",
        "heldout_tee_times", "heldout_tee_vertices", "heldout_tee_points",
    ):
        if not np.array_equal(getattr(before, field), getattr(after, field)):
            return False
    return True
