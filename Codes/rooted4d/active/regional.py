r"""AHA-style regional quantities, where plane coverage genuinely binds.

Ejection fraction is a global volume ratio, and one well-placed long-axis plane
determines it. That is why every acquisition strategy tied on it: there was
nothing for a second plane to add. Regional function is structurally different.
No single plane sees all segments, a short-axis plane covers one ring level's
sectors and a long-axis plane two opposing sectors across levels, so coverage has
to be assembled from several planes and which ones matter depends on where the
abnormality is.

Segments follow the AHA convention's structure: basal and mid thirds divided into
six circumferential sectors each, apical third into four, sixteen in total. The
apical cap segment is omitted because a cavity surface fitted from short-axis
slices does not resolve it.

The quantity per segment is endocardial radial excursion between end diastole and
end systole, which is the regional wall-motion proxy section 7.4 lists. It is a
proxy: true wall motion needs the epicardial surface too, and this uses the
cavity boundary only.
"""

from __future__ import annotations

import numpy as np

N_SEGMENTS = 16


def segment_assignment(n_rings: int, n_angles: int) -> np.ndarray:
    """Map each ring-and-angle vertex to an AHA-style segment index.

    Returns ``-1`` for the two cap vertices, which belong to no segment.
    """

    if n_rings < 3:
        raise ValueError("at least three rings are needed for basal, mid and apical")
    assignment = np.full(n_rings * n_angles + 2, -1, dtype=np.int64)
    basal_end = n_rings // 3
    mid_end = 2 * n_rings // 3
    for ring in range(n_rings):
        if ring < basal_end:
            level, sectors, offset = 0, 6, 0
        elif ring < mid_end:
            level, sectors, offset = 1, 6, 6
        else:
            level, sectors, offset = 2, 4, 12
        for angle in range(n_angles):
            sector = int(angle * sectors / n_angles) % sectors
            assignment[ring * n_angles + angle] = offset + sector
    return assignment


def regional_excursion(
    end_diastolic: np.ndarray,
    end_systolic: np.ndarray,
    assignment: np.ndarray,
) -> np.ndarray:
    """Mean endocardial radial excursion per segment, in millimetres.

    Radii are taken about each phase's own long-axis centre so the measure
    reflects wall motion rather than whole-heart translation between phases.
    """

    values = np.empty(N_SEGMENTS, dtype=np.float64)
    diastolic_centre = end_diastolic[:, :2].mean(axis=0)
    systolic_centre = end_systolic[:, :2].mean(axis=0)
    diastolic_radius = np.linalg.norm(end_diastolic[:, :2] - diastolic_centre, axis=1)
    systolic_radius = np.linalg.norm(end_systolic[:, :2] - systolic_centre, axis=1)
    for segment in range(N_SEGMENTS):
        mask = assignment == segment
        if not mask.any():
            values[segment] = 0.0
            continue
        values[segment] = float(
            diastolic_radius[mask].mean() - systolic_radius[mask].mean()
        )
    return values


def regional_gradients(
    model,
    theta: np.ndarray,
    assignment: np.ndarray,
    *,
    step: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Jacobian of the sixteen segment excursions with respect to the latent."""

    n_identity = model.n_identity

    def excursion(vector: np.ndarray) -> np.ndarray:
        return regional_excursion(
            model.shape_at(vector[:n_identity], vector[n_identity:], 0.0),
            model.shape_at(vector[:n_identity], vector[n_identity:], 1.0),
            assignment,
        )

    base = excursion(theta)
    rows = []
    for index in range(theta.size):
        offset = np.zeros_like(theta)
        offset[index] = step
        rows.append((excursion(theta + offset) - excursion(theta - offset)) / (2.0 * step))
    return np.stack(rows, axis=1), base
