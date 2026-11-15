r"""Clinical quantities the plane selector is trying to pin down.

Selecting planes to reduce uncertainty in *parameters* is the wrong objective: a
plane that determines coefficients nobody reports has bought nothing. The
criterion therefore targets the quantities a report actually contains. Ejection
fraction is the primary one, and is also the endpoint CUTE-MRI uses, which keeps
the comparison on the same footing.

Volumes come from the divergence theorem on the closed chamber surface, so they
are exact for the mesh rather than estimated by the method of disks. That matters
for the gradient: a disk-summation estimate would inject its own discretization
error into the Jacobian and contaminate the very quantity being minimized.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from rooted4d.metrics.topology import signed_enclosed_volume


def chamber_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Enclosed volume of the chamber surface, in cubic millimetres."""

    return abs(signed_enclosed_volume(vertices, faces))


def volume_curve(
    geometry_of: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    parameters: np.ndarray,
    phases: np.ndarray,
) -> np.ndarray:
    """Chamber volume across the cardiac cycle."""

    values = []
    for phase in np.asarray(phases, dtype=np.float64):
        vertices, faces = geometry_of(parameters, float(phase))
        values.append(chamber_volume(vertices, faces))
    return np.asarray(values, dtype=np.float64)


def ejection_fraction(volumes: np.ndarray) -> float:
    """``(EDV - ESV) / EDV`` as a fraction, from a sampled volume curve."""

    curve = np.asarray(volumes, dtype=np.float64)
    if curve.size < 2:
        raise ValueError("at least two phases are required")
    end_diastolic = float(np.max(curve))
    end_systolic = float(np.min(curve))
    if end_diastolic <= 0.0:
        raise ValueError("end-diastolic volume must be positive")
    return (end_diastolic - end_systolic) / end_diastolic


def clinical_quantities(
    geometry_of: Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]],
    parameters: np.ndarray,
    phases: np.ndarray,
) -> dict[str, float]:
    """End-diastolic volume, end-systolic volume, stroke volume, and EF."""

    curve = volume_curve(geometry_of, parameters, phases)
    end_diastolic = float(np.max(curve))
    end_systolic = float(np.min(curve))
    return {
        "edv_mm3": end_diastolic,
        "esv_mm3": end_systolic,
        "stroke_volume_mm3": end_diastolic - end_systolic,
        "ejection_fraction": ejection_fraction(curve),
    }


def quantity_gradients(
    geometry_of: Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]],
    parameters: np.ndarray,
    phases: np.ndarray,
    *,
    names: tuple[str, ...] = ("edv_mm3", "esv_mm3", "ejection_fraction"),
    step: float = 1e-4,
) -> tuple[np.ndarray, dict[str, float]]:
    """Central-difference gradients of each quantity with respect to the latent.

    Returns the gradient matrix, one row per quantity, and the quantity values at
    the linearization point. The values are needed to normalize: an ejection
    fraction near 0.6 and a volume near 10^5 cubic millimetres cannot be summed
    into one objective without scaling.
    """

    theta = np.asarray(parameters, dtype=np.float64)
    base = clinical_quantities(geometry_of, theta, phases)
    rows = []
    for index in range(theta.size):
        offset = np.zeros_like(theta)
        offset[index] = step
        forward = clinical_quantities(geometry_of, theta + offset, phases)
        backward = clinical_quantities(geometry_of, theta - offset, phases)
        rows.append(
            [(forward[name] - backward[name]) / (2.0 * step) for name in names]
        )
    gradients = np.asarray(rows, dtype=np.float64).T
    return gradients, {name: base[name] for name in names}


def relative_weights(values: dict[str, float], names: tuple[str, ...]) -> np.ndarray:
    """Weights making each quantity's variance comparable in relative terms.

    Without this the objective is dominated by whichever quantity has the largest
    units, and the selector would optimize volume precision while ignoring
    ejection fraction entirely.
    """

    return np.asarray(
        [1.0 / max(abs(values[name]), 1e-9) ** 2 for name in names], dtype=np.float64
    )
