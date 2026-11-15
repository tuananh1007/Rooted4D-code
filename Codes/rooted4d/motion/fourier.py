"""Periodic Fourier trajectories for exact cardiac-cycle closure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def periodic_basis(phases: np.ndarray, order: int) -> np.ndarray:
    """Return ``[cos(2πkτ), sin(2πkτ)]`` columns for ``k=1..order``."""

    if order < 1:
        raise ValueError("order must be at least one")
    tau = np.asarray(phases, dtype=np.float64)
    columns = []
    for harmonic in range(1, order + 1):
        angle = 2.0 * np.pi * harmonic * tau
        columns.extend((np.cos(angle), np.sin(angle)))
    return np.stack(columns, axis=-1)


def interpolation_basis(phases: np.ndarray, n_nodes: int) -> np.ndarray:
    """Linear-interpolation weights over ``n_nodes`` time nodes on ``[0, 1]``.

    This is the non-periodic counterpart of :func:`periodic_basis`, and it exists
    to make Proposition 4 falsifiable rather than assumed. Each node is a free
    parameter and the node at ``tau = 1`` is *separate* from the node at
    ``tau = 0``, so nothing forces the trajectory to close: the seam gap that
    periodicity forbids is exactly the degree of freedom restored here.

    Phases are reduced modulo one because phase lives on the circle regardless
    of which basis is used to represent motion over it. The final node is only
    approached as ``tau -> 1``, never attained, so it is weakly identified. That
    is not a defect in the implementation; it is the cost of giving up
    periodicity, and a model with this basis genuinely has a parameter the data
    barely constrains.
    """

    if n_nodes < 2:
        raise ValueError("n_nodes must be at least two")
    tau = np.asarray(phases, dtype=np.float64)
    reduced = np.mod(tau, 1.0)
    step = 1.0 / (n_nodes - 1)
    flat = reduced.reshape(-1)
    left = np.clip(np.floor(flat / step).astype(np.int64), 0, n_nodes - 2)
    fraction = (flat - left * step) / step
    basis = np.zeros((flat.size, n_nodes), dtype=np.float64)
    rows = np.arange(flat.size)
    basis[rows, left] = 1.0 - fraction
    basis[rows, left + 1] = fraction

    # Remove the constant component, so that this basis represents deviation
    # from the cycle mean exactly as the Fourier basis does. Without it the two
    # bases differ in two ways at once and the ablation is confounded: cosine
    # and sine columns have zero cycle mean, so a prior on Fourier coefficients
    # shrinks only the deviation, whereas interpolation columns form a partition
    # of unity and a prior on node values also shrinks the mean, dragging the
    # whole trajectory toward the static shape. Measured on validation, that
    # asymmetry moved the per-frame arm from 2.87 to 4.72 mm across the same
    # weight grid over which the periodic arm barely moved, so leaving it in
    # would have measured regularization semantics rather than periodicity.
    #
    # The mean of the node vector no longer affects the trajectory, so it is a
    # gauge direction; the prior fixes it at zero harmlessly.
    basis = basis - basis.mean(axis=1, keepdims=True)
    return basis.reshape(*reduced.shape, n_nodes)


@dataclass(frozen=True)
class FourierMotion:
    """A base physical geometry plus periodic per-vertex displacement modes."""

    base: np.ndarray
    coefficients: np.ndarray

    def __post_init__(self) -> None:
        base = np.asarray(self.base, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if base.ndim != 2 or base.shape[1] != 3:
            raise ValueError("base must have shape (vertices, 3)")
        if coefficients.ndim != 3 or coefficients.shape[1:] != base.shape:
            raise ValueError("coefficients must have shape (2 * order, vertices, 3)")
        if coefficients.shape[0] % 2:
            raise ValueError("coefficients require cosine/sine pairs")
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def order(self) -> int:
        return self.coefficients.shape[0] // 2

    def evaluate(self, phases: np.ndarray | float) -> np.ndarray:
        tau = np.asarray(phases, dtype=np.float64)
        basis = periodic_basis(tau, self.order)
        displacement = np.einsum("...b,bvc->...vc", basis, self.coefficients)
        return self.base + displacement

    def seam_error(self) -> float:
        endpoints = self.evaluate(np.array([0.0, 1.0]))
        return float(np.max(np.linalg.norm(endpoints[0] - endpoints[1], axis=-1)))

