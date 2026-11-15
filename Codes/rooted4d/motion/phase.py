"""Monotone phase warping on the cardiac cycle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _softplus(values: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, values)


@dataclass(frozen=True)
class MonotoneCircularPhaseWarp:
    """Piecewise-linear circular warp with strictly positive increments."""

    logits: np.ndarray
    offset: float = 0.0
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        logits = np.asarray(self.logits, dtype=np.float64)
        if logits.ndim != 1 or logits.size < 2:
            raise ValueError("logits must be a one-dimensional array with at least two entries")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "offset", float(self.offset % 1.0))

    @property
    def increments(self) -> np.ndarray:
        positive = _softplus(self.logits) + self.epsilon
        return positive / positive.sum()

    @property
    def source_knots(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.logits.size + 1)

    @property
    def target_knots(self) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(self.increments)))

    def evaluate_unwrapped(self, normalized_time: np.ndarray | float) -> np.ndarray:
        time = np.asarray(normalized_time, dtype=np.float64)
        if np.any(time < 0.0) or np.any(time > 1.0):
            raise ValueError("normalized_time must lie in [0, 1]")
        base = np.interp(time, self.source_knots, self.target_knots)
        return self.offset + base

    def evaluate(self, normalized_time: np.ndarray | float) -> np.ndarray:
        return np.mod(self.evaluate_unwrapped(normalized_time), 1.0)

    def minimum_increment(self) -> float:
        return float(self.increments.min())


def circular_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Shortest absolute distance in cycles."""

    difference = np.mod(np.asarray(left) - np.asarray(right) + 0.5, 1.0) - 0.5
    return np.abs(difference)

