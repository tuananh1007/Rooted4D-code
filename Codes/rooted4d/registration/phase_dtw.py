"""Circular dynamic-time-warp alignment from observed periodic features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CircularPhaseAlignment:
    target_to_source_phase: np.ndarray
    target_to_source_unwrapped: np.ndarray
    offset_cycles: float
    normalized_cost: float
    path: np.ndarray
    target_shift: int


def _features(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or len(array) < 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite shape (phases, features) with >=3 phases")
    return array


def _dtw(left: np.ndarray, right: np.ndarray, band: int) -> tuple[float, np.ndarray]:
    n_left, n_right = len(left), len(right)
    cumulative = np.full((n_left + 1, n_right + 1), np.inf, dtype=np.float64)
    predecessor = np.full((n_left + 1, n_right + 1), -1, dtype=np.int8)
    cumulative[0, 0] = 0.0
    for i in range(1, n_left + 1):
        center = int(round(i * n_right / n_left))
        lower = max(1, center - band)
        upper = min(n_right, center + band)
        for j in range(lower, upper + 1):
            local = float(np.sum((left[i - 1] - right[j - 1]) ** 2))
            candidates = (
                cumulative[i - 1, j - 1],
                cumulative[i - 1, j],
                cumulative[i, j - 1],
            )
            move = int(np.argmin(candidates))
            cumulative[i, j] = local + candidates[move]
            predecessor[i, j] = move
    if not np.isfinite(cumulative[n_left, n_right]):
        raise RuntimeError("DTW band is too narrow for the two sequences")
    path: list[tuple[int, int]] = []
    i, j = n_left, n_right
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            path.append((i - 1, j - 1))
            move = predecessor[i, j]
            if move == 0:
                i -= 1
                j -= 1
            elif move == 1:
                i -= 1
            elif move == 2:
                j -= 1
            else:
                raise RuntimeError("Invalid DTW predecessor")
    path.reverse()
    result = np.asarray(path, dtype=np.int64)
    return float(cumulative[n_left, n_right] / max(len(result), 1)), result


def align_periodic_features(
    source_features: np.ndarray,
    target_features: np.ndarray,
    *,
    maximum_shift_fraction: float = 0.5,
    band_fraction: float = 0.25,
) -> CircularPhaseAlignment:
    """Map target acquisition phases onto source phases using only observed features.

    A circular shift is enumerated, then each candidate is aligned with banded
    dynamic time warping. The returned map is circular and monotonically
    unwrapped. No ECG truth or synthetic phase parameters are consumed.
    """

    source = _features(source_features, "source_features")
    target = _features(target_features, "target_features")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target must have the same feature dimension")
    if not 0.0 <= maximum_shift_fraction <= 0.5:
        raise ValueError("maximum_shift_fraction must lie in [0, 0.5]")
    if band_fraction <= 0.0:
        raise ValueError("band_fraction must be positive")
    pooled = np.concatenate((source, target), axis=0)
    scale = pooled.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    center = pooled.mean(axis=0)
    source_normalized = (source - center) / scale
    target_normalized = (target - center) / scale

    maximum_shift = int(np.floor(maximum_shift_fraction * len(target)))
    shifts = range(-maximum_shift, maximum_shift + 1)
    band = max(abs(len(source) - len(target)), int(np.ceil(band_fraction * max(len(source), len(target)))))
    best: tuple[float, int, np.ndarray] | None = None
    for shift in shifts:
        rolled = np.roll(target_normalized, -shift, axis=0)
        cost, path = _dtw(source_normalized, rolled, band)
        candidate = (cost, abs(shift), shift)
        if best is None or candidate < (best[0], abs(best[1]), best[1]):
            best = (cost, shift, path)
    assert best is not None
    cost, shift, rolled_path = best

    mapped_rolled = np.empty(len(target), dtype=np.float64)
    for rolled_index in range(len(target)):
        source_indices = rolled_path[rolled_path[:, 1] == rolled_index, 0]
        if len(source_indices) == 0:
            raise RuntimeError("DTW path did not cover every target phase")
        mapped_rolled[rolled_index] = float(np.median(source_indices) / len(source))
    mapping = np.empty_like(mapped_rolled)
    for rolled_index, phase in enumerate(mapped_rolled):
        original_index = (rolled_index + shift) % len(target)
        mapping[original_index] = phase
    unwrapped = np.unwrap(2.0 * np.pi * mapping, discont=np.pi) / (2.0 * np.pi)
    for index in range(1, len(unwrapped)):
        while unwrapped[index] < unwrapped[index - 1] - 1e-12:
            unwrapped[index] += 1.0
        if unwrapped[index] <= unwrapped[index - 1]:
            unwrapped[index] = unwrapped[index - 1] + 1e-9
    offset = float(np.mod(unwrapped[0], 1.0))
    return CircularPhaseAlignment(
        target_to_source_phase=np.mod(unwrapped, 1.0),
        target_to_source_unwrapped=unwrapped,
        offset_cycles=offset,
        normalized_cost=cost,
        path=rolled_path,
        target_shift=shift,
    )
