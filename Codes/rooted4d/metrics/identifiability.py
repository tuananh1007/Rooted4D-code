"""Dependency-light local identifiability summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IdentifiabilitySummary:
    rows: int
    columns: int
    effective_rank: int
    smallest_nonzero_singular_value: float
    largest_singular_value: float
    condition_number: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "effective_rank": self.effective_rank,
            "smallest_nonzero_singular_value": self.smallest_nonzero_singular_value,
            "largest_singular_value": self.largest_singular_value,
            "condition_number": self.condition_number,
        }


def summarize_jacobian(jacobian: np.ndarray, rtol: float = 1e-8) -> IdentifiabilitySummary:
    matrix = np.asarray(jacobian, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("jacobian must be two-dimensional")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    threshold = rtol * largest
    nonzero = singular_values[singular_values > threshold]
    rank = int(nonzero.size)
    smallest = float(nonzero[-1]) if rank else 0.0
    condition = float(largest / smallest) if smallest else float("inf")
    return IdentifiabilitySummary(
        rows=matrix.shape[0],
        columns=matrix.shape[1],
        effective_rank=rank,
        smallest_nonzero_singular_value=smallest,
        largest_singular_value=largest,
        condition_number=condition,
    )

