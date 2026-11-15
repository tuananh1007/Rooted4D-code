r"""Paired statistics required by the preregistered protocol.

Section 8 asks for paired permutation tests for primary method differences and
bootstrap intervals rather than p-values alone, with the resampling unit being
the case rather than the frame or vertex. These helpers implement exactly that
and nothing more; no test is selected after seeing an outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedComparison:
    """One method-versus-method comparison on a paired primary endpoint."""

    n_pairs: int
    median_difference: float
    mean_difference: float
    confidence_interval: tuple[float, float]
    p_value: float
    favours: str

    def as_dict(self) -> dict[str, object]:
        return {
            "n_pairs": self.n_pairs,
            "median_difference": self.median_difference,
            "mean_difference": self.mean_difference,
            "ci_lower": self.confidence_interval[0],
            "ci_upper": self.confidence_interval[1],
            "p_value": self.p_value,
            "favours": self.favours,
            "interval_excludes_zero": bool(
                self.confidence_interval[0] > 0.0 or self.confidence_interval[1] < 0.0
            ),
        }


def paired_permutation_test(
    treatment: np.ndarray,
    reference: np.ndarray,
    *,
    n_permutations: int = 10000,
    seed: int = 0,
) -> float:
    """Two-sided exact-style sign-flip test on paired differences.

    Sign flipping is the correct permutation scheme for paired data: under the
    null the label assignment within a pair is exchangeable, so each difference
    may independently change sign.
    """

    left = np.asarray(treatment, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("treatment and reference must be one-dimensional and paired")
    if left.size < 2:
        raise ValueError("at least two pairs are required")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")

    differences = left - right
    observed = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, differences.size))
    null = signs @ differences / differences.size
    # Add-one smoothing keeps the p-value strictly positive, which is the
    # standard correction for a Monte Carlo permutation test.
    extreme = int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-15))
    return float((extreme + 1) / (n_permutations + 1))


def bootstrap_interval(
    values: np.ndarray,
    *,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean, resampling whole cases."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("values must be one-dimensional with at least two entries")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_resamples, array.size))
    means = array[indices].mean(axis=1)
    tail = 0.5 * (1.0 - confidence)
    return (
        float(np.percentile(means, 100.0 * tail)),
        float(np.percentile(means, 100.0 * (1.0 - tail))),
    )


def compare_paired(
    treatment: np.ndarray,
    reference: np.ndarray,
    *,
    treatment_name: str = "treatment",
    reference_name: str = "reference",
    lower_is_better: bool = True,
    seed: int = 0,
) -> PairedComparison:
    """Compare two arms on one paired endpoint."""

    left = np.asarray(treatment, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    differences = left - right
    mean_difference = float(np.mean(differences))
    if abs(mean_difference) < 1e-15:
        favours = "neither"
    elif (mean_difference < 0.0) == lower_is_better:
        favours = treatment_name
    else:
        favours = reference_name
    return PairedComparison(
        n_pairs=int(left.size),
        median_difference=float(np.median(differences)),
        mean_difference=mean_difference,
        confidence_interval=bootstrap_interval(differences, seed=seed),
        p_value=paired_permutation_test(left, right, seed=seed),
        favours=favours,
    )
