r"""Estimate the spatial correlation length of segmentation error along a contour.

This is the measurement the ActivePlaneTwin findings now hinge on. Whether
sequential per-subject plane selection beats an offline-fixed configuration
depends on whether several planes are genuinely needed, and that in turn depends
on how much independent information one contour carries. Treating contour samples
as independent lets a single plane saturate any precision target, at which point
every acquisition strategy ties and the comparison says nothing.

The quantity wanted is the correlation length of the *residual* between two
delineations of the same boundary: two expert observers, or an expert against an
automated segmentation. A residual that is independent from point to point means
a contour carries as many measurements as it has samples. A residual correlated
over a long arc means it carries far fewer.

Everything here operates on contours already extracted as ordered points, so it
imposes no dependency on a DICOM or NIfTI reader.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def resample_closed_contour(points: np.ndarray, n_samples: int) -> np.ndarray:
    """Resample a closed contour to uniform arclength spacing.

    Two delineations of the same boundary generally have different point counts
    and different starting points. Comparing them requires a common
    parameterization, and arclength is the one that does not privilege regions
    where one annotator happened to click more often.
    """

    curve = np.asarray(points, dtype=np.float64)
    if curve.ndim != 2 or curve.shape[1] != 2:
        raise ValueError("points must have shape (points, 2)")
    if len(curve) < 3:
        raise ValueError("a closed contour needs at least three points")
    if n_samples < 3:
        raise ValueError("n_samples must be at least three")
    closed = np.vstack([curve, curve[:1]])
    steps = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    if float(steps.sum()) <= 0.0:
        raise ValueError("contour has zero total length")
    arclength = np.concatenate([[0.0], np.cumsum(steps)])
    targets = np.linspace(0.0, arclength[-1], n_samples, endpoint=False)
    return np.stack(
        [np.interp(targets, arclength, closed[:, axis]) for axis in range(2)], axis=1
    )


def outward_normals(contour: np.ndarray) -> np.ndarray:
    """Unit outward normals of a closed contour, from centred differences."""

    curve = np.asarray(contour, dtype=np.float64)
    tangents = np.roll(curve, -1, axis=0) - np.roll(curve, 1, axis=0)
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(lengths, 1e-12)
    normals = np.stack([tangents[:, 1], -tangents[:, 0]], axis=1)
    centre = curve.mean(axis=0)
    outward = curve - centre
    flip = np.sign(np.sum(normals * outward, axis=1, keepdims=True))
    flip[flip == 0.0] = 1.0
    return normals * flip


def contour_residuals(
    reference: np.ndarray, comparison: np.ndarray, *, n_samples: int = 128
) -> np.ndarray:
    """Signed normal-direction residuals of one delineation against another.

    Signed rather than absolute, because the sign is the whole point: a
    systematic outward bias over an arc is exactly the correlated error that
    makes a contour carry less information than its sample count suggests.
    """

    first = resample_closed_contour(reference, n_samples)
    second = resample_closed_contour(comparison, n_samples)
    normals = outward_normals(first)
    # Nearest point on the comparison contour, projected onto the reference normal.
    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=-1)
    nearest = second[np.argmin(distances, axis=1)]
    return np.sum((nearest - first) * normals, axis=1)


def circular_autocorrelation(residuals: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Autocorrelation of a residual sequence on a closed contour.

    Circular because the contour is closed: lag wraps around rather than running
    off the end, and treating it otherwise would bias the long lags.
    """

    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 1 or values.size < 4:
        raise ValueError("residuals must be a one-dimensional array of at least four points")
    centred = values - values.mean()
    variance = float(np.mean(centred**2))
    if variance <= 0.0:
        raise ValueError("residuals have zero variance; correlation is undefined")
    limit = values.size // 2 if max_lag is None else int(max_lag)
    limit = max(1, min(limit, values.size - 1))
    return np.array(
        [float(np.mean(centred * np.roll(centred, -lag)) / variance) for lag in range(limit + 1)]
    )


@dataclass(frozen=True)
class CorrelationEstimate:
    """Correlation length of contour residuals, in samples and in millimetres."""

    correlation_length_samples: float
    correlation_length_mm: float | None
    residual_sd: float
    n_contours: int
    n_samples: int
    effective_independent_samples: float

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "correlation_length_samples": self.correlation_length_samples,
            "correlation_length_mm": self.correlation_length_mm,
            "residual_sd": self.residual_sd,
            "n_contours": self.n_contours,
            "n_samples": self.n_samples,
            "effective_independent_samples": self.effective_independent_samples,
        }


def _circular_exponential_kernel(n_samples: int, correlation_length: float) -> np.ndarray:
    index = np.arange(n_samples)
    separation = np.abs(index[:, None] - index[None, :])
    separation = np.minimum(separation, n_samples - separation)
    return np.exp(-separation / correlation_length) + 1e-8 * np.eye(n_samples)


def _profile_log_likelihood(
    residual_sets: list[np.ndarray], correlation_length: float
) -> float:
    """Gaussian log-likelihood with the amplitude profiled out."""

    n_samples = residual_sets[0].size
    kernel = _circular_exponential_kernel(n_samples, correlation_length)
    sign, log_determinant = np.linalg.slogdet(kernel)
    if sign <= 0:
        return -np.inf
    factor = np.linalg.cholesky(kernel)
    total = 0.0
    for residuals in residual_sets:
        solved = np.linalg.solve(factor, residuals)
        quadratic = float(solved @ solved)
        if quadratic <= 0.0:
            return -np.inf
        total += -0.5 * (n_samples * np.log(quadratic / n_samples) + log_determinant)
    return total


def estimate_correlation_length(
    residual_sets: list[np.ndarray],
    *,
    sample_spacing_mm: float | None = None,
    max_lag: int | None = None,
    search_grid: np.ndarray | None = None,
) -> CorrelationEstimate:
    """Maximum-likelihood fit of a circular exponential correlation kernel.

    An autocorrelation-and-log-fit estimator was tried first and is biased badly
    low: subtracting the sample mean removes genuine correlated structure, and
    for a strongly correlated series the sample mean is itself a poor estimate,
    so the bias grows exactly where the answer matters. Against known ground
    truth that estimator returned 5.9 for a true correlation length of 20.

    Maximum likelihood over the kernel avoids the problem because it never forms
    a sample autocorrelation. The residuals are modelled as zero mean; a constant
    delineation bias is therefore absorbed as very long correlation, which is the
    right behaviour, since a uniform offset over a whole contour is perfectly
    correlated error.

    ``effective_independent_samples`` is what the answer is for: a contour of
    ``n`` samples with correlation length ``L`` carries roughly ``n / (1 + 2L)``
    independent measurements.
    """

    if not residual_sets:
        raise ValueError("at least one residual sequence is required")
    lengths = {len(residuals) for residuals in residual_sets}
    if len(lengths) != 1:
        raise ValueError("all residual sequences must have the same length")
    n_samples = lengths.pop()
    if n_samples < 4:
        raise ValueError("at least four samples per contour are required")

    grid = (
        np.geomspace(0.05, max(n_samples / 3.0, 1.0), 60)
        if search_grid is None
        else np.asarray(search_grid, dtype=np.float64)
    )
    scores = [_profile_log_likelihood(residual_sets, float(value)) for value in grid]
    correlation_length = float(grid[int(np.argmax(scores))])

    residual_sd = float(np.mean([np.std(r) for r in residual_sets]))
    effective = n_samples / (1.0 + 2.0 * max(correlation_length, 0.0))
    return CorrelationEstimate(
        correlation_length_samples=correlation_length,
        correlation_length_mm=(
            correlation_length * sample_spacing_mm if sample_spacing_mm else None
        ),
        residual_sd=residual_sd,
        n_contours=len(residual_sets),
        n_samples=n_samples,
        effective_independent_samples=float(effective),
    )


def synthetic_correlated_residuals(
    n_samples: int,
    correlation_length: float,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw residuals with a known circular exponential correlation.

    Used to verify the estimator against ground truth before it is pointed at
    real delineations, so that a surprising number from real data can be
    attributed to the data rather than to the estimator.
    """

    if correlation_length <= 0.0:
        return amplitude * rng.normal(size=n_samples)
    index = np.arange(n_samples)
    separation = np.minimum(index[:, None] - index[None, :] % n_samples, 0)
    separation = np.abs(index[:, None] - index[None, :])
    separation = np.minimum(separation, n_samples - separation)
    kernel = np.exp(-separation / correlation_length) + 1e-8 * np.eye(n_samples)
    factor = np.linalg.cholesky(kernel)
    return amplitude * (factor @ rng.normal(size=n_samples))


def mask_boundary_contour(
    mask: np.ndarray, *, spacing_mm: tuple[float, float] = (1.0, 1.0)
) -> np.ndarray:
    """Ordered boundary contour of a 2D binary mask, in millimetres.

    Boundary pixels are ordered by angle about the centroid, which is correct for
    a star-shaped region and therefore fine for a left-ventricular cavity, the
    case this is for. It is *not* general: a horseshoe-shaped myocardium or a
    region with a concave inlet would be mis-ordered, and such labels need a
    proper marching-squares tracer instead. The assumption is stated here rather
    than buried because using this on the wrong label would silently produce a
    scrambled contour and a meaningless correlation length.
    """

    binary = np.asarray(mask).astype(bool)
    if binary.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if not binary.any():
        raise ValueError("mask is empty")
    padded = np.pad(binary, 1)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    boundary = binary & ~interior
    rows, columns = np.nonzero(boundary)
    if rows.size < 3:
        raise ValueError("mask boundary has fewer than three pixels")
    points = np.stack([rows * spacing_mm[0], columns * spacing_mm[1]], axis=1)
    centre = points.mean(axis=0)
    order = np.argsort(np.arctan2(points[:, 0] - centre[0], points[:, 1] - centre[1]))
    return points[order]


def residuals_from_mask_pairs(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    spacing_mm: tuple[float, float] = (1.0, 1.0),
    n_samples: int = 128,
) -> list[np.ndarray]:
    """Contour residuals for each pair of delineations of the same slice."""

    residuals = []
    for reference, comparison in pairs:
        first = mask_boundary_contour(reference, spacing_mm=spacing_mm)
        second = mask_boundary_contour(comparison, spacing_mm=spacing_mm)
        residuals.append(contour_residuals(first, second, n_samples=n_samples))
    return residuals


def contour_sample_spacing_mm(contour: np.ndarray, n_samples: int) -> float:
    """Arclength between resampled points, so a length in samples becomes millimetres."""

    curve = np.asarray(contour, dtype=np.float64)
    closed = np.vstack([curve, curve[:1]])
    perimeter = float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))
    return perimeter / n_samples
