"""End-to-end non-oracle pose/phase initialization from segmented annular sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rooted4d.geometry import RigidTransform
from rooted4d.registration import (
    CircularPhaseAlignment,
    RegistrationConfig,
    RegistrationResult,
    align_periodic_features,
    register_point_sets,
)


def _curve_sequence(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 3 or array.shape[1] < 6:
        raise ValueError(f"{name} must have shape (frames, ordered_annular_points>=6, 3)")
    if array.shape[0] < 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain at least three finite frames")
    return array


def _resample_closed_curve(curve: np.ndarray, count: int) -> np.ndarray:
    if count < 8:
        raise ValueError("resampled annulus must contain at least eight points")
    edges = np.roll(curve, -1, axis=0) - curve
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 1e-9):
        raise ValueError("annular curve contains a duplicate or collapsed edge")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    targets = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    indices = np.searchsorted(cumulative, targets, side="right") - 1
    indices = np.clip(indices, 0, len(curve) - 1)
    local = (targets - cumulative[indices]) / lengths[indices]
    following = (indices + 1) % len(curve)
    return curve[indices] + local[:, None] * (curve[following] - curve[indices])


def annular_frame_features(sequence: np.ndarray) -> np.ndarray:
    """Rigid-invariant observed features used for circular temporal alignment."""

    values = _curve_sequence(sequence, "sequence")
    features = []
    for curve in values:
        centered = curve - curve.mean(axis=0)
        perimeter = float(np.sum(np.linalg.norm(np.roll(curve, -1, axis=0) - curve, axis=1)))
        covariance = centered.T @ centered / len(centered)
        eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
        area_vector = 0.5 * np.sum(np.cross(centered, np.roll(centered, -1, axis=0)), axis=0)
        area = float(np.linalg.norm(area_vector))
        radii = np.linalg.norm(centered, axis=1)
        features.append(
            [
                perimeter,
                area,
                float(eigenvalues[0]),
                float(eigenvalues[1]),
                float(np.max(radii) - np.min(radii)),
            ]
        )
    return np.asarray(features, dtype=np.float64)


@dataclass(frozen=True)
class CrossModalAlignmentResult:
    cmr_to_tee: RigidTransform
    spatial: RegistrationResult
    temporal: CircularPhaseAlignment
    canonical_resample_count: int


def infer_cross_modal_alignment(
    cmr_annulus_sequence: np.ndarray,
    tee_annulus_sequence: np.ndarray,
    *,
    canonical_resample_count: int = 64,
    registration_config: RegistrationConfig = RegistrationConfig(
        max_iterations=60,
        distance_scale_mm=6.0,
        normal_weight=0.0,
        arc_weight=6.0,
        trim_fraction=0.10,
    ),
) -> CrossModalAlignmentResult:
    """Infer CMR-to-TEE pose and target-to-source phase from observed annular curves.

    Inputs are segmented, ordered annular curves anchored at the same anatomical
    commissure and traversed in the same canonical direction. This is information a
    clinical annotation/initializer can supply; no hidden pose, phase, generator latent,
    or vertex identity is consumed.
    """

    cmr = _curve_sequence(cmr_annulus_sequence, "cmr_annulus_sequence")
    tee = _curve_sequence(tee_annulus_sequence, "tee_annulus_sequence")
    cmr_mean = _resample_closed_curve(cmr.mean(axis=0), canonical_resample_count)
    tee_mean = _resample_closed_curve(tee.mean(axis=0), canonical_resample_count)
    coordinates = np.arange(canonical_resample_count, dtype=np.float64) / canonical_resample_count
    landmark_indices = np.asarray(
        [0, canonical_resample_count // 4, canonical_resample_count // 2, 3 * canonical_resample_count // 4]
    )
    spatial = register_point_sets(
        cmr_mean,
        tee_mean,
        source_arc_coordinates=coordinates,
        target_arc_coordinates=coordinates,
        source_landmarks=cmr_mean[landmark_indices],
        target_landmarks=tee_mean[landmark_indices],
        config=registration_config,
    )
    temporal = align_periodic_features(
        annular_frame_features(cmr),
        annular_frame_features(tee),
    )
    return CrossModalAlignmentResult(
        cmr_to_tee=spatial.transform,
        spatial=spatial,
        temporal=temporal,
        canonical_resample_count=canonical_resample_count,
    )
